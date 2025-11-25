import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pickle
import io
import os
import logging
import sys
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    # Use the training Vocabulary if available to match pickle metadata
    from training.train import Vocabulary as TrainingVocabulary
except Exception:
    TrainingVocabulary = None


class Vocabulary:
    """Fallback vocabulary class for inference-only usage."""

    def __init__(self):
        self.itos = {}
        self.stoi = {}

    def __len__(self):
        return len(self.itos)


def _register_safe_globals():
    """Allow torch.load to unpickle Vocabulary objects saved during training."""
    try:
        safe_list = [Vocabulary]
        if TrainingVocabulary is not None:
            safe_list.append(TrainingVocabulary)
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals(safe_list)
        # Some checkpoints may have been saved with __main__.Vocabulary; patch the module.
        main_mod = sys.modules.get("__main__")
        if main_mod and not hasattr(main_mod, "Vocabulary"):
            setattr(main_mod, "Vocabulary", TrainingVocabulary or Vocabulary)
    except Exception as exc:
        logger.warning(f"Could not register safe globals for torch.load: {exc}")


class _VocabularyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Vocabulary":
            if TrainingVocabulary is not None:
                return TrainingVocabulary
            return Vocabulary
        return super().find_class(module, name)

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=None)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
    
    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        
        features = features.reshape(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        embeddings = self.dropout(embeddings)
        
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        
        return outputs

class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)
    
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def _decode_step(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        """Run a single decoder step and return log-probabilities."""
        hiddens, states = self.decoder.lstm(inputs, states)
        output = self.decoder.linear(hiddens.squeeze(1))
        log_probs = torch.log_softmax(output, dim=1)
        return log_probs, states

    def caption_image(self, image, vocabulary, max_length: int = 50):
        """Greedy decoding for caption generation."""
        result_tokens = []

        with torch.no_grad():
            features = self.encoder(image.unsqueeze(0))
            inputs = features.unsqueeze(1)
            states = None

            for _ in range(max_length):
                log_probs, states = self._decode_step(inputs, states)
                predicted = log_probs.argmax(dim=1)
                token_id = predicted.item()
                result_tokens.append(token_id)

                if vocabulary.itos.get(token_id) == "<EOS>":
                    break

                inputs = self.decoder.embed(predicted).unsqueeze(1)

        return [vocabulary.itos.get(idx, "<UNK>") for idx in result_tokens]

    def caption_image_beam_search(self, image, vocabulary, beam_size: int = 3, max_length: int = 50):
        """Simple beam search decoding."""
        start_token = vocabulary.stoi.get("<SOS>")
        end_token = vocabulary.stoi.get("<EOS>")

        if start_token is None or end_token is None:
            return self.caption_image(image, vocabulary, max_length)

        with torch.no_grad():
            features = self.encoder(image.unsqueeze(0))

        sequences = [([start_token], 0.0, None)]

        for _ in range(max_length):
            all_candidates = []

            for tokens, score, states in sequences:
                if tokens[-1] == end_token:
                    all_candidates.append((tokens, score, states))
                    continue

                if len(tokens) == 1:
                    inputs = features.unsqueeze(1)
                else:
                    last_token = torch.tensor([[tokens[-1]]], device=DEVICE)
                    inputs = self.decoder.embed(last_token)

                log_probs, next_states = self._decode_step(inputs, states)
                top_log_probs, top_indices = torch.topk(log_probs, beam_size)

                for log_prob, idx in zip(top_log_probs[0], top_indices[0]):
                    candidate_tokens = tokens + [idx.item()]
                    candidate_score = score + log_prob.item()
                    all_candidates.append((candidate_tokens, candidate_score, next_states))

            sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:beam_size]

            if all(tokens[-1] == end_token for tokens, _, _ in sequences):
                break

        best_tokens = sequences[0][0] if sequences else [start_token, end_token]
        return [vocabulary.itos.get(idx, "<UNK>") for idx in best_tokens]

class ModelLoader:
    def __init__(self, model_path: str = "models/caption_model.pth", vocab_path: str = "models/vocab.pkl"):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.model = None
        self.vocab = None
        self.transform = None
        self.device = DEVICE
        
        self._setup_transforms()
        self._load_model()
    
    def _setup_transforms(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self):
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            if not os.path.exists(self.vocab_path):
                raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_path}")

            _register_safe_globals()

            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

            with open(self.vocab_path, 'rb') as f:
                self.vocab = _VocabularyUnpickler(f).load()
            
            vocab_size = len(self.vocab)
            embed_size = checkpoint.get('embed_size', 256)
            hidden_size = checkpoint.get('hidden_size', 512)
            
            self.model = ImageCaptioningModel(
                embed_size=embed_size,
                hidden_size=hidden_size,
                vocab_size=vocab_size
            ).to(self.device)
            
            missing, unexpected = self.model.load_state_dict(
                checkpoint.get('model_state_dict', {}),
                strict=False
            )
            if missing:
                logger.warning(f"Missing keys when loading model: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys when loading model: {unexpected}")
            self.model.eval()
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Vocabulary size: {vocab_size}")
            logger.info(f"Using device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_tensor = self.transform(image).to(self.device)
            return image_tensor
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def _calculate_confidence(self, caption_tokens: list) -> float:
        try:
            meaningful_tokens = [token for token in caption_tokens 
                               if token not in ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]]
            
            if len(meaningful_tokens) == 0:
                return 0.0
            
            confidence = min(0.9, 0.5 + (len(meaningful_tokens) * 0.05))
            confidence = max(0.1, confidence)
            
            return round(confidence, 3)
            
        except Exception:
            return 0.5
    
    def predict_image(self, image_bytes: bytes, use_beam_search: bool = True) -> Tuple[str, float]:
        try:
            if self.model is None or self.vocab is None:
                raise RuntimeError("Model not loaded")
            
            image_tensor = self._preprocess_image(image_bytes)
            
            if use_beam_search:
                caption_tokens = self.model.caption_image_beam_search(image_tensor, self.vocab)
            else:
                caption_tokens = self.model.caption_image(image_tensor, self.vocab)
            
            caption = " ".join([token for token in caption_tokens 
                              if token not in ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]]).strip()
            
            if not caption:
                caption = "No caption generated"
                confidence = 0.0
            else:
                confidence = self._calculate_confidence(caption_tokens)
            
            return caption, confidence
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

model_loader = None

def get_model_loader() -> ModelLoader:
    global model_loader
    if model_loader is None:
        model_loader = ModelLoader()
    return model_loader

def predict_image(image_bytes: bytes, use_beam_search: bool = True) -> Tuple[str, float]:
    loader = get_model_loader()
    return loader.predict_image(image_bytes, use_beam_search)
