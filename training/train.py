import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageDraw
import os
import json
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from tqdm import tqdm
import pickle

nltk.download('punkt', quiet=True)
BASE_DIR = Path(__file__).resolve().parent

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold
    
    def __len__(self):
        return len(self.itos)
    
    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        
        for sentence in sentence_list:
            tokens = word_tokenize(sentence.lower())
            frequencies.update(tokens)
        
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
    
    def numericalize(self, text):
        tokenized_text = word_tokenize(text.lower())
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]

class CaptionDataset(Dataset):
    def __init__(self, image_dir, captions_file, vocab=None, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        with open(captions_file, 'r') as f:
            self.captions_data = json.load(f)
        
        self.images = list(self.captions_data.keys())
        self.captions = list(self.captions_data.values())
        
        if vocab is None:
            vocab = Vocabulary()
            vocab.build_vocabulary(self.captions)
        
        self.vocab = vocab
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_name = self.images[index]
        caption = self.captions[index]
        
        image_path = self.image_dir / image_name
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        
        return image, torch.tensor(numericalized_caption)

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_cnn=False):
        super(EncoderCNN, self).__init__()
        self.train_cnn = train_cnn
        
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
        if not train_cnn:
            for param in self.resnet.parameters():
                param.requires_grad = False
    
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
    def __init__(self, embed_size=256, hidden_size=512, vocab_size=None, num_layers=1):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)
    
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
    
    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []
        
        with torch.no_grad():
            features = self.encoder(image.unsqueeze(0))
            inputs = features.unsqueeze(1)
            states = None
            
            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(inputs, states)
                output = self.decoder.linear(hiddens.squeeze(1))
                predicted = output.argmax(1)
                
                result_caption.append(predicted.item())
                
                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break
                
                inputs = self.decoder.embed(predicted).unsqueeze(1)
        
        return [vocabulary.itos[idx] for idx in result_caption]

def _generate_placeholder_image(path: Path, color: tuple[int, int, int], text: str):
    """Create a simple placeholder image for quick training runs."""
    img = Image.new("RGB", (256, 256), color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([30, 30, 226, 226], outline=(255, 255, 255), width=4)
    draw.text((40, 40), text[:20], fill=(255, 255, 255))
    img.save(path, format="JPEG")


def create_sample_data():
    sample_dir = BASE_DIR / "sample_images"
    captions_path = BASE_DIR / "sample_captions.json"

    sample_captions = {
        "image1.jpg": "A dog playing in the park",
        "image2.jpg": "A beautiful sunset over the ocean",
        "image3.jpg": "People walking on a busy street",
        "image4.jpg": "A cat sleeping on a couch",
        "image5.jpg": "Mountains covered with snow"
    }
    
    sample_dir.mkdir(parents=True, exist_ok=True)
    with open(captions_path, "w") as f:
        json.dump(sample_captions, f, indent=2)

    # Generate simple placeholder images for quick training/testing
    palette = [
        (52, 152, 219),
        (231, 76, 60),
        (46, 204, 113),
        (155, 89, 182),
        (241, 196, 15),
    ]

    for (filename, caption), color in zip(sample_captions.items(), palette):
        image_path = sample_dir / filename
        _generate_placeholder_image(image_path, color, caption)
    
    print(f"Sample data created in {sample_dir}.")

def collate_fn(batch):
    """Custom collate function to handle variable length captions."""
    images = [item[0] for item in batch]
    captions = [item[1] for item in batch]
    
    # Pad captions to the same length
    max_length = max(len(cap) for cap in captions)
    padded_captions = []
    
    for cap in captions:
        padding = torch.zeros(max_length - len(cap), dtype=torch.long)
        padded_cap = torch.cat([cap, padding])
        padded_captions.append(padded_cap)
    
    return torch.stack(images), torch.stack(padded_captions)

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = CaptionDataset(
        image_dir=BASE_DIR / "sample_images",
        captions_file=BASE_DIR / "sample_captions.json",
        transform=transform
    )
    
    vocab_size = len(dataset.vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn, drop_last=True)
    
    model = ImageCaptioningModel(embed_size=256, hidden_size=512, vocab_size=vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    num_epochs = int(os.getenv("NUM_EPOCHS", "10"))
    if os.getenv("CI"):
        num_epochs = min(num_epochs, 1)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for images, captions in progress_bar:
            images = images.to(device)
            captions = captions.to(device)
            
            outputs = model(images, captions)
            
            # The model outputs one more token than input captions
            # We need to align them properly
            targets = captions[:, 1:]  # Remove SOS token -> [batch, seq_len-1]
            outputs = outputs[:, :-1, :]  # Remove last output -> [batch, seq_len-1, vocab_size]
            
            # Check if we need further adjustment
            if outputs.shape[1] != targets.shape[1]:
                if outputs.shape[1] > targets.shape[1]:
                    outputs = outputs[:, :targets.shape[1], :]
                else:
                    targets = targets[:, :outputs.shape[1], :]
            
            # Create mask to ignore padding in loss calculation
            mask = (targets != dataset.vocab.stoi["<PAD>"]).float()
            
            # Calculate loss only on non-padded tokens
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss_per_token = loss_fct(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            loss_per_token = loss_per_token * mask.reshape(-1)
            loss = loss_per_token.sum() / mask.sum()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
    
    models_dir = BASE_DIR.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': dataset.vocab,
        'embed_size': 256,
        'hidden_size': 512
    }, models_dir / "caption_model.pth")
    
    with open(models_dir / "vocab.pkl", "wb") as f:
        pickle.dump(dataset.vocab, f)
    
    print(f"Model saved to {models_dir / 'caption_model.pth'}")
    print(f"Vocabulary saved to {models_dir / 'vocab.pkl'}")
    
    return model, dataset.vocab

def test_inference(model, vocab, transform):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    def predict_caption(image_path):
        image = Image.open(image_path).convert("RGB")
        image = transform(image).to(device)
        
        caption_tokens = model.caption_image(image, vocab)
        caption = " ".join([token for token in caption_tokens if token not in ["<SOS>", "<EOS>", "<PAD>"]])
        
        return caption
    
    return predict_caption

if __name__ == "__main__":
    print("Image Captioning Training Script")
    print("=" * 40)
    
    sample_dir = BASE_DIR / "sample_images"
    captions_path = BASE_DIR / "sample_captions.json"

    if not sample_dir.exists() or not captions_path.exists():
        print("Creating sample data...")
        create_sample_data()
        print("Please add your images to sample_images/ and update sample_captions.json")
        print("Then run this script again.")
    else:
        model, vocab = train_model()
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        predict_caption = test_inference(model, vocab, transform)
        print("Training completed! Model is ready for inference.")
