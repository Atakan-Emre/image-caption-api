#!/usr/bin/env python3
"""
COCO Image Captioning Training Script
Trains image captioning model on COCO dataset
"""

import os
import sys
import json
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

class Vocabulary:
    def __init__(self):
        self.itos = {}
        self.stoi = {}
        self.idx = 4  # Start from 4 (0: PAD, 1: SOS, 2: EOS, 3: UNK)
    
    def add_token(self, token):
        if token not in self.stoi:
            self.stoi[token] = self.idx
            self.itos[self.idx] = token
            self.idx += 1
    
    def __len__(self):
        return len(self.itos)
    
    def encode(self, sentence):
        tokens = word_tokenize(sentence.lower())
        return [self.stoi.get(token, 3) for token in tokens]  # 3 = UNK
    
    def decode(self, indices):
        return [self.itos.get(idx, "<UNK>") for idx in indices]

class COCODataset(Dataset):
    def __init__(self, image_dir, captions_file, vocab, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.vocab = vocab
        
        with open(captions_file, 'r') as f:
            self.captions_data = json.load(f)
        
        self.images = list(self.captions_data.keys())
        logger.info(f"Loaded {len(self.images)} images with captions")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = self.image_dir / image_name
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Get captions and encode one randomly
        captions = self.captions_data[image_name]
        caption = np.random.choice(captions)
        
        # Encode caption
        tokens = [self.vocab.stoi['<SOS>']]
        tokens.extend(self.vocab.encode(caption))
        tokens.append(self.vocab.stoi['<EOS>'])
        
        caption_tensor = torch.tensor(tokens, dtype=torch.long)
        
        return image, caption_tensor

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-1]  # Remove last fc layer
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
        # Freeze early layers
        for param in list(self.resnet.parameters())[: -20]:  # Freeze all but last few layers
            param.requires_grad = False
    
    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.embed(features))
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

def collate_fn(batch):
    """Custom collate function for variable length captions"""
    images = [item[0] for item in batch]
    captions = [item[1] for item in batch]
    
    # Pad captions to same length
    max_length = max(len(cap) for cap in captions)
    padded_captions = []
    
    for cap in captions:
        padding = torch.zeros(max_length - len(cap), dtype=torch.long)
        padded_cap = torch.cat([cap, padding])
        padded_captions.append(padded_cap)
    
    return torch.stack(images), torch.stack(padded_captions)

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load vocabulary
    with open(args.vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    
    vocab_size = len(vocab)
    logger.info(f"Vocabulary size: {vocab_size}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = COCODataset(
        image_dir=args.image_dir,
        captions_file=args.train_captions,
        vocab=vocab,
        transform=transform
    )
    
    val_dataset = COCODataset(
        image_dir=args.image_dir,
        captions_file=args.val_captions,
        vocab=vocab,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = ImageCaptioningModel(
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        vocab_size=vocab_size
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
        
        for images, captions in train_progress:
            images = images.to(device)
            captions = captions.to(device)
            
            outputs = model(images, captions)
            
            # Calculate loss
            targets = captions[:, 1:]  # Remove SOS
            outputs = outputs[:, :-1, :]  # Remove last output
            
            # Handle padding
            if outputs.shape[1] != targets.shape[1]:
                min_len = min(outputs.shape[1], targets.shape[1])
                outputs = outputs[:, :min_len, :]
                targets = targets[:, :min_len]
            
            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            train_loss += loss.item()
            train_progress.set_postfix({"loss": loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]")
        
        with torch.no_grad():
            for images, captions in val_progress:
                images = images.to(device)
                captions = captions.to(device)
                
                outputs = model(images, captions)
                
                targets = captions[:, 1:]
                outputs = outputs[:, :-1, :]
                
                if outputs.shape[1] != targets.shape[1]:
                    min_len = min(outputs.shape[1], targets.shape[1])
                    outputs = outputs[:, :min_len, :]
                    targets = targets[:, :min_len]
                
                loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
                val_loss += loss.item()
                val_progress.set_postfix({"loss": loss.item()})
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'embed_size': args.embed_size,
                'hidden_size': args.hidden_size,
                'vocab_size': vocab_size
            }, args.model_output)
            logger.info(f"New best model saved with val loss: {avg_val_loss:.4f}")
    
    # Save final model and vocabulary
    final_model_path = MODEL_DIR / "coco_caption_model_final.pth"
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': avg_val_loss,
        'embed_size': args.embed_size,
        'hidden_size': args.hidden_size,
        'vocab_size': vocab_size
    }, final_model_path)
    
    # Copy vocabulary to models directory
    vocab_output = MODEL_DIR / "coco_vocab.pkl"
    with open(vocab_output, 'wb') as f:
        pickle.dump(vocab, f)
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(MODEL_DIR / "training_curve.png")
    plt.close()
    
    logger.info("🎉 Training completed!")
    logger.info(f"📁 Best model saved to: {args.model_output}")
    logger.info(f"📁 Final model saved to: {final_model_path}")
    logger.info(f"📁 Vocabulary saved to: {vocab_output}")
    logger.info(f"📊 Training curve saved to: {MODEL_DIR / 'training_curve.png'}")
    
    return model, vocab

def main():
    parser = argparse.ArgumentParser(description='Train image captioning model on COCO dataset')
    parser.add_argument('--data-dir', type=str, default='./data/processed',
                       help='Directory containing processed COCO data')
    parser.add_argument('--image-dir', type=str, default='./data/train2017',
                       help='Directory containing COCO training images')
    parser.add_argument('--model-output', type=str, default='./models/coco_caption_model.pth',
                       help='Path to save trained model')
    parser.add_argument('--vocab-file', type=str, default='./data/processed/vocab.pkl',
                       help='Path to vocabulary file')
    parser.add_argument('--train-captions', type=str, default='./data/processed/train_captions.json',
                       help='Training captions file')
    parser.add_argument('--val-captions', type=str, default='./data/processed/val_captions.json',
                       help='Validation captions file')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--embed-size', type=int, default=256,
                       help='Embedding size')
    parser.add_argument('--hidden-size', type=int, default=512,
                       help='Hidden size of LSTM')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Create directories
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Download NLTK data if needed
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        logger.info("Downloading NLTK data...")
        nltk.download('punkt_tab', quiet=True)
    
    # Check if data exists
    if not Path(args.vocab_file).exists():
        logger.error(f"❌ Vocabulary file not found: {args.vocab_file}")
        logger.error("Please run scripts/preprocess_coco.py first")
        sys.exit(1)
    
    if not Path(args.train_captions).exists():
        logger.error(f"❌ Training captions not found: {args.train_captions}")
        sys.exit(1)
    
    if not Path(args.image_dir).exists():
        logger.error(f"❌ Image directory not found: {args.image_dir}")
        sys.exit(1)
    
    logger.info("🚀 Starting COCO image captioning training...")
    logger.info(f"📊 Training parameters:")
    logger.info(f"   Batch size: {args.batch_size}")
    logger.info(f"   Epochs: {args.num_epochs}")
    logger.info(f"   Learning rate: {args.learning_rate}")
    logger.info(f"   Embed size: {args.embed_size}")
    logger.info(f"   Hidden size: {args.hidden_size}")
    
    # Start training
    model, vocab = train_model(args)
    
    logger.info("✅ Training completed successfully!")
    logger.info("🎯 Model is ready for inference!")

if __name__ == "__main__":
    main()
