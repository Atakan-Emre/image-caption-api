#!/usr/bin/env python3
"""
COCO Dataset Preprocessor
Preprocesses COCO dataset for image captioning training
"""

import os
import sys
import json
import argparse
from pathlib import Path
import logging
from collections import Counter
import nltk
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from training.train import Vocabulary, CaptionDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        logger.info("Downloading NLTK punkt_tab data...")
        nltk.download('punkt_tab', quiet=True)

def analyze_captions(captions_file: Path):
    """Analyze caption statistics"""
    logger.info("Analyzing captions...")
    
    with open(captions_file, 'r') as f:
        captions_data = json.load(f)
    
    all_captions = []
    caption_lengths = []
    
    for filename, captions in captions_data.items():
        for caption in captions:
            all_captions.append(caption)
            caption_lengths.append(len(caption.split()))
    
    # Calculate statistics
    avg_length = sum(caption_lengths) / len(caption_lengths)
    max_length = max(caption_lengths)
    min_length = min(caption_lengths)
    
    # Word frequency
    all_words = ' '.join(all_captions).split()
    word_freq = Counter(all_words)
    
    logger.info(f"📊 Caption Statistics:")
    logger.info(f"   Total captions: {len(all_captions)}")
    logger.info(f"   Total images: {len(captions_data)}")
    logger.info(f"   Average caption length: {avg_length:.2f} words")
    logger.info(f"   Max caption length: {max_length} words")
    logger.info(f"   Min caption length: {min_length} words")
    logger.info(f"   Unique words: {len(word_freq)}")
    logger.info(f"   Most common words: {word_freq.most_common(10)}")
    
    return {
        'total_captions': len(all_captions),
        'total_images': len(captions_data),
        'avg_length': avg_length,
        'max_length': max_length,
        'min_length': min_length,
        'unique_words': len(word_freq),
        'vocab_size': min(10000, len(word_freq))  # Target vocab size
    }

def build_vocabulary(captions_file: Path, vocab_size: int = 10000, min_freq: int = 2):
    """Build vocabulary from captions"""
    logger.info(f"Building vocabulary (size: {vocab_size}, min_freq: {min_freq})...")
    
    vocab = Vocabulary()
    word_freq = Counter()
    
    # Count word frequencies
    with open(captions_file, 'r') as f:
        captions_data = json.load(f)
    
    for filename, captions in captions_data.items():
        for caption in captions:
            tokens = nltk.word_tokenize(caption.lower())
            word_freq.update(tokens)
    
    # Add tokens to vocabulary
    vocab.add_token('<PAD>')
    vocab.add_token('<SOS>')
    vocab.add_token('<EOS>')
    vocab.add_token('<UNK>')
    
    # Add most frequent words
    for word, freq in word_freq.most_common():
        if freq >= min_freq and len(vocab) < vocab_size:
            vocab.add_token(word)
    
    logger.info(f"✅ Vocabulary built with {len(vocab)} tokens")
    return vocab

def validate_images(captions_file: Path, image_dir: Path):
    """Validate that all images in captions exist"""
    logger.info("Validating images...")
    
    with open(captions_file, 'r') as f:
        captions_data = json.load(f)
    
    missing_images = []
    corrupted_images = []
    valid_images = []
    
    for filename in tqdm(captions_data.keys(), desc="Checking images"):
        image_path = image_dir / filename
        
        if not image_path.exists():
            missing_images.append(filename)
            continue
        
        try:
            with Image.open(image_path) as img:
                img.verify()
            valid_images.append(filename)
        except Exception:
            corrupted_images.append(filename)
    
    logger.info(f"📊 Image Validation Results:")
    logger.info(f"   Valid images: {len(valid_images)}")
    logger.info(f"   Missing images: {len(missing_images)}")
    logger.info(f"   Corrupted images: {len(corrupted_images)}")
    
    if missing_images or corrupted_images:
        logger.warning("⚠️  Some images have issues. Consider cleaning the dataset.")
    
    return valid_images

def create_clean_dataset(captions_file: Path, image_dir: Path, output_dir: Path, valid_images: list):
    """Create clean dataset with only valid images"""
    logger.info("Creating clean dataset...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(captions_file, 'r') as f:
        captions_data = json.load(f)
    
    # Filter captions for valid images only
    clean_captions = {}
    for filename in valid_images:
        if filename in captions_data:
            clean_captions[filename] = captions_data[filename]
    
    # Save clean captions
    clean_captions_file = output_dir / 'clean_captions.json'
    with open(clean_captions_file, 'w') as f:
        json.dump(clean_captions, f, indent=2)
    
    logger.info(f"✅ Clean dataset created with {len(clean_captions)} images")
    logger.info(f"📁 Saved to: {clean_captions_file}")
    
    return clean_captions_file

def create_data_splits(captions_file: Path, output_dir: Path, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Create train/val/test splits"""
    logger.info("Creating data splits...")
    
    with open(captions_file, 'r') as f:
        captions_data = json.load(f)
    
    # Shuffle and split
    filenames = list(captions_data.keys())
    import random
    random.shuffle(filenames)
    
    total = len(filenames)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    
    train_files = filenames[:train_end]
    val_files = filenames[train_end:val_end]
    test_files = filenames[val_end:]
    
    # Create splits
    splits = {
        'train': {f: captions_data[f] for f in train_files},
        'val': {f: captions_data[f] for f in val_files},
        'test': {f: captions_data[f] for f in test_files}
    }
    
    # Save splits
    for split_name, split_data in splits.items():
        split_file = output_dir / f'{split_name}_captions.json'
        with open(split_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        logger.info(f"✅ {split_name.capitalize()} split: {len(split_data)} images")
    
    return splits

def main():
    parser = argparse.ArgumentParser(description='Preprocess COCO dataset for image captioning')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory containing COCO dataset')
    parser.add_argument('--output-dir', type=str, default='./data/processed',
                       help='Directory to save processed data')
    parser.add_argument('--vocab-size', type=int, default=10000,
                       help='Vocabulary size')
    parser.add_argument('--min-freq', type=int, default=2,
                       help='Minimum word frequency')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation set ratio')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip image validation (faster)')
    
    args = parser.parse_args()
    
    # Download NLTK data
    download_nltk_data()
    
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🔧 Preprocessing COCO dataset from: {data_dir}")
    logger.info(f"📁 Output directory: {output_dir}")
    
    # Input files
    captions_file = data_dir / 'processed' / 'train_captions.json'
    image_dir = data_dir / 'train2017'
    
    if not captions_file.exists():
        logger.error(f"❌ Captions file not found: {captions_file}")
        logger.error("Please run download_coco.py first")
        sys.exit(1)
    
    if not image_dir.exists():
        logger.error(f"❌ Image directory not found: {image_dir}")
        sys.exit(1)
    
    # Analyze captions
    stats = analyze_captions(captions_file)
    
    # Validate images (unless skipped)
    if not args.skip_validation:
        valid_images = validate_images(captions_file, image_dir)
    else:
        with open(captions_file, 'r') as f:
            captions_data = json.load(f)
        valid_images = list(captions_data.keys())
        logger.info(f"⏭️  Skipping validation. Using all {len(valid_images)} images.")
    
    # Create clean dataset
    clean_captions_file = create_clean_dataset(captions_file, image_dir, output_dir, valid_images)
    
    # Create data splits
    splits = create_data_splits(clean_captions_file, output_dir, args.train_ratio, args.val_ratio)
    
    # Build vocabulary
    vocab = build_vocabulary(clean_captions_file, args.vocab_size, args.min_freq)
    
    # Save vocabulary
    vocab_file = output_dir / 'vocab.pkl'
    import pickle
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab, f)
    logger.info(f"✅ Vocabulary saved to: {vocab_file}")
    
    # Save statistics
    stats_file = output_dir / 'dataset_stats.json'
    stats['splits'] = {k: len(v) for k, v in splits.items()}
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("🎉 COCO dataset preprocessing completed!")
    logger.info(f"📁 Processed data saved to: {output_dir}")
    logger.info("🚀 Ready for training with: python training/train_coco.py")

if __name__ == "__main__":
    main()
