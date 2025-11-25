#!/usr/bin/env python3
"""
COCO Dataset Downloader
Downloads COCO 2017 dataset for image captioning training
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COCO 2017 URLs
COCO_URLS = {
    'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
    'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
    'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
}

def download_file(url: str, destination: Path, description: str):
    """Download file with progress bar"""
    logger.info(f"Downloading {description}...")
    
    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100, (block_num * block_size * 100) // total_size)
            print(f"\rProgress: {percent}% ({block_num * block_size}/{total_size} bytes)", end='')
    
    try:
        urllib.request.urlretrieve(url, destination, progress_hook)
        print(f"\n✅ {description} downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download {description}: {e}")
        return False

def extract_archive(archive_path: Path, extract_to: Path):
    """Extract zip or tar.gz archive"""
    logger.info(f"Extracting {archive_path.name}...")
    
    try:
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.name.endswith('.tar.gz'):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            logger.error(f"Unsupported archive format: {archive_path.suffix}")
            return False
        
        logger.info(f"✅ Extracted {archive_path.name}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to extract {archive_path.name}: {e}")
        return False

def create_caption_files(data_dir: Path):
    """Create caption files from COCO annotations"""
    logger.info("Creating caption files...")
    
    annotations_file = data_dir / 'annotations' / 'captions_train2017.json'
    output_dir = data_dir / 'processed'
    output_dir.mkdir(exist_ok=True)
    
    try:
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        
        # Create image to captions mapping
        image_captions = {}
        for annotation in data['annotations']:
            image_id = annotation['image_id']
            caption = annotation['caption'].strip().lower()
            
            if image_id not in image_captions:
                image_captions[image_id] = []
            image_captions[image_id].append(caption)
        
        # Create image filename to captions mapping
        filename_captions = {}
        for image in data['images']:
            image_id = image['id']
            filename = image['file_name']
            
            if image_id in image_captions:
                filename_captions[filename] = image_captions[image_id]
        
        # Save caption files
        with open(output_dir / 'train_captions.json', 'w') as f:
            json.dump(filename_captions, f, indent=2)
        
        logger.info(f"✅ Created caption files with {len(filename_captions)} images")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create caption files: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download COCO dataset for image captioning')
    parser.add_argument('--data-dir', type=str, default='./data', 
                       help='Directory to store COCO dataset')
    parser.add_argument('--skip-images', action='store_true',
                       help='Skip downloading images (annotations only)')
    parser.add_argument('--mini', action='store_true',
                       help='Download mini dataset for testing')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading COCO dataset to: {data_dir}")
    
    if args.mini:
        logger.info("🔬 Downloading mini COCO dataset...")
        # For mini dataset, we could use a subset or different source
        logger.warning("Mini dataset not implemented yet. Using full dataset.")
    
    # Download and extract annotations (always needed)
    annotations_zip = data_dir / 'annotations.zip'
    if not annotations_zip.exists():
        if download_file(COCO_URLS['annotations'], annotations_zip, "COCO annotations"):
            extract_archive(annotations_zip, data_dir)
    
    # Download images if not skipped
    if not args.skip_images:
        for split_name, url in COCO_URLS.items():
            if 'images' not in split_name:
                continue
                
            zip_name = f"{split_name}.zip"
            zip_path = data_dir / zip_name
            
            if not zip_path.exists():
                if download_file(url, zip_path, f"COCO {split_name}"):
                    extract_archive(zip_path, data_dir)
    
    # Create processed caption files
    if create_caption_files(data_dir):
        logger.info("🎉 COCO dataset setup completed successfully!")
        logger.info(f"📁 Dataset location: {data_dir}")
        logger.info("📝 Ready for training!")
    else:
        logger.error("❌ Dataset setup failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
