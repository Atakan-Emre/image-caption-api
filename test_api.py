#!/usr/bin/env python3
"""
Smoke test script for Image Captioning API.
Tests health endpoint and prediction with a sample image.
"""

import requests
import os
import sys
import time
from pathlib import Path
import json
from PIL import Image, ImageDraw
import io

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
SKIP_MODEL_CHECK = os.environ.get("SKIP_MODEL_CHECK", "false").lower() in {"1", "true", "yes"}

def create_test_image():
    """Create a simple test image programmatically."""
    img = Image.new('RGB', (224, 224), color='blue')
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 174, 174], fill='white')
    draw.ellipse([75, 75, 149, 149], fill='red')
    
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()

def test_health_endpoint():
    """Test the /health endpoint."""
    print("🔍 Testing health endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data.get('status')}")
            print(f"   Message: {data.get('message')}")
            return True
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is the server running?")
        return False
    except requests.exceptions.Timeout:
        print("❌ Health check timed out")
        return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

def test_model_info():
    """Test the /model/info endpoint."""
    print("\n🔍 Testing model info endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/model/info", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Model info retrieved:")
            print(f"   Model type: {data.get('model_type')}")
            print(f"   Encoder: {data.get('encoder')}")
            print(f"   Vocabulary size: {data.get('vocabulary_size')}")
            print(f"   Device: {data.get('device')}")
            return True
        else:
            print(f"❌ Model info failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Model info error: {str(e)}")
        return False

def test_prediction_endpoint():
    """Test the /predict endpoint with a sample image."""
    print("\n🔍 Testing prediction endpoint...")
    
    try:
        test_image = create_test_image()
        
        files = {'file': ('test_image.jpg', test_image, 'image/jpeg')}
        data = {'use_beam_search': 'true'}
        
        print("   Sending test image...")
        start_time = time.time()
        
        response = requests.post(
            f"{API_BASE_URL}/predict",
            files=files,
            data=data,
            timeout=30
        )
        
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            caption = result.get('predicted_caption')
            confidence = result.get('confidence')
            
            print(f"✅ Prediction successful in {end_time - start_time:.2f}s")
            print(f"   Caption: '{caption}'")
            print(f"   Confidence: {confidence}")
            
            if caption and len(caption.strip()) > 0:
                print("   ✅ Caption generated successfully")
                return True
            else:
                print("   ⚠️  Empty caption generated")
                return False
                
        else:
            print(f"❌ Prediction failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Prediction timed out")
        return False
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return False

def test_batch_endpoint():
    """Test the /predict/batch endpoint with multiple images."""
    print("\n🔍 Testing batch prediction endpoint...")
    
    try:
        test_image1 = create_test_image()
        test_image2 = create_test_image()
        
        files = [
            ('files', ('test_image1.jpg', test_image1, 'image/jpeg')),
            ('files', ('test_image2.jpg', test_image2, 'image/jpeg'))
        ]
        data = {'use_beam_search': 'false'}
        
        print("   Sending 2 test images...")
        start_time = time.time()
        
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            files=files,
            data=data,
            timeout=30
        )
        
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            results = result.get('results', [])
            errors = result.get('errors', [])
            
            print(f"✅ Batch prediction successful in {end_time - start_time:.2f}s")
            print(f"   Successful: {result.get('successful_predictions')}")
            print(f"   Failed: {result.get('failed_predictions')}")
            
            for i, prediction in enumerate(results):
                print(f"   Image {i+1}: '{prediction.get('predicted_caption')}' (confidence: {prediction.get('confidence')})")
            
            if errors:
                print("   Errors:")
                for error in errors:
                    print(f"     - {error.get('error')}")
            
            return len(results) > 0
            
        else:
            print(f"❌ Batch prediction failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Batch prediction error: {str(e)}")
        return False

def check_model_files():
    """Check if model files exist."""
    print("🔍 Checking model files...")
    
    model_path = Path("models/caption_model.pth")
    vocab_path = Path("models/vocab.pkl")
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model file found: {size_mb:.1f}MB")
    else:
        print("❌ Model file not found: models/caption_model.pth")
        return False
    
    if vocab_path.exists():
        size_kb = vocab_path.stat().st_size / 1024
        print(f"✅ Vocabulary file found: {size_kb:.1f}KB")
    else:
        print("❌ Vocabulary file not found: models/vocab.pkl")
        return False
    
    return True

def main():
    """Run all smoke tests."""
    print("🚀 Image Captioning API Smoke Test")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("app/main.py").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Check model files (can be skipped for remote endpoints)
    if not SKIP_MODEL_CHECK:
        if not check_model_files():
            print("\n💡 Tip: Run 'python training/train.py' to create model files or set SKIP_MODEL_CHECK=true to skip this check.")
            sys.exit(1)
    
    # Run API tests
    tests = [
        ("Health Check", test_health_endpoint),
        ("Model Info", test_model_info),
        ("Single Prediction", test_prediction_endpoint),
        ("Batch Prediction", test_batch_endpoint)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            passed += 1
        time.sleep(1)  # Brief pause between tests
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
