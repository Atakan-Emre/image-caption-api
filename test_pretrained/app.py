"""
Pretrained BLIP Model Test API
Image Captioning with BLIP model from Hugging Face
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import base64
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Pretrained Image Captioning API",
    description="BLIP model based image captioning service",
    version="1.0.0"
)

# Global variables for model
processor = None
model = None
device = None

def load_model():
    """Load BLIP model and processor"""
    global processor, model, device
    
    try:
        logger.info("Loading BLIP model...")
        
        # Check for GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load pretrained BLIP model
        model_name = "Salesforce/blip-image-captioning-base"
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        
        # Move model to device
        model = model.to(device)
        model.eval()
        
        logger.info("✅ BLIP model loaded successfully!")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    load_model()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Pretrained BLIP Image Captioning API",
        "model": "Salesforce/blip-image-captioning-base",
        "status": "ready"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "message": "BLIP model is loaded and ready",
        "device": str(device),
        "model_loaded": model is not None
    }

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model_name": "Salesforce/blip-image-captioning-base",
        "model_type": "BLIP (Bootstrapping Language-Image Pre-training)",
        "total_parameters": f"{total_params:,}",
        "trainable_parameters": f"{trainable_params:,}",
        "device": str(device),
        "framework": "PyTorch + Transformers",
        "image_size": "384x384",
        "vocabulary_size": "30522 (BERT tokenizer)"
    }

def preprocess_image(image_bytes: bytes) -> Image.Image:
    """Preprocess uploaded image"""
    try:
        # Open and convert image
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

def generate_caption(image: Image.Image, use_beam_search: bool = True) -> tuple[str, float]:
    """Generate caption for image"""
    try:
        start_time = time.time()
        
        # Prepare inputs
        inputs = processor(image, return_tensors="pt").to(device)
        
        # Generate caption
        with torch.no_grad():
            if use_beam_search:
                outputs = model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=5,
                    early_stopping=True,
                    temperature=1.0,
                    do_sample=False
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_length=50,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
        
        # Decode caption
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return caption, processing_time
        
    except Exception as e:
        logger.error(f"Caption generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Caption generation failed: {e}")

@app.post("/predict")
async def predict_caption(
    file: UploadFile = File(...),
    use_beam_search: bool = True
):
    """Generate caption for uploaded image"""
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Check file size (max 10MB)
    file_size = 0
    contents = await file.read()
    file_size = len(contents)
    
    if file_size > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")
    
    try:
        # Preprocess image
        image = preprocess_image(contents)
        
        # Generate caption
        caption, processing_time = generate_caption(image, use_beam_search)
        
        # Calculate confidence (simulated based on caption length and beam search)
        confidence = min(0.95, 0.5 + (len(caption.split()) * 0.05))
        if use_beam_search:
            confidence += 0.1
        
        return {
            "predicted_caption": caption,
            "confidence": round(confidence, 3),
            "processing_time": round(processing_time, 3),
            "beam_search_used": use_beam_search,
            "model": "BLIP-base",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/predict/batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    use_beam_search: bool = True
):
    """Generate captions for multiple images"""
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
    
    results = []
    failed = []
    
    for i, file in enumerate(files):
        try:
            # Validate file type
            if not file.content_type.startswith('image/'):
                failed.append({"file_index": i, "filename": file.filename, "error": "Not an image"})
                continue
            
            # Read file
            contents = await file.read()
            
            # Check file size
            if len(contents) > 10 * 1024 * 1024:
                failed.append({"file_index": i, "filename": file.filename, "error": "File too large"})
                continue
            
            # Preprocess and predict
            image = preprocess_image(contents)
            caption, processing_time = generate_caption(image, use_beam_search)
            
            # Calculate confidence
            confidence = min(0.95, 0.5 + (len(caption.split()) * 0.05))
            if use_beam_search:
                confidence += 0.1
            
            results.append({
                "file_index": i,
                "filename": file.filename,
                "predicted_caption": caption,
                "confidence": round(confidence, 3),
                "processing_time": round(processing_time, 3)
            })
            
        except Exception as e:
            failed.append({"file_index": i, "filename": file.filename, "error": str(e)})
    
    return {
        "total_files": len(files),
        "successful": len(results),
        "failed": len(failed),
        "results": results,
        "failed_files": failed,
        "model": "BLIP-base",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

@app.post("/predict/url")
async def predict_from_url(
    url: str,
    use_beam_search: bool = True
):
    """Generate caption for image from URL"""
    
    try:
        # Download image
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Validate content type
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="URL does not point to an image")
        
        # Preprocess and predict
        image = preprocess_image(response.content)
        caption, processing_time = generate_caption(image, use_beam_search)
        
        # Calculate confidence
        confidence = min(0.95, 0.5 + (len(caption.split()) * 0.05))
        if use_beam_search:
            confidence += 0.1
        
        return {
            "url": url,
            "predicted_caption": caption,
            "confidence": round(confidence, 3),
            "processing_time": round(processing_time, 3),
            "beam_search_used": use_beam_search,
            "model": "BLIP-base",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
