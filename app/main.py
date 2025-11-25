from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
from datetime import datetime
from typing import Optional
import io
from PIL import Image

from app.schemas import PredictionResponse, HealthResponse, ErrorResponse
from app.model_loader import predict_image, get_model_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Image Captioning API",
    description="API for generating image captions using deep learning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    try:
        model_loader = get_model_loader()
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {str(e)}")

@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="healthy",
        message="Image Captioning API is running",
        timestamp=datetime.now().isoformat()
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        model_loader = get_model_loader()
        return HealthResponse(
            status="healthy",
            message="Model is loaded and ready",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.post("/predict", response_model=PredictionResponse)
async def predict_caption(
    file: UploadFile = File(...),
    use_beam_search: Optional[bool] = True
):
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        start_time = time.time()
        
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )

        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="File size must be less than 10MB"
            )
        
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file"
            )
        
        caption, confidence = predict_image(image_bytes, use_beam_search)
        
        processing_time = time.time() - start_time
        logger.info(f"Prediction completed in {processing_time:.2f}s with confidence {confidence}")
        
        return PredictionResponse(
            predicted_caption=caption,
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction"
        )

@app.post("/predict/batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    use_beam_search: Optional[bool] = True
):
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 files allowed per batch"
        )
    
    results = []
    errors = []
    
    for i, file in enumerate(files):
        try:
            if not file.content_type.startswith('image/'):
                errors.append({"file_index": i, "filename": file.filename, "error": "Not an image file"})
                continue
            
            image_bytes = await file.read()

            if len(image_bytes) == 0:
                errors.append({"file_index": i, "filename": file.filename, "error": "Empty file"})
                continue

            if len(image_bytes) > 10 * 1024 * 1024:
                errors.append({"file_index": i, "filename": file.filename, "error": "File too large"})
                continue
            
            try:
                img = Image.open(io.BytesIO(image_bytes))
                img.verify()
            except Exception:
                errors.append({"file_index": i, "filename": file.filename, "error": "Invalid image file"})
                continue
            
            caption, confidence = predict_image(image_bytes, use_beam_search)
            
            results.append({
                "file_index": i,
                "filename": file.filename,
                "predicted_caption": caption,
                "confidence": confidence
            })
            
        except Exception as e:
            errors.append({"file_index": i, "filename": file.filename, "error": str(e)})
    
    return {
        "results": results,
        "errors": errors,
        "total_files": len(files),
        "successful_predictions": len(results),
        "failed_predictions": len(errors),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info")
async def model_info():
    try:
        model_loader = get_model_loader()
        vocab_size = len(model_loader.vocab)
        
        return {
            "model_type": "CNN + LSTM",
            "encoder": "ResNet50",
            "decoder": "LSTM",
            "vocabulary_size": vocab_size,
            "device": str(model_loader.device),
            "beam_search_available": True,
            "max_sequence_length": 50,
            "supported_formats": ["JPEG", "PNG", "BMP", "GIF"],
            "max_file_size": "10MB"
        }
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not retrieve model information")

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP {exc.status_code}",
            message=exc.detail,
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message="An unexpected error occurred",
            timestamp=datetime.now().isoformat()
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
