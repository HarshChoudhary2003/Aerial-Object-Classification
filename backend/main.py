from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import logging

from ml_service import ml_service

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Aerial Object Detection API",
    description="Advanced AI system for Bird vs Drone classification and detection",
    version="2.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing models...")
    # Pre-load models on startup to reduce latency on first request
    # Try to load them if they exist
    try:
        ml_service.load_models()
    except Exception as e:
        logger.error(f"Failed to pre-load models: {e}")

@app.get("/")
def read_root():
    return {"status": "online", "message": "Aerial Surveillance API is running"}

@app.post("/api/analyze/image")
async def analyze_image(
    file: UploadFile = File(...),
    task: str = Form("both"), # "classification", "detection", "both"
    cls_model_type: str = Form("resnet50"),
    conf_threshold: float = Form(0.5)
):
    try:
        contents = await file.read()
        
        # Validate file size (e.g., max 50MB)
        if len(contents) > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Max 50MB.")
        
        logger.info(f"Processing image {file.filename} for task: {task}")
        
        # Load the selected classification model if it changed
        # We handle this loosely here, in production we might keep multiple loaded
        # if RAM permits, but for now we just make sure ml_service has models loaded
        
        results = ml_service.analyze_image(
            image_bytes=contents, 
            task=task, 
            cls_model_type=cls_model_type,
            conf_threshold=conf_threshold
        )
        
        return {
            "success": True,
            "filename": file.filename,
            "task": task,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
