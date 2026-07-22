from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import logging

from app.services.ml_service import ml_service

logger = logging.getLogger(__name__)

router = APIRouter()

from app.core.config import settings

@router.post("/image")
async def analyze_image(
    file: UploadFile = File(...),
    task: str = Form("both"), # "classification", "detection", "both"
    cls_model_type: str = Form("resnet50"),
    conf_threshold: float = Form(0.5)
):
    try:
        contents = await file.read()
        
        # Validate file size
        if len(contents) > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=400, detail="File too large. Max 50MB.")
        
        logger.info(f"Processing image {file.filename} for task: {task}")
        
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
