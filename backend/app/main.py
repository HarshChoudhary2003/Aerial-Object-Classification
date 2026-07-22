from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.api.routers import api_router
from app.services.ml_service import ml_service

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
    try:
        ml_service.load_models()
    except Exception as e:
        logger.error(f"Failed to pre-load models: {e}")

@app.get("/")
def read_root():
    return {"status": "online", "message": "Aerial Surveillance API is running"}

app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
