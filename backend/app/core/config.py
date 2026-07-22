import os
from pydantic import BaseModel

class Settings(BaseModel):
    PROJECT_NAME: str = "Aerial Object Detection"
    VERSION: str = "2.0.0"
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB

settings = Settings()
