# Aerial-Object-Classification Setup Script
Write-Host "🚀 Starting Aerial Surveillance AI Setup..." -ForegroundColor Cyan

# 1. Create Virtual Environment
if (!(Test-Path -Path "venv")) {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# 2. Activate Virtual Environment
Write-Host "🔌 Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# 3. Upgrade Pip
Write-Host "🆙 Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# 4. Install Dependencies
Write-Host "📥 Installing dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt

# 5. Download base YOLO model if not exists
if (!(Test-Path -Path "yolov8n.pt")) {
    Write-Host "📥 Downloading base YOLOv8n model for fallback..." -ForegroundColor Yellow
    # This will be downloaded automatically by ultralytics on first run, 
    # but we can trigger it here if we wanted to.
}

Write-Host "✅ Setup Complete! Run '.\run.ps1' to start the application." -ForegroundColor Green
