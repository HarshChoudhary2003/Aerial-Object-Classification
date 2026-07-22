# Aerial-Object-Classification Setup Script
Write-Host "🚀 Starting Aerial Surveillance AI Setup..." -ForegroundColor Cyan

# 1. Create Virtual Environment for Backend
if (!(Test-Path -Path "venv")) {
    Write-Host "📦 Creating virtual environment for backend..." -ForegroundColor Yellow
    python -m venv venv
}

# 2. Activate Virtual Environment & Install Backend Dependencies
Write-Host "🔌 Activating virtual environment & installing backend dependencies..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend/requirements.txt

# 3. Install Frontend Dependencies
Write-Host "📥 Installing frontend dependencies..." -ForegroundColor Yellow
Set-Location frontend
npm install
Set-Location ..

Write-Host "✅ Setup Complete! Run '.\run.ps1' to start the full application." -ForegroundColor Green
