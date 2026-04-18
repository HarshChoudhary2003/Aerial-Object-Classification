# Aerial-Object-Classification Run Script
Write-Host "🛰 Launching Aerial Surveillance AI..." -ForegroundColor Cyan

# Activate Virtual Environment
if (Test-Path -Path "venv") {
    .\venv\Scripts\Activate.ps1
} else {
    Write-Host "⚠️ Virtual environment not found. Running with global python..." -ForegroundColor Yellow
}

# Run Streamlit
streamlit run streamlit_app/app.py
