# Aerial-Object-Classification Run Script
Write-Host "Launching Aerial Surveillance AI System..." -ForegroundColor Cyan

# Start Backend in background
Write-Host "Starting FastAPI Backend..." -ForegroundColor Yellow
$BackendJob = Start-Job {
    Set-Location $using:PWD
    if (Test-Path -Path "venv") {
        .\venv\Scripts\Activate.ps1
    }
    cd backend
    ..\venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
}

# Start Frontend
Write-Host "Starting React Frontend..." -ForegroundColor Yellow
Set-Location frontend
npm run dev

# Cleanup jobs if frontend is closed
Stop-Job $BackendJob
Remove-Job $BackendJob
