# Backend Server Starter
Write-Host "`nStarting Backend API Server..." -ForegroundColor Cyan

# Navigate to backend directory
Set-Location $PSScriptRoot

# Activate virtual environment (from parent directory)
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "..\venv\Scripts\Activate.ps1"

# Start Flask server
Write-Host "Starting Flask server on http://localhost:5000..." -ForegroundColor Green
python app.py
