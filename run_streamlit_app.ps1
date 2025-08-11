Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  K-Means Electrofacies Clustering" -ForegroundColor Cyan
Write-Host "  Streamlit Web Application Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Checking Python environment..." -ForegroundColor Yellow
python --version
Write-Host ""

Write-Host "Installing/Updating required packages..." -ForegroundColor Yellow
pip install -r requirements.txt
Write-Host ""

Write-Host "Starting Streamlit application..." -ForegroundColor Green
Write-Host ""
Write-Host "The web application will open in your default browser." -ForegroundColor Green
Write-Host "If it doesn't open automatically, go to: http://localhost:8501" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the application." -ForegroundColor Yellow
Write-Host ""

streamlit run streamlit_kmeans_app.py

Read-Host "Press Enter to exit"
