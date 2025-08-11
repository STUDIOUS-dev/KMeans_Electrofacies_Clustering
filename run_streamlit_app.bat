@echo off
echo ========================================
echo   K-Means Electrofacies Clustering
echo   Streamlit Web Application Launcher
echo ========================================
echo.

echo Checking Python environment...
python --version
echo.

echo Starting Streamlit application...
echo.
echo The web application will open in your default browser.
echo If it doesn't open automatically, go to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application.
echo.

python -m streamlit run streamlit_kmeans_app.py --server.port 8501

pause
