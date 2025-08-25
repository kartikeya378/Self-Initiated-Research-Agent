@echo off
echo 🚀 Launching AI Research Agent Web UI...
echo.

echo 📦 Installing web dependencies...
pip install -r web_requirements.txt

echo.
echo 🌐 Starting web interface...
echo 📱 Open your browser and go to: http://localhost:5000
echo 💡 Press Ctrl+C to stop the server
echo.

python web_ui.py

pause
