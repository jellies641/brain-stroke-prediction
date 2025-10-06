#!/usr/bin/env python3
"""
Railway Startup Script for Brain Stroke Prediction App
This script ensures the app starts correctly without Node.js dependencies
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if the environment is properly set up"""
    logger.info("🔍 Checking environment setup...")

    # Check if we're in the right directory
    if not os.path.exists('requirements.txt'):
        logger.error("❌ requirements.txt not found - wrong directory?")
        sys.exit(1)

    # Check if backend directory exists
    if not os.path.exists('backend/app.py'):
        logger.error("❌ backend/app.py not found")
        sys.exit(1)

    logger.info("✅ API-only backend mode")

    logger.info("✅ Environment check passed")

def set_environment_variables():
    """Set required environment variables"""
    logger.info("🔧 Setting environment variables...")

    # Set Python path
    python_path = os.environ.get('PYTHONPATH', '')
    backend_path = os.path.abspath('backend')
    ml_model_path = os.path.abspath('ml-model')

    if backend_path not in python_path:
        python_path = f"{backend_path}:{ml_model_path}:{python_path}".strip(':')
        os.environ['PYTHONPATH'] = python_path
        logger.info(f"✅ PYTHONPATH set to: {python_path}")

    # Set Flask environment
    if not os.environ.get('FLASK_ENV'):
        os.environ['FLASK_ENV'] = 'production'
        logger.info("✅ FLASK_ENV set to production")

    # Set port
    port = os.environ.get('PORT', '5000')
    os.environ['PORT'] = port
    logger.info(f"✅ PORT set to: {port}")

def start_flask_app():
    """Start the Flask application"""
    logger.info("🚀 Starting Flask application...")

    try:
        # Change to backend directory
        os.chdir('backend')
        logger.info("📁 Changed to backend directory")

        # Start the Flask app
        logger.info("🧠 Starting Brain Stroke Risk Prediction API...")
        logger.info("=" * 60)
        logger.info(f"🌐 API will be available on port {os.environ.get('PORT', '5000')}")
        logger.info("📊 API endpoints:")
        logger.info("     GET  /                 - Health check")
        logger.info("     GET  /api/info         - API information")
        logger.info("     POST /api/auth/signup  - User registration")
        logger.info("     POST /api/auth/login   - User login")
        logger.info("     POST /api/predict      - Stroke prediction")
        logger.info("     GET  /api/history      - Prediction history")
        logger.info("=" * 60)

        # Execute the Flask app
        subprocess.run([sys.executable, 'app.py'], check=True)

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Flask app failed to start: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("🛑 App stopped by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

def main():
    """Main startup function"""
    logger.info("🧠 Brain Stroke Prediction API - Railway Backend")
    logger.info("=" * 60)

    try:
        check_environment()
        set_environment_variables()
        start_flask_app()
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
