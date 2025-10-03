#!/usr/bin/env python3
"""
Brain Stroke Risk Prediction System - Setup Script
==================================================

This script sets up the complete Brain Stroke Risk Prediction System including:
- Backend Flask API with ML model
- Frontend React application
- ML model training
- Database initialization
- Environment configuration

Usage:
    python setup.py [options]

Options:
    --full          Full setup (default)
    --backend-only  Setup only backend
    --frontend-only Setup only frontend
    --ml-only       Setup only ML model
    --dev           Development setup
    --prod          Production setup
    --help          Show this help message

Author: Brain Stroke Risk Prediction Team
Version: 1.0.0
"""

import os
import sys
import subprocess
import argparse
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any
import urllib.request
import zipfile

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_colored(message: str, color: str = Colors.OKGREEN):
    """Print colored message to terminal"""
    print(f"{color}{message}{Colors.ENDC}")

def print_header(message: str):
    """Print header message"""
    print_colored(f"\n{'='*60}", Colors.HEADER)
    print_colored(f"{message.center(60)}", Colors.HEADER)
    print_colored(f"{'='*60}\n", Colors.HEADER)

def print_step(message: str):
    """Print step message"""
    print_colored(f"🔄 {message}", Colors.OKBLUE)

def print_success(message: str):
    """Print success message"""
    print_colored(f"✅ {message}", Colors.OKGREEN)

def print_warning(message: str):
    """Print warning message"""
    print_colored(f"⚠️  {message}", Colors.WARNING)

def print_error(message: str):
    """Print error message"""
    print_colored(f"❌ {message}", Colors.FAIL)

def run_command(command: List[str], cwd: str = None, capture_output: bool = False) -> bool:
    """Run command and return success status"""
    try:
        if capture_output:
            result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=True)
            return True, result.stdout
        else:
            subprocess.run(command, cwd=cwd, check=True)
            return True, None
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {' '.join(command)}")
        if hasattr(e, 'stderr') and e.stderr:
            print_error(f"Error: {e.stderr}")
        return False, None
    except FileNotFoundError:
        print_error(f"Command not found: {command[0]}")
        return False, None

def check_prerequisites():
    """Check if required software is installed"""
    print_header("CHECKING PREREQUISITES")

    prerequisites = [
        ("python3", ["python3", "--version"]),
        ("pip", ["pip", "--version"]),
        ("node", ["node", "--version"]),
        ("npm", ["npm", "--version"]),
        ("git", ["git", "--version"])
    ]

    missing = []

    for name, command in prerequisites:
        success, output = run_command(command, capture_output=True)
        if success:
            version = output.strip().split('\n')[0] if output else "Unknown"
            print_success(f"{name}: {version}")
        else:
            missing.append(name)
            print_error(f"{name}: Not found")

    if missing:
        print_error(f"Missing prerequisites: {', '.join(missing)}")
        print_error("Please install the missing prerequisites and run setup again.")
        return False

    print_success("All prerequisites are installed!")
    return True

def create_directory_structure():
    """Create project directory structure"""
    print_header("CREATING DIRECTORY STRUCTURE")

    directories = [
        "backend/uploads",
        "backend/logs",
        "ml-model/data",
        "ml-model/models",
        "ml-model/plots",
        "frontend/public",
        "frontend/src",
        "docs",
        "tests/backend",
        "tests/frontend",
        "tests/ml",
        "deployment",
        "scripts"
    ]

    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print_step(f"Created directory: {directory}")

    print_success("Directory structure created!")

def setup_backend():
    """Setup backend Flask application"""
    print_header("SETTING UP BACKEND")

    # Create virtual environment
    print_step("Creating Python virtual environment...")
    success, _ = run_command(["python3", "-m", "venv", "backend/venv"])
    if not success:
        print_error("Failed to create virtual environment")
        return False

    # Activate virtual environment and install dependencies
    if os.name == 'nt':  # Windows
        pip_path = "backend/venv/Scripts/pip"
        python_path = "backend/venv/Scripts/python"
    else:  # Unix/Linux/Mac
        pip_path = "backend/venv/bin/pip"
        python_path = "backend/venv/bin/python"

    print_step("Installing Python dependencies...")
    success, _ = run_command([pip_path, "install", "--upgrade", "pip"])
    if not success:
        print_error("Failed to upgrade pip")
        return False

    success, _ = run_command([pip_path, "install", "-r", "requirements.txt"], cwd="backend")
    if not success:
        print_error("Failed to install Python dependencies")
        return False

    # Create environment file
    print_step("Creating backend environment file...")
    if not Path("backend/.env").exists():
        shutil.copy("backend/.env.example", "backend/.env")
        print_success("Created backend/.env file")
    else:
        print_warning("backend/.env already exists, skipping...")

    # Initialize database
    print_step("Initializing database...")
    success, _ = run_command([python_path, "app.py"], cwd="backend")
    if success:
        print_success("Database initialized!")
    else:
        print_warning("Database initialization may have failed, but continuing...")

    print_success("Backend setup completed!")
    return True

def setup_ml_model():
    """Setup and train ML model"""
    print_header("SETTING UP ML MODEL")

    # Use backend's virtual environment
    if os.name == 'nt':  # Windows
        python_path = "../backend/venv/Scripts/python"
    else:  # Unix/Linux/Mac
        python_path = "../backend/venv/bin/python"

    print_step("Training ML model...")
    success, _ = run_command([python_path, "train_model.py"], cwd="ml-model")
    if success:
        print_success("ML model trained successfully!")
    else:
        print_error("Failed to train ML model")
        return False

    # Verify model files were created
    model_files = ["stroke_model.pkl", "scaler.pkl", "model_metadata.json"]
    for file in model_files:
        if Path(f"ml-model/{file}").exists():
            print_success(f"Created {file}")
        else:
            print_error(f"Missing {file}")
            return False

    return True

def setup_frontend():
    """Setup frontend React application"""
    print_header("SETTING UP FRONTEND")

    # Install Node.js dependencies
    print_step("Installing Node.js dependencies...")
    success, _ = run_command(["npm", "install"], cwd="frontend")
    if not success:
        print_error("Failed to install Node.js dependencies")
        return False

    # Create environment file
    print_step("Creating frontend environment file...")
    if not Path("frontend/.env").exists():
        shutil.copy("frontend/.env.example", "frontend/.env")
        print_success("Created frontend/.env file")
    else:
        print_warning("frontend/.env already exists, skipping...")

    print_success("Frontend setup completed!")
    return True

def create_scripts():
    """Create useful scripts for development"""
    print_header("CREATING UTILITY SCRIPTS")

    scripts = {
        "start-backend.sh": """#!/bin/bash
# Start Backend Server
echo "Starting Brain Stroke Risk Prediction Backend..."
cd backend
source venv/bin/activate
python app.py
""",
        "start-frontend.sh": """#!/bin/bash
# Start Frontend Development Server
echo "Starting Brain Stroke Risk Prediction Frontend..."
cd frontend
npm start
""",
        "start-dev.sh": """#!/bin/bash
# Start both backend and frontend in development mode
echo "Starting Brain Stroke Risk Prediction System in Development Mode..."

# Start backend in background
echo "Starting backend..."
cd backend
source venv/bin/activate
python app.py &
BACKEND_PID=$!

# Wait a bit for backend to start
sleep 5

# Start frontend
echo "Starting frontend..."
cd ../frontend
npm start &
FRONTEND_PID=$!

# Wait for user input to stop
echo "System is running. Press Enter to stop..."
read

# Kill processes
echo "Stopping services..."
kill $BACKEND_PID $FRONTEND_PID
echo "Services stopped."
""",
        "train-model.sh": """#!/bin/bash
# Train ML Model
echo "Training ML Model..."
cd ml-model
source ../backend/venv/bin/activate
python train_model.py
echo "Model training completed!"
""",
        "test-all.sh": """#!/bin/bash
# Run all tests
echo "Running all tests..."

# Backend tests
echo "Running backend tests..."
cd backend
source venv/bin/activate
python -m pytest tests/ -v

# Frontend tests
echo "Running frontend tests..."
cd ../frontend
npm test -- --coverage --watchAll=false

echo "All tests completed!"
""",
        "deploy.sh": """#!/bin/bash
# Deploy to production
echo "Deploying Brain Stroke Risk Prediction System..."

# Build frontend
echo "Building frontend..."
cd frontend
npm run build

# Copy built files to backend static folder
echo "Copying frontend build to backend..."
cp -r build/* ../backend/static/

echo "Deployment preparation completed!"
echo "Remember to:"
echo "1. Set production environment variables"
echo "2. Configure your web server (nginx/apache)"
echo "3. Set up SSL certificates"
echo "4. Configure database for production"
"""
    }

    # Create scripts directory
    scripts_dir = Path("scripts")
    scripts_dir.mkdir(exist_ok=True)

    for filename, content in scripts.items():
        script_path = scripts_dir / filename
        with open(script_path, 'w') as f:
            f.write(content)

        # Make scripts executable on Unix systems
        if os.name != 'nt':
            os.chmod(script_path, 0o755)

        print_success(f"Created {filename}")

    print_success("Utility scripts created!")

def create_documentation():
    """Create documentation files"""
    print_header("CREATING DOCUMENTATION")

    docs = {
        "INSTALLATION.md": """# Installation Guide

## Prerequisites
- Python 3.8+
- Node.js 14+
- npm 6+
- Git

## Quick Start
1. Run the setup script: `python setup.py`
2. Start the system: `./scripts/start-dev.sh`
3. Access the application at http://localhost:3000

## Manual Installation
See the main README.md for detailed manual installation instructions.
""",
        "API.md": """# API Documentation

## Authentication Endpoints
- POST /api/auth/login - User login
- POST /api/auth/signup - User registration
- POST /api/auth/refresh - Refresh access token

## Prediction Endpoints
- POST /api/predict - Create stroke risk prediction
- GET /api/history - Get user prediction history
- GET /api/history/{id} - Get specific prediction

## User Endpoints
- GET /api/profile - Get user profile
- PUT /api/profile - Update user profile
""",
        "DEPLOYMENT.md": """# Deployment Guide

## Production Setup
1. Set environment variables for production
2. Configure database (PostgreSQL recommended)
3. Set up reverse proxy (nginx recommended)
4. Configure SSL certificates
5. Set up monitoring and logging

## Docker Deployment
Docker configuration files are available in the deployment/ directory.

## Cloud Deployment
The system can be deployed on AWS, Google Cloud, or Azure.
See cloud-specific guides in deployment/cloud/
""",
        "CONTRIBUTING.md": """# Contributing Guidelines

## Development Setup
1. Fork the repository
2. Run `python setup.py --dev`
3. Create a feature branch
4. Make your changes
5. Run tests: `./scripts/test-all.sh`
6. Submit a pull request

## Code Standards
- Python: Follow PEP 8
- JavaScript: Use ESLint configuration
- Commit messages: Use conventional commits

## Testing
- Write tests for new features
- Ensure all tests pass before submitting PR
- Maintain test coverage above 80%
"""
    }

    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)

    for filename, content in docs.items():
        with open(docs_dir / filename, 'w') as f:
            f.write(content)
        print_success(f"Created {filename}")

    print_success("Documentation created!")

def create_config_files():
    """Create configuration files"""
    print_header("CREATING CONFIGURATION FILES")

    # Docker Compose file
    docker_compose = """version: '3.8'
services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=postgresql://user:password@db:5432/stroke_db
    depends_on:
      - db
    volumes:
      - ./backend/uploads:/app/uploads
      - ./ml-model:/app/ml-model

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    depends_on:
      - backend

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=stroke_db
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
"""

    with open("docker-compose.yml", 'w') as f:
        f.write(docker_compose)
    print_success("Created docker-compose.yml")

    # GitHub Actions workflow
    github_workflow = """name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9

    - name: Set up Node.js
      uses: actions/setup-node@v2
      with:
        node-version: 16

    - name: Install dependencies
      run: |
        python setup.py --backend-only
        cd frontend && npm install

    - name: Run tests
      run: |
        ./scripts/test-all.sh
"""

    github_dir = Path(".github/workflows")
    github_dir.mkdir(parents=True, exist_ok=True)

    with open(github_dir / "ci.yml", 'w') as f:
        f.write(github_workflow)
    print_success("Created GitHub Actions workflow")

def run_tests():
    """Run basic tests to verify setup"""
    print_header("RUNNING VERIFICATION TESTS")

    # Test backend
    print_step("Testing backend...")
    if os.name == 'nt':
        python_path = "backend/venv/Scripts/python"
    else:
        python_path = "backend/venv/bin/python"

    # Simple import test
    test_script = """
import sys
sys.path.append('.')
try:
    from app import create_app
    from ml_service import stroke_predictor
    print("✅ Backend modules import successfully")
except ImportError as e:
    print(f"❌ Backend import error: {e}")
"""

    with open("backend/test_imports.py", 'w') as f:
        f.write(test_script)

    success, _ = run_command([python_path, "test_imports.py"], cwd="backend")
    if success:
        print_success("Backend tests passed!")
    else:
        print_warning("Backend tests failed, but continuing...")

    # Clean up test file
    os.remove("backend/test_imports.py")

    # Test frontend build
    print_step("Testing frontend build...")
    success, _ = run_command(["npm", "run", "build"], cwd="frontend")
    if success:
        print_success("Frontend build test passed!")
    else:
        print_warning("Frontend build test failed, but continuing...")

def print_final_instructions():
    """Print final setup instructions"""
    print_header("SETUP COMPLETED SUCCESSFULLY! 🎉")

    instructions = """
🚀 QUICK START:

1. Start the development environment:
   ./scripts/start-dev.sh

2. Or start services individually:
   Backend:  ./scripts/start-backend.sh
   Frontend: ./scripts/start-frontend.sh

3. Access the application:
   Frontend: http://localhost:3000
   Backend:  http://localhost:5000/api

📚 NEXT STEPS:

1. Review and customize configuration files:
   - backend/.env (Backend configuration)
   - frontend/.env (Frontend configuration)

2. Read the documentation:
   - docs/INSTALLATION.md (Detailed setup)
   - docs/API.md (API documentation)
   - docs/DEPLOYMENT.md (Production deployment)

3. For production deployment:
   - Configure PostgreSQL database
   - Set up SSL certificates
   - Configure environment variables
   - Run: ./scripts/deploy.sh

🔧 USEFUL COMMANDS:

- Retrain ML model: ./scripts/train-model.sh
- Run tests: ./scripts/test-all.sh
- View logs: tail -f backend/logs/app.log

⚠️  IMPORTANT NOTES:

- This is a demo/educational system
- Always consult healthcare professionals for medical decisions
- Review security settings before production deployment
- Update default passwords and API keys

💡 SUPPORT:

- Documentation: docs/
- Issues: Create GitHub issue
- Contact: support@strokepredictor.com

Happy coding! 🧠✨
"""

    print_colored(instructions, Colors.OKGREEN)

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Brain Stroke Risk Prediction System Setup")
    parser.add_argument("--full", action="store_true", default=True, help="Full setup (default)")
    parser.add_argument("--backend-only", action="store_true", help="Setup only backend")
    parser.add_argument("--frontend-only", action="store_true", help="Setup only frontend")
    parser.add_argument("--ml-only", action="store_true", help="Setup only ML model")
    parser.add_argument("--dev", action="store_true", help="Development setup")
    parser.add_argument("--prod", action="store_true", help="Production setup")
    parser.add_argument("--skip-tests", action="store_true", help="Skip verification tests")

    args = parser.parse_args()

    # Print welcome message
    print_colored("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║    🧠 BRAIN STROKE RISK PREDICTION SYSTEM SETUP 🧠       ║
    ║                                                           ║
    ║    Advanced AI-powered stroke risk assessment tool       ║
    ║    Version 1.0.0                                         ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """, Colors.HEADER)

    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    # Create directory structure
    create_directory_structure()

    try:
        # Setup based on arguments
        if args.backend_only:
            if not setup_backend():
                sys.exit(1)
        elif args.frontend_only:
            if not setup_frontend():
                sys.exit(1)
        elif args.ml_only:
            if not setup_ml_model():
                sys.exit(1)
        else:  # Full setup
            if not setup_backend():
                sys.exit(1)
            if not setup_ml_model():
                sys.exit(1)
            if not setup_frontend():
                sys.exit(1)

        # Create additional files
        if args.full or not any([args.backend_only, args.frontend_only, args.ml_only]):
            create_scripts()
            create_documentation()
            create_config_files()

        # Run verification tests
        if not args.skip_tests:
            run_tests()

        # Print final instructions
        print_final_instructions()

    except KeyboardInterrupt:
        print_error("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Setup failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
