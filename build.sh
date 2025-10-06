#!/bin/bash

# Build script for Brain Stroke Prediction Full-Stack App
# This script builds both the React frontend and prepares the Flask backend for deployment

set -e  # Exit on any error

echo "🧠 Building Brain Stroke Risk Prediction Full-Stack App..."
echo "=" * 60

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found. Make sure you're in the project root directory."
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Check if frontend directory exists
if [ ! -d "frontend" ]; then
    echo "❌ Error: frontend directory not found"
    exit 1
fi

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
cd frontend
npm install

# Build React frontend
echo "🔨 Building React frontend..."
npm run build

# Go back to root directory
cd ..

# Create static directory for Flask
echo "📂 Setting up static files for Flask..."
mkdir -p backend/static

# Copy built React files to Flask static directory
echo "📋 Copying built frontend to backend static folder..."
cp -r frontend/build/* backend/static/

echo "✅ Build completed successfully!"
echo ""
echo "🚀 The app is now ready for deployment."
echo "Frontend files are served from: backend/static/"
echo "Flask backend serves both API and frontend routes"
echo ""
echo "To run locally:"
echo "  cd backend && python app.py"
echo ""
echo "The app will be available at: http://localhost:5000"
