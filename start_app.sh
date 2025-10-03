#!/bin/bash

# Brain Stroke Risk Prediction System - Startup Script
# This script starts both the backend API and frontend React app

echo "🧠 Brain Stroke Risk Prediction System"
echo "======================================"
echo "Starting the complete application..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a port is in use
port_in_use() {
    lsof -i:$1 >/dev/null 2>&1
}

# Function to wait for a service to be ready
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=30
    local attempt=1

    echo "Waiting for $name to be ready..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" >/dev/null 2>&1; then
            echo -e "${GREEN}✅ $name is ready!${NC}"
            return 0
        fi
        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done

    echo -e "${RED}❌ $name failed to start within $max_attempts seconds${NC}"
    return 1
}

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command_exists python3; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi

if ! command_exists npm; then
    echo -e "${RED}❌ Node.js/npm is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"
echo ""

# Check if backend dependencies are installed
echo "📦 Checking backend dependencies..."
if python3 -c "import flask, pandas, numpy, sklearn, joblib" 2>/dev/null; then
    echo -e "${GREEN}✅ Backend dependencies are installed${NC}"
else
    echo -e "${YELLOW}⚠️  Installing backend dependencies...${NC}"
    cd backend
    python3 -m pip install flask flask-cors pandas numpy scikit-learn joblib --user
    cd ..
fi

# Check if frontend dependencies are installed
echo "📦 Checking frontend dependencies..."
if [ -d "frontend/node_modules" ]; then
    echo -e "${GREEN}✅ Frontend dependencies are installed${NC}"
else
    echo -e "${YELLOW}⚠️  Installing frontend dependencies...${NC}"
    cd frontend
    npm install --legacy-peer-deps
    cd ..
fi

echo ""

# Check if ports are available
BACKEND_PORT=5000
FRONTEND_PORT=3000

if port_in_use $BACKEND_PORT; then
    echo -e "${YELLOW}⚠️  Port $BACKEND_PORT is already in use${NC}"
    echo "   You may need to stop the existing service or use a different port"
fi

if port_in_use $FRONTEND_PORT; then
    echo -e "${YELLOW}⚠️  Port $FRONTEND_PORT is already in use${NC}"
    echo "   You may need to stop the existing service or use a different port"
fi

# Create log directory
mkdir -p logs

# Start backend server
echo "🚀 Starting backend server..."
cd backend
python3 simple_app.py > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
cd ..

echo "   Backend PID: $BACKEND_PID"
echo "   Backend logs: logs/backend.log"

# Wait for backend to be ready
if wait_for_service "http://localhost:$BACKEND_PORT" "Backend API"; then
    echo ""

    # Start frontend server
    echo "🎨 Starting frontend server..."
    cd frontend
    BROWSER=none npm start > ../logs/frontend.log 2>&1 &
    FRONTEND_PID=$!
    cd ..

    echo "   Frontend PID: $FRONTEND_PID"
    echo "   Frontend logs: logs/frontend.log"

    # Wait for frontend to be ready
    if wait_for_service "http://localhost:$FRONTEND_PORT" "Frontend App"; then
        echo ""
        echo "🎉 Application started successfully!"
        echo ""
        echo "📍 Access the application:"
        echo "   🌐 Web App:     http://localhost:$FRONTEND_PORT"
        echo "   🔗 API:        http://localhost:$BACKEND_PORT"
        echo "   📚 API Docs:   http://localhost:$BACKEND_PORT/api/info"
        echo ""
        echo "👤 Demo Login Credentials:"
        echo "   📧 Email:      demo@strokeprediction.com"
        echo "   🔒 Password:   demo123"
        echo ""
        echo "📊 Features Available:"
        echo "   • 🧠 AI-powered stroke risk assessment"
        echo "   • 📈 Real-time prediction results"
        echo "   • 📋 Assessment history tracking"
        echo "   • 👤 User profile management"
        echo "   • 📱 Mobile-responsive design"
        echo ""
        echo "🛑 To stop the application:"
        echo "   Press Ctrl+C or run: ./stop_app.sh"
        echo ""
        echo -e "${YELLOW}⚠️  MEDICAL DISCLAIMER:${NC}"
        echo "   This system is for educational purposes only."
        echo "   Always consult healthcare professionals for medical decisions."
        echo ""

        # Save PIDs for cleanup
        echo $BACKEND_PID > logs/backend.pid
        echo $FRONTEND_PID > logs/frontend.pid

        # Keep the script running and handle cleanup on exit
        cleanup() {
            echo ""
            echo "🛑 Shutting down application..."

            if [ -f logs/frontend.pid ]; then
                FRONTEND_PID=$(cat logs/frontend.pid)
                if kill -0 $FRONTEND_PID 2>/dev/null; then
                    echo "   Stopping frontend (PID: $FRONTEND_PID)..."
                    kill $FRONTEND_PID 2>/dev/null
                fi
                rm -f logs/frontend.pid
            fi

            if [ -f logs/backend.pid ]; then
                BACKEND_PID=$(cat logs/backend.pid)
                if kill -0 $BACKEND_PID 2>/dev/null; then
                    echo "   Stopping backend (PID: $BACKEND_PID)..."
                    kill $BACKEND_PID 2>/dev/null
                fi
                rm -f logs/backend.pid
            fi

            echo "✅ Application stopped successfully"
            exit 0
        }

        # Set up signal handlers
        trap cleanup SIGINT SIGTERM

        # Wait for user to stop the application
        echo "Application is running... Press Ctrl+C to stop"
        while true; do
            sleep 1

            # Check if processes are still running
            if ! kill -0 $BACKEND_PID 2>/dev/null; then
                echo -e "${RED}❌ Backend process died unexpectedly${NC}"
                echo "   Check logs/backend.log for errors"
                break
            fi

            if ! kill -0 $FRONTEND_PID 2>/dev/null; then
                echo -e "${RED}❌ Frontend process died unexpectedly${NC}"
                echo "   Check logs/frontend.log for errors"
                break
            fi
        done

    else
        echo -e "${RED}❌ Frontend failed to start${NC}"
        kill $BACKEND_PID 2>/dev/null
        exit 1
    fi
else
    echo -e "${RED}❌ Backend failed to start${NC}"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi
