#!/bin/bash

# Brain Stroke Risk Prediction System - Stop Script
# This script stops both the backend API and frontend React app

echo "🛑 Brain Stroke Risk Prediction System"
echo "======================================"
echo "Stopping the application..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to kill process by PID file
kill_process_by_pidfile() {
    local pidfile=$1
    local service_name=$2

    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo "   Stopping $service_name (PID: $pid)..."
            kill "$pid" 2>/dev/null

            # Wait for process to terminate
            local count=0
            while kill -0 "$pid" 2>/dev/null && [ $count -lt 10 ]; do
                sleep 1
                count=$((count + 1))
            done

            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                echo "   Force stopping $service_name..."
                kill -9 "$pid" 2>/dev/null
            fi

            echo -e "   ${GREEN}✅ $service_name stopped${NC}"
        else
            echo -e "   ${YELLOW}⚠️  $service_name process not running${NC}"
        fi
        rm -f "$pidfile"
    else
        echo -e "   ${YELLOW}⚠️  No PID file found for $service_name${NC}"
    fi
}

# Function to kill processes by name
kill_processes_by_name() {
    local process_name=$1
    local service_name=$2

    local pids=$(pgrep -f "$process_name" 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "   Found $service_name processes: $pids"
        for pid in $pids; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "   Stopping $service_name (PID: $pid)..."
                kill "$pid" 2>/dev/null
                sleep 1

                # Force kill if still running
                if kill -0 "$pid" 2>/dev/null; then
                    kill -9 "$pid" 2>/dev/null
                fi
            fi
        done
        echo -e "   ${GREEN}✅ $service_name processes stopped${NC}"
    else
        echo -e "   ${YELLOW}⚠️  No $service_name processes found${NC}"
    fi
}

# Stop processes using PID files
echo "🔍 Checking for running processes..."

# Stop frontend
if [ -f "logs/frontend.pid" ]; then
    kill_process_by_pidfile "logs/frontend.pid" "Frontend"
else
    # Try to find and stop React development server
    kill_processes_by_name "react-scripts start" "Frontend (react-scripts)"
fi

# Stop backend
if [ -f "logs/backend.pid" ]; then
    kill_process_by_pidfile "logs/backend.pid" "Backend"
else
    # Try to find and stop Python Flask server
    kill_processes_by_name "simple_app.py" "Backend (Flask)"
fi

# Additional cleanup - stop any processes on the default ports
echo ""
echo "🧹 Cleaning up processes on default ports..."

# Check port 3000 (React default)
if lsof -ti:3000 >/dev/null 2>&1; then
    echo "   Stopping processes on port 3000..."
    lsof -ti:3000 | xargs kill -9 2>/dev/null
    echo -e "   ${GREEN}✅ Port 3000 cleared${NC}"
fi

# Check port 5000 (Flask default)
if lsof -ti:5000 >/dev/null 2>&1; then
    echo "   Stopping processes on port 5000..."
    lsof -ti:5000 | xargs kill -9 2>/dev/null
    echo -e "   ${GREEN}✅ Port 5000 cleared${NC}"
fi

# Clean up log files if they exist
echo ""
echo "🗂️  Cleaning up log files..."
if [ -d "logs" ]; then
    rm -f logs/backend.pid logs/frontend.pid
    if [ -f "logs/backend.log" ]; then
        echo "   Backend log saved: logs/backend.log"
    fi
    if [ -f "logs/frontend.log" ]; then
        echo "   Frontend log saved: logs/frontend.log"
    fi
else
    echo "   No log directory found"
fi

echo ""
echo "✅ Application stopped successfully!"
echo ""
echo "📊 To start the application again, run:"
echo "   ./start_app.sh"
echo ""
echo "📋 To check if any processes are still running:"
echo "   ps aux | grep -E '(react-scripts|simple_app.py)'"
echo ""
echo "🔍 To check port usage:"
echo "   lsof -i:3000  # Frontend port"
echo "   lsof -i:5000  # Backend port"
echo ""
