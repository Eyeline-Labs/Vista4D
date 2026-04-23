#!/bin/bash

# Vista4D camera UI startup script
# Starts both the Python/Viser backend and React frontend

set -e  # Exit on error

echo "======================================"
echo "Vista4D user interface startup"
echo "======================================"

# Get the directory where this script is located (cam_ui/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# The base repo directory is one level up - Python must run from here for utils.media imports
REPO_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
# ^Using SCRIPT_DIR and REPO_DIR makes the script work even if you don't run it from the base directory

# Ports (configures all scripts)
VISER_PORT=9997
FASTAPI_PORT=9998
REACT_PORT=9999

# Trap to cleanup background processes on exit
cleanup() {
    echo ""
    echo "======================================"
    echo "Shutting down servers..."
    echo "======================================"

    if [ ! -z "$PYTHON_PID" ]; then
        echo "Stopping Python server (PID: $PYTHON_PID)"
        kill $PYTHON_PID 2>/dev/null || true
    fi

    if [ ! -z "$REACT_PID" ]; then
        echo "Stopping React dev server (PID: $REACT_PID)"
        kill $REACT_PID 2>/dev/null || true
    fi

    echo "[OK] Cleanup complete"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Start Python/Viser server in background
# IMPORTANT: Run from the repo root so that `from utils.media import ...` resolves
echo ""
echo "======================================"
echo "Starting Python/Viser Server"
echo "======================================"
cd "$REPO_DIR"

if [ ! -f "cam_ui/python_server/main.py" ]; then
    echo "[ERROR] cam_ui/python_server/main.py not found"
    exit 1
fi

PYTHONPATH="$REPO_DIR" PYTHONUNBUFFERED=1 python -u cam_ui/python_server/main.py --viser-port $VISER_PORT --fastapi-port $FASTAPI_PORT &
PYTHON_PID=$!

echo "[OK] Python server starting (PID: $PYTHON_PID)"
echo "   Viser UI:   http://localhost:$VISER_PORT"
echo "   FastAPI:    http://localhost:$FASTAPI_PORT"

# Wait for the Python server to start and verify it's running
echo ""
echo "Waiting for Python server to start..."
MAX_WAIT=60
WAITED=0
until curl -s "http://localhost:$FASTAPI_PORT/api/health" > /dev/null 2>&1; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "[ERROR] Python server did not start after ${MAX_WAIT}s"
        exit 1
    fi
    sleep 1
    WAITED=$((WAITED + 1))
done
echo "[OK] Python server is ready (after ${WAITED}s)"

# Start React dev server
echo ""
echo "======================================"
echo "Starting React Server"
echo "======================================"
cd "$SCRIPT_DIR/react_app"

if [ ! -f "package.json" ]; then
    echo "[ERROR] package.json not found in cam_ui/react_app/"
    exit 1
fi

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "[WARN] node_modules not found. Running npm install..."
    npm install
fi

echo "Building React app..."
VISER_PORT=$VISER_PORT FASTAPI_PORT=$FASTAPI_PORT REACT_PORT=$REACT_PORT npm run build
echo "[OK] Build complete"

VISER_PORT=$VISER_PORT FASTAPI_PORT=$FASTAPI_PORT REACT_PORT=$REACT_PORT npm run preview &
REACT_PID=$!

echo "[OK] React preview server started (PID: $REACT_PID)"
echo "   React App:  http://localhost:$REACT_PORT"

echo ""
echo "======================================"
echo "All servers running!"
echo "======================================"
echo "Viser UI:   http://localhost:$VISER_PORT"
echo "FastAPI:    http://localhost:$FASTAPI_PORT"
echo "React App:  http://localhost:$REACT_PORT"
echo ""
echo "Press Ctrl+C to stop all servers"
echo "======================================"

# Wait for all background processes
wait
