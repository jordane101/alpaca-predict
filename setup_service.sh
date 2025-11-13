#!/bin/bash
# Setup script for Alpaca Trading Bot systemctl service

set -e

PROJECT_DIR="/home/eli/alpaca-predict"
SERVICE_FILE="alpaca-trader.service"
SYSTEMD_DIR="/etc/systemd/system"

echo "=== Alpaca Trading Bot Service Setup ==="
echo ""

# Check if running as root for installation
if [ "$1" == "install" ]; then
    if [ "$EUID" -ne 0 ]; then 
        echo "❌ Please run 'sudo $0 install' to install the service"
        exit 1
    fi
    
    echo "📦 Installing service..."
    
    # Copy service file to systemd
    cp "$PROJECT_DIR/$SERVICE_FILE" "$SYSTEMD_DIR/$SERVICE_FILE"
    echo "✓ Service file copied to $SYSTEMD_DIR/$SERVICE_FILE"
    
    # Reload systemd
    systemctl daemon-reload
    echo "✓ Systemd daemon reloaded"
    
    # Enable service to start on boot
    systemctl enable $SERVICE_FILE
    echo "✓ Service enabled to start on boot"
    
    echo ""
    echo "✅ Service installed successfully!"
    echo ""
    echo "To start the service:"
    echo "  sudo systemctl start alpaca-trader"
    echo ""
    echo "To check status:"
    echo "  sudo systemctl status alpaca-trader"
    echo ""
    echo "To view logs:"
    echo "  tail -f $PROJECT_DIR/data/logs/trader_service.log"
    echo ""

elif [ "$1" == "uninstall" ]; then
    if [ "$EUID" -ne 0 ]; then 
        echo "❌ Please run 'sudo $0 uninstall' to uninstall the service"
        exit 1
    fi
    
    echo "🗑️  Uninstalling service..."
    
    # Stop service if running
    systemctl stop $SERVICE_FILE 2>/dev/null || true
    echo "✓ Service stopped"
    
    # Disable service
    systemctl disable $SERVICE_FILE 2>/dev/null || true
    echo "✓ Service disabled"
    
    # Remove service file
    rm -f "$SYSTEMD_DIR/$SERVICE_FILE"
    echo "✓ Service file removed"
    
    # Reload systemd
    systemctl daemon-reload
    echo "✓ Systemd daemon reloaded"
    
    echo ""
    echo "✅ Service uninstalled successfully!"
    echo ""

elif [ "$1" == "test" ]; then
    echo "🧪 Testing service configuration..."
    echo ""
    
    # Check if service file exists
    if [ ! -f "$PROJECT_DIR/$SERVICE_FILE" ]; then
        echo "❌ Service file not found: $PROJECT_DIR/$SERVICE_FILE"
        exit 1
    fi
    echo "✓ Service file exists"
    
    # Check if Python script exists
    if [ ! -f "$PROJECT_DIR/scripts/trader.py" ]; then
        echo "❌ Trader script not found: $PROJECT_DIR/scripts/trader.py"
        exit 1
    fi
    echo "✓ Trader script exists"
    
    # Check if virtual environment exists
    if [ ! -f "$PROJECT_DIR/.venv/bin/python3" ]; then
        echo "⚠️  Virtual environment not found at $PROJECT_DIR/.venv/"
        echo "   You may need to create it: python3 -m venv .venv"
    else
        echo "✓ Virtual environment exists"
    fi
    
    # Check if .env file exists
    if [ ! -f "$PROJECT_DIR/.env" ]; then
        echo "❌ .env file not found: $PROJECT_DIR/.env"
        echo "   Create it with your Alpaca API keys"
        exit 1
    fi
    echo "✓ .env file exists"
    
    # Check if log directory exists
    mkdir -p "$PROJECT_DIR/data/logs"
    echo "✓ Log directory exists"
    
    # Test Python import
    echo ""
    echo "Testing Python imports..."
    cd "$PROJECT_DIR"
    if [ -f ".venv/bin/python3" ]; then
        .venv/bin/python3 -c "
import sys
sys.path.insert(0, '/home/eli/alpaca-predict')
from src.trading.orchestrator import Orchestrator
from src.trading.strategies import HMMStrategy
print('✓ Python imports successful')
" 2>&1
    else
        python3 -c "
import sys
sys.path.insert(0, '/home/eli/alpaca-predict')
from src.trading.orchestrator import Orchestrator
from src.trading.strategies import HMMStrategy
print('✓ Python imports successful')
" 2>&1
    fi
    
    echo ""
    echo "✅ All tests passed!"
    echo ""
    echo "To install the service:"
    echo "  sudo $0 install"
    echo ""

elif [ "$1" == "status" ]; then
    echo "📊 Service Status:"
    echo ""
    systemctl status alpaca-trader --no-pager || echo "Service not installed or not running"
    echo ""
    echo "Recent logs:"
    tail -n 20 "$PROJECT_DIR/data/logs/trader_service.log" 2>/dev/null || echo "No logs yet"

else
    echo "Usage: $0 {install|uninstall|test|status}"
    echo ""
    echo "Commands:"
    echo "  test       - Test configuration before installing"
    echo "  install    - Install and enable the systemd service (requires sudo)"
    echo "  uninstall  - Stop and remove the systemd service (requires sudo)"
    echo "  status     - Show service status and recent logs"
    echo ""
    echo "Examples:"
    echo "  $0 test                    # Test configuration"
    echo "  sudo $0 install            # Install service"
    echo "  sudo systemctl start alpaca-trader    # Start service"
    echo "  sudo systemctl stop alpaca-trader     # Stop service"
    echo "  $0 status                  # Check status"
    echo "  sudo $0 uninstall          # Remove service"
    exit 1
fi
