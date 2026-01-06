#!/bin/bash

# Setup script for USD/TRY Forecasting Model
# This script creates a virtual environment and installs dependencies

echo "Setting up USD/TRY Forecasting Model..."
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing required packages..."
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "To run the forecasting model:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Run the script: python usd_try_forecast.py"
echo ""

