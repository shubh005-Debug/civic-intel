#!/bin/bash
echo "=== CityPulse Smart City Setup ==="
pip install -r requirements.txt
python train_model.py
echo ""
echo "=== Setup complete! Run with: python app.py ==="
