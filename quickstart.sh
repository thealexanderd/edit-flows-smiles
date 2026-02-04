#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Quick Start Guide for SMILES Edit Flow

echo "=========================================="
echo "SMILES Edit Flow - Quick Start"
echo "=========================================="
echo ""
echo "Virtual environment: ACTIVATED ✓"
echo ""
echo "Available commands:"
echo ""
echo "1. Run core tests (no GPU needed):"
echo "   python test_core.py"
echo ""
echo "2. Run all unit tests:"
echo "   python -m pytest tests/ -v"
echo ""
echo "3. Train on tiny dataset (quick test):"
echo "   python smiles_editflow/train.py --tiny"
echo ""
echo "4. Train on custom data:"
echo "   python smiles_editflow/train.py --data your_data.txt --steps 1000"
echo ""
echo "5. Test individual modules:"
echo "   python smiles_editflow/tokenizer.py"
echo ""
echo "=========================================="
echo ""
echo "Note: You're now in the virtual environment."
echo "To deactivate when done, type: deactivate"
echo ""
