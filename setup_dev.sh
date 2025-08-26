#!/bin/bash
echo "Setting up ENVnet development environment..."

# Add blink submodule (update with correct URL)
echo "Adding blink submodule..."
git submodule add https://github.com/biorack/blink.git external/blink

# Update submodules
git submodule update --init --recursive

# Install envnet in editable mode
echo "Installing envnet in editable mode..."
pip install -e .

echo "Setup complete!"
echo "Next steps:"
echo "1. Copy metatlas functions to envnet/vendor/"
echo "2. Copy your deconvolution code to envnet/deconvolution/"
echo "3. Update imports in your migrated code"