#!/bin/bash

set -e  # Exit on any error

echo "Starting Render build process..."

# Install Python dependencies properly for Render
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Verify installations
echo "Verifying package installations..."
python -c "import streamlit; print('✓ Streamlit installed')"
python -c "import nltk; print('✓ NLTK installed')"
python -c "import sklearn; print('✓ Scikit-learn installed')"

# Create NLTK data directory
echo "Setting up NLTK data directory..."
mkdir -p nltk_data

# Download NLTK data
echo "Downloading NLTK data..."
python -c "
import nltk
import ssl
import os

# Handle SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Set NLTK data directory
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

nltk.data.path.append(nltk_data_dir)

# Download required NLTK data
try:
    print('Downloading punkt_tab...')
    nltk.download('punkt_tab', download_dir=nltk_data_dir)
    print('✓ punkt_tab downloaded successfully')
except Exception as e:
    print(f'Error downloading punkt_tab: {e}')

try:
    print('Downloading stopwords...')
    nltk.download('stopwords', download_dir=nltk_data_dir)
    print('✓ stopwords downloaded successfully')
except Exception as e:
    print(f'Error downloading stopwords: {e}')

print('NLTK data setup complete!')
"

# Verify model files
if [ -f "vectorizer.pkl" ] && [ -f "model.pkl" ]; then
    echo "✓ Model files verified"
else
    echo "⚠️  Warning: Model files not found"
    ls -la *.pkl 2>/dev/null || echo "No .pkl files found in directory"
fi

# Test streamlit installation
streamlit --version

echo "Build completed successfully! 🎉"
