#!/bin/bash

set -e  # Exit on any error

echo "Starting build process..."

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

# Verify critical packages
echo "Verifying package installations..."
python -c "import streamlit; print('Streamlit OK')"
python -c "import sklearn; print('Scikit-learn OK')"
python -c "import nltk; print('NLTK OK')"
python -c "import pickle; print('Pickle OK')"

# Create NLTK data directory
echo "Creating NLTK data directory..."
mkdir -p nltk_data

# Download NLTK data with verification
echo "Downloading NLTK data..."
python << 'EOF'
import nltk
import ssl
import os
import sys

# Handle SSL issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Set NLTK data path
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
print(f"NLTK data directory: {nltk_data_dir}")

if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

nltk.data.path.append(nltk_data_dir)

# Download with verification
success = True

try:
    print("Downloading punkt_tab...")
    nltk.download('punkt_tab', download_dir=nltk_data_dir)
    # Verify download
    nltk.data.find('tokenizers/punkt_tab')
    print('✓ punkt_tab downloaded and verified')
except Exception as e:
    print(f'✗ Error with punkt_tab: {e}')
    success = False

try:
    print("Downloading stopwords...")
    nltk.download('stopwords', download_dir=nltk_data_dir)
    # Verify download
    nltk.data.find('corpora/stopwords')
    print('✓ stopwords downloaded and verified')
except Exception as e:
    print(f'✗ Error with stopwords: {e}')
    success = False

if not success:
    print("NLTK download failed!")
    sys.exit(1)
else:
    print("All NLTK data downloaded successfully!")

EOF

echo "Build completed successfully!"
