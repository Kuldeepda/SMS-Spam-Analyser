#!/bin/bash

# Optimize for faster builds and startups
echo "Optimizing for Render deployment..."

# Use pip cache and parallel processing
pip install --upgrade pip
pip install --no-warn-script-location --user -r requirements.txt

# Pre-create and populate NLTK directory
echo "Setting up NLTK data..."
mkdir -p /opt/render/project/src/nltk_data/tokenizers/punkt_tab
mkdir -p /opt/render/project/src/nltk_data/corpora/stopwords

# Download NLTK data to permanent location
python -c "
import nltk
import ssl
import os

# Handle SSL
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Set permanent NLTK data path
nltk_data_dir = '/opt/render/project/src/nltk_data'
os.environ['NLTK_DATA'] = nltk_data_dir
nltk.data.path.append(nltk_data_dir)

# Download efficiently
print('Downloading punkt_tab...')
nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)
print('Downloading stopwords...')
nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
print('NLTK setup complete!')
"

# Verify model files exist
if [ -f "vectorizer.pkl" ] && [ -f "model.pkl" ]; then
    echo "✓ Model files found"
else
    echo "⚠ Warning: Model files not found"
fi

echo "Build optimization complete!"
