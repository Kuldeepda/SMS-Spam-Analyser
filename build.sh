#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt

# Create NLTK data directory
mkdir -p nltk_data

# Download NLTK data
python -c "
import nltk
import ssl
import os

# Handle SSL issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Set NLTK data path
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_dir)

# Download required resources
try:
    nltk.download('punkt_tab', download_dir=nltk_data_dir)
    print('Downloaded punkt_tab')
except Exception as e:
    print(f'Error downloading punkt_tab: {e}')

try:
    nltk.download('stopwords', download_dir=nltk_data_dir)
    print('Downloaded stopwords')
except Exception as e:
    print(f'Error downloading stopwords: {e}')
"

echo "Build completed successfully!"
