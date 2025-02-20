import nltk
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

def ensure_nltk_data():
    """Ensure all required NLTK data is downloaded"""
    required_packages = [
        'punkt',
        'punkt_tab',
        'averaged_perceptron_tagger',
        'wordnet',
        'omw-1.4'
    ]
    
    # Set NLTK data path to a directory we can write to
    nltk_data_dir = Path.home() / 'nltk_data'
    nltk_data_dir.mkdir(exist_ok=True)
    nltk.data.path.append(str(nltk_data_dir))
    
    try:
        for package in required_packages:
            try:
                nltk.data.find(f'tokenizers/{package}')
                logger.info(f"Package {package} already exists")
            except LookupError:
                logger.info(f"Downloading {package}")
                nltk.download(package, download_dir=str(nltk_data_dir))
                
        logger.info("All NLTK packages downloaded successfully")
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {str(e)}")
        raise 