import nltk
from pathlib import Path
import sys

def download_nltk_packages():
    """Download required NLTK packages"""
    try:
        # Set NLTK data path
        current_dir = Path(__file__).parent.parent
        nltk_data_dir = current_dir / 'nltk_data'
        nltk_data_dir.mkdir(exist_ok=True)
        
        # Required packages
        packages = ['punkt', 'averaged_perceptron_tagger', 'wordnet']
        
        # Download packages
        for package in packages:
            print(f"Downloading {package}...")
            nltk.download(package, download_dir=str(nltk_data_dir), quiet=True)
            print(f"Successfully downloaded {package}")
            
        print("All NLTK packages downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        return False

if __name__ == "__main__":
    success = download_nltk_packages()
    sys.exit(0 if success else 1) 