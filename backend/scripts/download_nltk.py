import nltk
from pathlib import Path

def download_nltk_packages():
    # Create nltk_data directory in user's home
    nltk_data_dir = Path.home() / 'nltk_data'
    nltk_data_dir.mkdir(exist_ok=True)
    
    # Download required packages
    packages = [
        'punkt',
        'punkt_tab',
        'averaged_perceptron_tagger',
        'wordnet',
        'omw-1.4'
    ]
    
    for package in packages:
        print(f"Downloading {package}...")
        nltk.download(package, download_dir=str(nltk_data_dir))
        
    print("All packages downloaded successfully!")

if __name__ == "__main__":
    download_nltk_packages() 