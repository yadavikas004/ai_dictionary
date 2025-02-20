from http.client import HTTPException
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.routes import word_routes
from app.database import engine, Base, get_db, init_db
from app.services.ml_service import MLService
import logging
import os
import warnings
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
import csv
from io import StringIO
from sqlalchemy.orm import Session
from .models import UserSearchHistory  # Assuming you have a model for search history
import sys
import nltk
from contextlib import asynccontextmanager
from pathlib import Path

# Load environment variables
load_dotenv()

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Suppress uvicorn access logs
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Global ML Service instance
ml_service = None

def download_nltk_data():
    """Download required NLTK data"""
    try:
        # Set NLTK data path to current directory
        current_dir = Path(__file__).parent.parent
        nltk_data_dir = current_dir / 'nltk_data'
        nltk_data_dir.mkdir(exist_ok=True)
        
        # Set NLTK data path
        nltk.data.path.append(str(nltk_data_dir))
        
        # Required NLTK packages
        required_packages = [
            'punkt',
            'averaged_perceptron_tagger',
            'wordnet'
        ]
        
        # Download required packages
        for package in required_packages:
            try:
                logger.info(f"Downloading NLTK package: {package}")
                nltk.download(package, download_dir=str(nltk_data_dir), quiet=True)
                logger.info(f"Successfully downloaded {package}")
            except Exception as e:
                logger.error(f"Error downloading {package}: {str(e)}")
                raise
                
        logger.info("All NLTK packages downloaded successfully")
        
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {str(e)}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI application"""
    # Startup
    try:
        logger.info("Application starting up...")
        
        # Initialize database
        logger.info("Initializing database...")
        init_db()
        
        # Download NLTK data
        logger.info("Downloading NLTK data...")
        download_nltk_data()
        
        # Initialize ML Service
        logger.info("Initializing ML Service...")
        global ml_service
        ml_service = MLService()
        logger.info("ML Service initialized successfully!")
        
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise
    
    yield  # Server is running
    
    # Shutdown
    try:
        logger.info("Application shutting down gracefully...")
        # Cleanup code here if needed
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")

# Create FastAPI app with lifespan
app = FastAPI(
    title="AI Dictionary API",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    word_routes.router,
    prefix="/api",
    tags=["words"]
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ml_service": "initialized" if ml_service and ml_service._initialized else "not initialized"
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    try:
        # Check NLTK data
        nltk_status = all(
            nltk.data.find(f'tokenizers/{pkg}') 
            for pkg in ['punkt', 'averaged_perceptron_tagger', 'wordnet']
        )
        
        return {
            "status": "healthy",
            "database": "connected",
            "ml_service": bool(ml_service and ml_service._initialized),
            "nltk_data": "downloaded" if nltk_status else "missing"
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/api/search-history")
async def download_history(db: Session = Depends(get_db)):
    try:
        from app.models.word import UserSearchHistory, Word
        
        # Query search history with word text
        history = (
            db.query(
                UserSearchHistory.timestamp,
                Word.text.label('word')
            )
            .join(Word)
            .order_by(UserSearchHistory.timestamp.desc())
            .all()
        )
        
        return [
            {
                "timestamp": entry.timestamp.isoformat(),
                "word": entry.word
            }
            for entry in history
        ]
    except Exception as e:
        logger.error(f"Error fetching search history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching search history: {str(e)}"
        )

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception handler: {str(exc)}")
    return {"detail": str(exc)}, 500
