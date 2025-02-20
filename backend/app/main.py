from http.client import HTTPException
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.routes import word_routes
from app.database import get_db, init_db
from app.services.ml_service import MLService
import logging
import os
import warnings
from dotenv import load_dotenv
from sqlalchemy.orm import Session
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

def setup_nltk():
    """Set up NLTK data"""
    try:
        # Set NLTK data path
        current_dir = Path(__file__).parent.parent
        nltk_data_dir = current_dir / 'nltk_data'
        nltk_data_dir.mkdir(exist_ok=True)
        nltk.data.path.append(str(nltk_data_dir))
        
        # Download required packages
        packages = ['punkt', 'averaged_perceptron_tagger', 'wordnet']
        for package in packages:
            try:
                nltk.download(package, download_dir=str(nltk_data_dir), quiet=True)
            except Exception:
                pass  # Skip if already downloaded
    except Exception as e:
        logger.error(f"NLTK setup error: {str(e)}")
        return False
    return True

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI application"""
    # Startup
    try:
        logger.info("Initializing application...")
        
        # Initialize database
        init_db()
        
        # Setup NLTK
        setup_nltk()
        
        # Initialize ML Service
        global ml_service
        ml_service = MLService()
        
        logger.info("Application initialized successfully!")
        
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise
    
    yield  # Application is running
    
    # Shutdown
    try:
        logger.info("Shutting down application...")
        # Cleanup code here if needed
        if ml_service:
            del ml_service
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")

# Create FastAPI app with lifespan manager
app = FastAPI(
    title="AI Dictionary API",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
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
    """Root endpoint"""
    return {
        "status": "healthy",
        "ml_service": "initialized" if ml_service and ml_service._initialized else "not initialized"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "database": "connected",
            "ml_service": bool(ml_service and ml_service._initialized),
            "nltk_data": "downloaded" if setup_nltk() else "missing"
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

# Error handler for graceful shutdown
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {str(exc)}")
    return {"detail": "Internal server error"}, 500
