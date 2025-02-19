from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.routes.word_routes import router as word_router
from app.database import create_tables, get_db
import logging
import os
import warnings
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
import csv
from io import StringIO
from sqlalchemy.orm import Session
from .models import UserSearchHistory  # Assuming you have a model for search history

# Load environment variables
load_dotenv()

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
warnings.filterwarnings('ignore')  # Suppress Python warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable Tensorflow logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("CORS_ORIGIN", "http://localhost:5173")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(word_router, prefix="/api")

# Create tables on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Creating database tables...")
    create_tables()
    logger.info("Database tables created successfully!")

@app.get("/")
async def root():
    return {"message": "Welcome to AI Dictionary API"}

@app.get("/download-history")
async def download_history(db: Session = Depends(get_db)):
    # Fetch search history from the database
    history = db.query(UserSearchHistory).all()

    # Create a CSV in memory
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Query', 'Timestamp'])  # Header row
    for record in history:
        writer.writerow([record.id, record.query, record.timestamp])  # Adjust based on your model

    output.seek(0)  # Move to the beginning of the StringIO buffer

    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=search_history.csv"}
    )
