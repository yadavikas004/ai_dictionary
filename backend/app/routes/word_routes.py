from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.word import Word, UserSearchHistory
from app.services.word_service import (
    get_meaning,
    pos_tagging,
    word_tokenization,
    word_lemmatization,
    get_word_analysis,
    get_advanced_analysis,
    store_word_data
)
from pydantic import BaseModel
from typing import Dict, List
import logging
from sqlalchemy import func, desc
from datetime import datetime

router = APIRouter()
logger = logging.getLogger(__name__)

class WordCreate(BaseModel):
    text: str
    meaning: str
    language: str = "English"

class WordResponse(WordCreate):
    id: int

class TokenizationRequest(BaseModel):
    text: str

class LemmatizationRequest(BaseModel):
    text: str

@router.post("/words/", response_model=WordResponse)
def add_word(word_data: WordCreate, db: Session = Depends(get_db)):
    db_word = Word(text=word_data.text, meaning=word_data.meaning)
    db.add(db_word)
    db.commit()
    db.refresh(db_word)
    return db_word

@router.get("/word/{word_text}")
def get_word(word_text: str, db: Session = Depends(get_db)):
    word = db.query(Word).filter(Word.text == word_text).first()
    if not word:
        raise HTTPException(status_code=404, detail="Word not found")

    # Store search history
    search_entry = UserSearchHistory(word_id=word.id)
    db.add(search_entry)
    db.commit()

    return {"word": word.text, "meaning": word.meaning}

@router.get("/meaning/{word}")
async def get_word_meaning(
    word: str,
    db: Session = Depends(get_db)
):
    """Get word meaning with database caching"""
    try:
        word_data = await get_meaning(word, db)
        
        # Add to search history
        db_word = await store_word_data(db, word, word_data)
        
        history_entry = UserSearchHistory(word_id=db_word.id)
        db.add(history_entry)
        db.commit()
        
        return word_data
        
    except Exception as e:
        logger.error(f"Error processing word '{word}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing word: {str(e)}"
        )

@router.post("/pos-tagging/")
def get_pos_tagging(text: str):
    return {"text": text, "pos_tags": pos_tagging(text)}

@router.post("/tokenize/", response_model=List[str])
def get_tokenization(request: TokenizationRequest, db: Session = Depends(get_db)):
    """Tokenize the input text."""
    try:
        tokens = word_tokenization(request.text)
        return tokens
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/lemmatize/", response_model=List[str])
def get_lemmatization(request: LemmatizationRequest, db: Session = Depends(get_db)):
    """Lemmatize the input text."""
    try:
        lemmas = word_lemmatization(request.text)
        return lemmas
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_search_history(
    page: int = 1,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get search history"""
    skip = (page - 1) * limit
    total = db.query(Word).count()
    words = db.query(Word).order_by(Word.id.desc()).offset(skip).limit(limit).all()
    
    return {
        "words": [{"id": word.id, "text": word.text} for word in words],
        "total_pages": (total + limit - 1) // limit,
        "current_page": page,
        "total_items": total
    }

@router.post("/search-history")
async def add_to_search_history(
    word_id: int,
    db: Session = Depends(get_db)
):
    # Create new search history entry
    history_entry = UserSearchHistory(word_id=word_id)
    db.add(history_entry)
    db.commit()
    return {"message": "Search history updated"}

@router.get("/analyze/{word}")
async def analyze_word(word: str):
    """Get complete word analysis"""
    try:
        analysis = await get_word_analysis(word)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/advanced/{word}")
async def advanced_analysis(word: str):
    """Get advanced ML-based analysis"""
    try:
        analysis = await get_advanced_analysis(word)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search-history", response_model=List[Dict])
async def get_search_history(db: Session = Depends(get_db)):
    """Get search history with word details"""
    try:
        # Query to get search history with word counts and last searched time
        history = (
            db.query(
                Word.text.label('word'),
                func.count(UserSearchHistory.id).label('count'),
                func.max(UserSearchHistory.timestamp).label('last_searched')
            )
            .join(UserSearchHistory, Word.id == UserSearchHistory.word_id)
            .group_by(Word.id, Word.text)
            .order_by(desc(func.max(UserSearchHistory.timestamp)))
            .all()
        )

        if not history:
            return []

        # Format the response
        formatted_history = [
            {
                "word": str(item.word),
                "count": int(item.count),
                "last_searched": item.last_searched.isoformat() if item.last_searched else None
            }
            for item in history
        ]

        return formatted_history

    except Exception as e:
        logger.error(f"Error fetching search history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch search history"
        )

@router.get("/search-history/counts", response_model=List[Dict])
async def get_search_frequency(db: Session = Depends(get_db)):
    """Get word search frequency"""
    try:
        # Query to get search frequency
        frequency = (
            db.query(
                Word.text.label('word'),
                func.count(UserSearchHistory.id).label('count')
            )
            .join(UserSearchHistory, Word.id == UserSearchHistory.word_id)
            .group_by(Word.id, Word.text)
            .order_by(desc('count'))
            .all()
        )

        if not frequency:
            return []

        # Format the response
        formatted_frequency = [
            {
                "word": str(item.word),
                "count": int(item.count)
            }
            for item in frequency
        ]

        return formatted_frequency

    except Exception as e:
        logger.error(f"Error fetching search frequency: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch search frequency"
        )

@router.post("/search-history/{word_id}")
async def add_to_history(word_id: int, db: Session = Depends(get_db)):
    """Add a word to search history"""
    try:
        # Verify word exists
        word = db.query(Word).filter(Word.id == word_id).first()
        if not word:
            raise HTTPException(status_code=404, detail="Word not found")

        # Add history entry
        history_entry = UserSearchHistory(
            word_id=word_id,
            timestamp=datetime.utcnow()
        )
        db.add(history_entry)
        db.commit()
        
        return {"status": "success", "message": "Added to history"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding to history: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Failed to add to history"
        )
