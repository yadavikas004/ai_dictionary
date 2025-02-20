from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base

class Word(Base):
    __tablename__ = "words"
    
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String(255), unique=True, nullable=False)
    meaning = Column(Text(length=4294967295), nullable=False)
    bert_embedding = Column(Text, nullable=True)  # Store BERT embeddings
    sentiment_score = Column(Float, nullable=True)  # Store sentiment
    similar_words = Column(Text, nullable=True)  # Store similar words as JSON
    context_examples = Column(Text, nullable=True)  # Store examples as JSON
    phonetics = Column(String(255), nullable=True)  # Store pronunciation
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationship with search history
    search_history = relationship("UserSearchHistory", back_populates="word")

class UserSearchHistory(Base):
    __tablename__ = "search_history"

    id = Column(Integer, primary_key=True, index=True)
    word_id = Column(Integer, ForeignKey("words.id"), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship with word
    word = relationship("Word", back_populates="search_history")
