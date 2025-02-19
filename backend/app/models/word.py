from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime, UTC
from app.database import Base

class Word(Base):
    __tablename__ = "words"
    
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String(255), unique=True, nullable=False)
    meaning = Column(Text(length=4294967295), nullable=False)
    search_history = relationship("UserSearchHistory", back_populates="word")

class UserSearchHistory(Base):
    __tablename__ = "search_history"
    
    id = Column(Integer, primary_key=True, index=True)
    word_id = Column(Integer, ForeignKey("words.id"))
    timestamp = Column(DateTime, default=lambda: datetime.now(UTC))
    
    word = relationship("Word", back_populates="search_history")
