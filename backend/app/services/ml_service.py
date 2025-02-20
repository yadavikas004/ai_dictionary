import tensorflow as tf
import torch
from transformers import (
    BertTokenizer,
    BertModel,
    AutoModelForSequenceClassification,
    pipeline
)
import numpy as np
import logging
import os
import warnings
from typing import Dict, List, Optional
# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

logger = logging.getLogger(__name__)

class MLService:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize ML models using singleton pattern"""
        if self._initialized:
            return
            
        try:
            logger.info("Initializing ML models...")
            
            # Initialize models with offline mode first, then online if needed
            try:
                # Try loading models from cache first
                self.bert_tokenizer = BertTokenizer.from_pretrained(
                    'bert-base-uncased',
                    local_files_only=True
                )
                self.bert_model = BertModel.from_pretrained(
                    'bert-base-uncased',
                    local_files_only=True
                )
                self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                    'bert-base-uncased',
                    num_labels=2,
                    local_files_only=True
                )
            except Exception as cache_error:
                logger.warning("Cache miss, downloading models...")
                # If cache miss, download models
                self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                self.bert_model = BertModel.from_pretrained('bert-base-uncased')
                self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                    'bert-base-uncased',
                    num_labels=2
                )
            
            # Move models to CPU (or GPU if available)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.bert_model = self.bert_model.to(self.device)
            self.sentiment_model = self.sentiment_model.to(self.device)
            
            # Set models to evaluation mode
            self.bert_model.eval()
            self.sentiment_model.eval()
            
            self._initialized = True
            logger.info(f"ML models initialized successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {str(e)}")
            self._initialized = False
            raise

    def get_sentiment(self, text: str) -> Dict:
        """Get sentiment analysis"""
        if not self._initialized:
            return {"label": "NEUTRAL", "score": 0.5}
            
        try:
            # Tokenize input
            inputs = self.bert_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            ).to(self.device)
            
            # Get sentiment
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get probabilities
            negative_prob, positive_prob = predictions[0].cpu().numpy()
            
            return {
                "label": "POSITIVE" if positive_prob > negative_prob else "NEGATIVE",
                "score": float(max(positive_prob, negative_prob))
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {"label": "NEUTRAL", "score": 0.5}

    def get_word_embedding(self, word: str) -> Optional[np.ndarray]:
        """Get BERT embedding for a word"""
        if not self._initialized:
            return np.zeros(768)
            
        try:
            # Tokenize input
            inputs = self.bert_tokenizer(
                word,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            ).to(self.device)
            
            # Get embedding
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error getting word embedding: {str(e)}")
            return np.zeros(768)

    def initialize_models(self):
        """Initialize all ML models"""
        try:
            # BERT Models
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
            
            # TensorFlow Word Embedding Model
            self.embedding_dim = 50
            self.vocab_size = 10000
            self.tf_model = self.create_tf_model()
            
            logger.info("ML models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ML models: {str(e)}")
            raise

    def create_tf_model(self) -> tf.keras.Model:
        """Create TensorFlow model for word embeddings"""
        return tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu')
        ])

    def get_bert_embeddings(self, word: str) -> List[float]:
        """Get BERT word embeddings"""
        try:
            inputs = self.bert_tokenizer(word, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            return embeddings.tolist()[:5]  # Return first 5 dimensions
        except Exception as e:
            logger.error(f"Error getting BERT embeddings: {str(e)}")
            return [0.0] * 5

    def get_tf_analysis(self, word: str) -> List[float]:
        """Get TensorFlow model analysis"""
        try:
            # Convert word to sequence
            sequence = self.bert_tokenizer.encode(word, padding=True, truncation=True)
            # Get predictions
            predictions = self.tf_model.predict([sequence], verbose=0)
            return predictions[0].tolist()
        except Exception as e:
            logger.error(f"Error in TF analysis: {str(e)}")
            return [0.0] * 16

    def __del__(self):
        """Cleanup when the service is destroyed"""
        try:
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
