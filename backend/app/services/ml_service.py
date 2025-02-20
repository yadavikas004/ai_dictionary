import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertModel
import tensorflow as tf
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class MLService:
    def __init__(self):
        self.initialize_models()

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

    def get_sentiment(self, word: str) -> Dict:
        """Get sentiment analysis using BERT"""
        try:
            inputs = self.bert_tokenizer(word, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            return {
                "positive": float(scores[0][1]),
                "negative": float(scores[0][0])
            }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {"error": str(e)}

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
