import requests
from googletrans import Translator
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertModel
import spacy
from typing import List, Dict, Tuple
import torch
import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging
from nltk.corpus import wordnet

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize the translation and definition pipelines
translator = Translator()
definition_pipeline = pipeline("text2text-generation", model="t5-small")

# Initialize the text generation pipeline
generator = pipeline('text-generation', model='gpt2')

# Initialize other tools
lemmatizer = WordNetLemmatizer()

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Initialize models
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
sentiment_analyzer = pipeline('sentiment-analysis')

# Word embeddings model (TensorFlow)
embedding_dim = 50
vocab_size = 10000
embedding_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.GlobalAveragePooling1D()
])

# Initialize logger
logger = logging.getLogger(__name__)

def get_meaning(word: str) -> Dict:
    """Get comprehensive word analysis"""
    try:
        # Get basic meanings
        meanings = get_dictionary_meaning(word)
        
        # Enhanced AI analysis
        ai_analysis = {
            "meanings": meanings,
            "sentiment": analyze_sentiment(word),
            "similar_words": find_similar_words(word),
            "word_vector": get_word_vector(word),
            "context_examples": generate_context(word)
        }
        
        return ai_analysis
        
    except Exception as e:
        logger.error(f"Error analyzing '{word}': {str(e)}")
        return {"error": f"Unable to analyze: {word}"}

def get_dictionary_meaning(word: str) -> List[str]:
    """Get dictionary meanings"""
    try:
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word.lower()}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            meanings = []
            
            for entry in data:
                for meaning in entry.get('meanings', []):
                    for definition in meaning.get('definitions', []):
                        if definition.get('definition'):
                            meanings.append(definition['definition'])
            
            return meanings if meanings else ["No meaning found"]
            
    except Exception as e:
        logger.error(f"Error fetching meaning: {str(e)}")
        return ["Unable to fetch meaning"]

def analyze_sentiment(word: str) -> Dict:
    """Analyze word sentiment using BERT"""
    try:
        result = sentiment_analyzer(word)[0]
        return {
            "label": result["label"],
            "score": float(result["score"])
        }
    except Exception as e:
        return {"error": str(e)}

def get_word_vector(word: str) -> List[float]:
    """Get word embedding using BERT"""
    try:
        inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        # Get the word embedding and convert to list
        word_vector = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        # Return first 5 dimensions for readability
        return word_vector[:5] if isinstance(word_vector, list) else word_vector.tolist()[:5]
    except Exception as e:
        logger.error(f"Error getting word vector: {str(e)}")
        return [0.0] * 5

def find_similar_words(word: str) -> List[str]:
    """Find similar words using WordNet"""
    try:
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() != word:
                    synonyms.add(lemma.name())
        
        # Get related words from dictionary API
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word.lower()}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for entry in data:
                for meaning in entry.get('meanings', []):
                    synonyms.update(meaning.get('synonyms', []))
        
        return list(synonyms)[:5] if synonyms else [f"No similar words found for: {word}"]
    except Exception as e:
        logger.error(f"Error finding similar words: {str(e)}")
        return [f"Error finding similar words: {str(e)}"]

def generate_context(word: str) -> List[str]:
    """Generate example contexts using dictionary API and templates"""
    try:
        examples = []
        
        # Get examples from dictionary API
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word.lower()}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for entry in data:
                for meaning in entry.get('meanings', []):
                    for definition in meaning.get('definitions', []):
                        if definition.get('example'):
                            examples.append(definition['example'])
        
        # If no examples found, generate some
        if not examples:
            templates = [
                f"The {word} was used in this context.",
                f"Here's an example with '{word}'.",
                f"You might use '{word}' in a sentence like this."
            ]
            examples = templates
        
        return examples[:3]  # Return up to 3 examples
    except Exception as e:
        logger.error(f"Error generating context: {str(e)}")
        return [f"Error generating examples: {str(e)}"]

def pos_tagging(text: str) -> List[Tuple[str, str]]:
    """Get POS tags using spaCy"""
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

def word_tokenization(text: str) -> List[str]:
    """Perform word tokenization using NLTK."""
    tokens = word_tokenize(text)
    return tokens

def word_lemmatization(text: str) -> List[str]:
    """Perform lemmatization using spaCy."""
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    return lemmas

def lemmatize_text(text: str) -> List[Tuple[str, str]]:
    """Perform lemmatization using both NLTK and spaCy"""
    # SpaCy lemmatization
    doc = nlp(text)
    spacy_lemmas = [(token.text, token.lemma_) for token in doc]
    
    # NLTK lemmatization
    words = word_tokenize(text)
    nltk_lemmas = [(word, lemmatizer.lemmatize(word)) for word in words]
    
    return list(set(spacy_lemmas + nltk_lemmas))

def get_word_analysis(word: str) -> Dict:
    """Get complete word analysis including meanings, POS, tokens, and lemmas"""
    text = word.strip().lower()
    
    return {
        "word": text,
        "meanings": get_meaning(text)["meanings"],
        "pos_tags": pos_tagging(text),
        "tokens": word_tokenization(text),
        "lemmas": lemmatize_text(text)
    }

# TensorFlow and PyTorch models for advanced processing
def initialize_ml_models():
    """Initialize TensorFlow and PyTorch models"""
    # TensorFlow text classification
    tf_model = tf.keras.Sequential([
        tf.keras.layers.Embedding(10000, 16),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # PyTorch transformer model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    pt_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    
    return tf_model, pt_model, tokenizer

def get_advanced_analysis(word: str) -> Dict:
    """Get advanced analysis using ML models"""
    tf_model, pt_model, tokenizer = initialize_ml_models()
    
    # Basic analysis
    basic_analysis = get_word_analysis(word)
    
    # PyTorch BERT analysis
    inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        bert_output = pt_model(**inputs)
    
    # TensorFlow analysis
    tf_output = tf_model.predict([word])
    
    # Combine all analyses
    return {
        **basic_analysis,
        "bert_analysis": bert_output.logits.numpy().tolist(),
        "tf_analysis": tf_output.tolist()
    }

def translate_text(text: str, target_language: str = "hi"):
    return translator.translate(text, dest=target_language).text
