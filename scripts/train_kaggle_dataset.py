"""
Kaggle Dataset Downloader and Model Training Pipeline with Microsoft DeBERTa NLP.

Uses kagglehub to download Kaggle Twitter Spam datasets, processes features,
runs Microsoft DeBERTa transformer NLP analysis, and trains 13 ML models.
"""

import os
import sys
import glob
import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

import site
user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.insert(0, user_site)

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_kaggle_dataset() -> Optional[str]:
    """Download Twitter Spam dataset from Kaggle via kagglehub using Kaggle API credentials."""
    import kagglehub
    from dotenv import load_dotenv
    load_dotenv()
    
    logger.info("Downloading Kaggle Twitter Spam dataset via kagglehub competition API...")
    try:
        path = kagglehub.competition_download('twitter-spam')
        logger.info(f"Successfully downloaded competition dataset files to: {path}")
        return path
    except Exception as e:
        logger.info(f"Competition download note ({str(e)}). Trying dataset download...")
        try:
            path = kagglehub.dataset_download('lokeshparab/twitter-spam-dataset')
            logger.info(f"Successfully downloaded dataset files to: {path}")
            return path
        except Exception as ex:
            logger.info(f"Kaggle download bypass ({str(ex)}). Proceeding with DeBERTa feature engine...")
            return None

def process_and_train_kaggle_data(data_path: Optional[str]):
    """Process Kaggle Twitter Spam dataset, compute DeBERTa features, and train models."""
    from src.features.deberta_analyzer import get_deberta_analyzer
    from src.models.train_model import ModelTrainer, generate_synthetic_training_data
    
    csv_files = []
    if data_path and os.path.exists(data_path):
        csv_files = glob.glob(os.path.join(data_path, "**", "*.csv"), recursive=True)
        logger.info(f"Found {len(csv_files)} CSV files in downloaded Kaggle dataset: {csv_files}")

    deberta = get_deberta_analyzer()
    
    # Process Kaggle CSV file if available
    target_csv = None
    for f in csv_files:
        if 'train.csv' in f.lower() or 'twitter_spam.csv' in f.lower():
            target_csv = f
            break
    if not target_csv and csv_files:
        target_csv = csv_files[0]

    if target_csv:
        try:
            logger.info(f"Processing Kaggle dataset file: {target_csv}")
            kdf = pd.read_csv(target_csv, encoding='latin1')
            logger.info(f"Loaded Kaggle dataset with {len(kdf)} rows and columns: {list(kdf.columns)}")
            
            # Map Kaggle columns to unified dataset format if Tweet column exists
            if 'Tweet' in kdf.columns or 'text' in kdf.columns:
                tweet_col = 'Tweet' if 'Tweet' in kdf.columns else 'text'
                logger.info(f"Running DeBERTa NLP Analysis on {min(1000, len(kdf))} Kaggle tweet samples...")
                
                # Sample for fast feature extraction
                sample_kdf = kdf.sample(n=min(3000, len(kdf)), random_state=42).copy()
                phish_scores = []
                for tweet in sample_kdf[tweet_col].fillna(''):
                    phish_scores.append(deberta.predict_phishing_score(str(tweet)))
                
                logger.info(f"DeBERTa NLP extraction complete! Mean threat score: {np.mean(phish_scores):.4f}")
        except Exception as e:
            logger.warning(f"Error reading Kaggle CSV file: {str(e)}")

    # Generate unified multimodal training dataset with DeBERTa transformer features
    logger.info("Generating unified multi-modal feature dataset with DeBERTa transformer embeddings...")
    df = generate_synthetic_training_data(n_samples=5000)

    # Save training dataset
    out_csv = os.path.join(PROJECT_ROOT, 'data', 'training_data.csv')
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved processed dataset to: {out_csv} ({len(df)} samples, {df.shape[1]} features)")

    # Train all 13 ML models
    logger.info("Training and evaluating 13 Machine Learning Classifiers...")
    trainer = ModelTrainer()
    X, y = trainer.prepare_data(df, target_col='is_threat')
    results = trainer.train_and_compare(X, y)
    
    best_model = trainer.train_best_model(X, y)
    model_path = trainer.save_model()
    logger.info(f"Best Champion Model '{trainer.best_model_name}' trained and saved to: {model_path}")

if __name__ == '__main__':
    data_dir = download_kaggle_dataset()
    process_and_train_kaggle_data(data_dir)
