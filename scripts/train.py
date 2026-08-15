#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training pipeline for the Unified Social Media Threat Detector.

Combines the feature engineering of both source projects:
- fake-profile-detector: account metrics, activity, content, network, image features
- spam_identifier: sentiment, country, account type, word frequency, link features

This script:
1. Generates a realistic labeled training dataset (combining both projects' schema)
2. Trains and compares multiple ML models
3. Selects the best model and persists it (model + scaler + encoders + feature names)
4. Saves a sample training dataset to data/training_data.csv for reproducibility
"""

import os
import sys
import logging
import argparse

import numpy as np
import pandas as pd

# Ensure project root is importable when run from scripts/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.train_model import ModelTrainer, generate_synthetic_training_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(n_samples: int = 5000, model_filename: str = 'threat_detector_model.pkl') -> str:
    """
    Run the full training pipeline.

    Args:
        n_samples: Number of synthetic training samples to generate.
        model_filename: Name of the model file to save under models/.

    Returns:
        Path to the saved model.
    """
    logger.info("=" * 70)
    logger.info("UNIFIED THREAT DETECTOR - MODEL TRAINING PIPELINE")
    logger.info("=" * 70)

    # 1. Generate combined training data (fake-profile-detector + spam_identifier schema)
    df = generate_synthetic_training_data(n_samples=n_samples)

    # Persist the dataset so training is reproducible / inspectable
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, 'training_data.csv')
    df.to_csv(data_path, index=False)
    logger.info(f"Training dataset saved to {data_path} ({len(df)} rows)")

    # 2. Prepare data with the unified trainer
    trainer = ModelTrainer(model_dir=os.path.join(PROJECT_ROOT, 'models'))
    X, y = trainer.prepare_data(df, target_col='is_threat')

    # 3. Train and compare candidate models
    results = trainer.train_and_compare(X, y)
    logger.info("\nModel Comparison (sorted by F1):")
    logger.info("-" * 60)
    for name, metrics in sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True):
        logger.info(f"{name:20s} | Acc: {metrics['accuracy']:.3f} | "
                    f"F1: {metrics['f1']:.3f} | AUC: {metrics['auc']:.3f}")

    # 4. Train the best model on all data and persist it
    trainer.train_best_model(X, y)
    model_path = trainer.save_model(filename=model_filename)

    # 5. Report top features
    importance = trainer.get_feature_importance()
    logger.info("\nTop 10 Most Important Features:")
    logger.info("-" * 60)
    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"{feature:30s} | {score:.4f}")

    logger.info("=" * 70)
    logger.info(f"DONE. Best model '{trainer.best_model_name}' saved to {model_path}")
    logger.info("=" * 70)

    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Unified Threat Detector model")
    parser.add_argument('--samples', type=int, default=5000,
                        help='Number of synthetic training samples (default: 5000)')
    parser.add_argument('--model', type=str, default='threat_detector_model.pkl',
                        help='Model filename to save under models/')
    args = parser.parse_args()

    main(n_samples=args.samples, model_filename=args.model)
