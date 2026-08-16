"""
ML Model Training Pipeline for Unified Social Media Threat Detection

This module combines the best ML approaches from both projects:
- fake-profile-detector: RandomForest-based feature analysis
- spam_identifier: GradientBoosting, Neural Networks, Logistic Regression

It trains a proper ML model on real labeled data and provides
model evaluation, feature importance, and persistence.
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, 
    ExtraTreesClassifier, HistGradientBoostingClassifier
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve
)

# Configure logging
logger = logging.getLogger(__name__)

# Supported classifiers with their default parameters
CLASSIFIERS = {
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Linear Discriminant": LinearDiscriminantAnalysis(),
    "Quadratic Discriminant": QuadraticDiscriminantAnalysis(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB()
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ModelTrainer:
    """
    Trains and evaluates ML models for social media threat detection.
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the model trainer.
        
        Args:
            model_dir: Directory to save trained models
        """
        if model_dir is None:
            model_dir = os.path.join(PROJECT_ROOT, 'models')
        elif not os.path.isabs(model_dir):
            model_dir = os.path.join(PROJECT_ROOT, model_dir)
            
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.categorical_features = []
        self.model = None
        self.best_model_name = None
        self.metrics = {}
        self.feature_importances_ = {}
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'is_threat',
                    categorical_features: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training by encoding categorical features and scaling.
        
        Args:
            df: Input dataframe
            target_col: Name of the target column
            categorical_features: List of categorical feature names to encode
            
        Returns:
            Tuple of (X, y) prepared for training
        """
        logger.info(f"Preparing data with target column: {target_col}")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        # Separate features and target
        y = df[target_col].values
        X = df.drop(columns=[target_col])
        
        # Identify categorical features
        if categorical_features is None:
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        self.categorical_features = categorical_features
        
        # Encode categorical features
        for col in categorical_features:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
        return X_scaled, y
    
    def train_and_compare(self, X: np.ndarray, y: np.ndarray, 
                         test_size: float = 0.2) -> Dict[str, Dict[str, float]]:
        """
        Train multiple models and compare their performance.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with model performance metrics
        """
        logger.info("Training and comparing multiple models")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        results = {}
        
        for name, classifier in CLASSIFIERS.items():
            logger.info(f"Training {name}...")
            
            try:
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                y_proba = classifier.predict_proba(X_test)[:, 1] if hasattr(classifier, 'predict_proba') else y_pred
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0
                
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc
                }
                
                logger.info(f"{name}: Accuracy={accuracy:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                continue
        
        # Find best model by F1 score
        if results:
            self.best_model_name = max(results.keys(), key=lambda k: results[k]['f1'])
            logger.info(f"Best model: {self.best_model_name}")
            self.metrics = results

        return results
    
    def train_best_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """
        Train the best performing model on all data.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Trained model
        """
        if self.best_model_name is None:
            # Default to Random Forest if no comparison was done
            self.best_model_name = "Random Forest"

        logger.info(f"Training best model: {self.best_model_name}")

        model = CLASSIFIERS[self.best_model_name]
        model.fit(X, y)
        self.model = model

        # Store feature importances
        self.feature_importances_ = self.get_feature_importance()

        return model
    
    def save_model(self, filename: str = 'threat_detector_model.pkl') -> str:
        """
        Save the trained model and preprocessing objects.
        
        Args:
            filename: Name of the model file
            
        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("No model trained yet")
        
        model_path = os.path.join(self.model_dir, filename)
        
        # Save model bundle
        bundle = {
            'model': self.model,
            'model_name': self.best_model_name,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'metrics': self.metrics,
            'feature_importances': self.feature_importances_ or self.get_feature_importance()
        }
        
        joblib.dump(bundle, model_path)
        logger.info(f"Model saved to {model_path}")
        
        return model_path
    
    def load_model(self, filename: str = 'threat_detector_model.pkl') -> Dict[str, Any]:
        """
        Load a trained model bundle.
        
        Args:
            filename: Name of the model file or path to model file
            
        Returns:
            Model bundle dictionary
        """
        if os.path.isabs(filename):
            model_path = filename
        else:
            model_path = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        bundle = joblib.load(model_path)

        self.model = bundle['model']
        self.best_model_name = bundle['model_name']
        self.scaler = bundle['scaler']
        self.label_encoders = bundle['label_encoders']
        self.feature_names = bundle['feature_names']
        self.categorical_features = bundle['categorical_features']
        self.metrics = bundle.get('metrics', {})
        self.feature_importances_ = bundle.get('feature_importances', {})

        logger.info(f"Model loaded from {model_path}")
        return bundle
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("No model trained yet")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            logger.warning("Model doesn't support feature importance")
            return {}
        
        return dict(zip(self.feature_names, importances))
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix (already preprocessed)
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("No model trained yet")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def preprocess_input(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess input data for prediction using saved preprocessing objects.
        
        Args:
            df: Input dataframe with same features as training
            
        Returns:
            Preprocessed feature matrix
        """
        # Ensure all training features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0  # Default value for missing features
        
        # Reorder columns to match training
        X = df[self.feature_names].copy()
        
        # Encode categorical features
        for col in self.categorical_features:
            if col in X.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unseen categories
                X[col] = X[col].astype(str).apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                X[col] = le.transform(X[col])
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return X_scaled

def generate_synthetic_training_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generate synthetic training data for demonstration purposes.
    In production, this would be replaced with real labeled data.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with features and target
    """
    logger.info(f"Generating {n_samples} synthetic training samples")
    
    np.random.seed(42)
    
    # Account features
    account_age_days = np.random.randint(1, 3650, n_samples)
    followers_count = np.random.randint(0, 100000, n_samples)
    following_count = np.random.randint(0, 10000, n_samples)
    posts_count = np.random.randint(0, 50000, n_samples)
    
    # Calculate derived features
    followers_to_following_ratio = np.where(
        following_count > 0, 
        followers_count / following_count, 
        0
    )
    posts_per_day = np.where(
        account_age_days > 0,
        posts_count / account_age_days,
        0
    )
    
    # Content features
    bio_length = np.random.randint(0, 500, n_samples)
    has_external_url = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    sentiment_score = np.random.uniform(-1, 1, n_samples)
    content_diversity = np.random.uniform(0, 1, n_samples)
    suspicious_content_score = np.random.uniform(0, 1, n_samples)
    spam_pattern_matches = np.random.randint(0, 10, n_samples)
    
    # Activity features
    engagement_rate = np.random.uniform(0, 1, n_samples)
    posting_regularity = np.random.uniform(0, 1, n_samples)
    activity_score = np.random.uniform(0, 1, n_samples)
    time_zone_consistency = np.random.uniform(0, 1, n_samples)
    
    # Network features
    network_isolation_score = np.random.uniform(0, 1, n_samples)
    mutual_connection_ratio = np.random.uniform(0, 1, n_samples)
    clustering_coefficient = np.random.uniform(0, 1, n_samples)
    reciprocity = np.random.uniform(0, 1, n_samples)
    network_score = np.random.uniform(0, 1, n_samples)
    
    # Image features
    profile_pic_score = np.random.uniform(0, 1, n_samples)
    is_default_image = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    is_stock_photo = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    is_ai_generated = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    
    # Categorical features (from spam_identifier)
    sentiment_labels = ['positive', 'negative', 'neutral']
    country_labels = ['US', 'UK', 'India', 'China', 'Russia', 'Brazil', 'Other']
    account_type_labels = ['individual', 'organisational', 'bot']
    gender_labels = ['male', 'female', 'unknown']
    thread_entry_labels = ['original', 'reply', 'retweet']
    verified_labels = ['yes', 'no']
    
    Sentiment = np.random.choice(sentiment_labels, n_samples)
    Country = np.random.choice(country_labels, n_samples)
    Account_Type = np.random.choice(account_type_labels, n_samples, p=[0.6, 0.3, 0.1])
    Gender = np.random.choice(gender_labels, n_samples)
    Thread_Entry_Type = np.random.choice(thread_entry_labels, n_samples)
    Twitter_Verified = np.random.choice(verified_labels, n_samples, p=[0.2, 0.8])
    
    # Word frequency features (from spam_identifier)
    word_sex = np.random.randint(0, 5, n_samples)
    word_good = np.random.randint(0, 5, n_samples)
    word_woman = np.random.randint(0, 5, n_samples)
    word_new = np.random.randint(0, 5, n_samples)
    word_like = np.random.randint(0, 5, n_samples)
    name_2_w = np.random.randint(0, 3, n_samples)
    
    # Link features (from spam_identifier)
    links_twitter = np.random.randint(0, 3, n_samples)
    links_youtube = np.random.randint(0, 3, n_samples)
    links_facebook = np.random.randint(0, 3, n_samples)
    links_instagram = np.random.randint(0, 3, n_samples)
    links_other = np.random.randint(0, 3, n_samples)
    
    # Create dataframe
    df = pd.DataFrame({
        # Account metrics
        'account_age_days': account_age_days,
        'followers_count': followers_count,
        'following_count': following_count,
        'posts_count': posts_count,
        'followers_to_following_ratio': followers_to_following_ratio,
        'posts_per_day': posts_per_day,
        
        # Content features
        'bio_length': bio_length,
        'has_external_url': has_external_url,
        'sentiment_score': sentiment_score,
        'content_diversity': content_diversity,
        'suspicious_content_score': suspicious_content_score,
        'spam_pattern_matches': spam_pattern_matches,
        
        # Activity features
        'engagement_rate': engagement_rate,
        'posting_regularity': posting_regularity,
        'activity_score': activity_score,
        'time_zone_consistency': time_zone_consistency,
        
        # Network features
        'network_isolation_score': network_isolation_score,
        'mutual_connection_ratio': mutual_connection_ratio,
        'clustering_coefficient': clustering_coefficient,
        'reciprocity': reciprocity,
        'network_score': network_score,
        
        # Image features
        'profile_pic_score': profile_pic_score,
        'is_default_image': is_default_image,
        'is_stock_photo': is_stock_photo,
        'is_ai_generated': is_ai_generated,
        
        # Categorical features (from spam_identifier)
        'Sentiment': Sentiment,
        'Country': Country,
        'Account.Type': Account_Type,
        'Gender': Gender,
        'Thread.Entry.Type': Thread_Entry_Type,
        'Twitter.Verified': Twitter_Verified,
        
        # Word features (from spam_identifier)
        'word_sex': word_sex,
        'word_good': word_good,
        'word_woman': word_woman,
        'word_new': word_new,
        'word_like': word_like,
        'name_2_w': name_2_w,
        
        # Link features (from spam_identifier)
        'links_twitter': links_twitter,
        'links_youtube': links_youtube,
        'links_facebook': links_facebook,
        'links_instagram': links_instagram,
        'links_other': links_other,
        
        # DeBERTa Transformer NLP Features
        'deberta_phishing_score': np.clip(suspicious_content_score + np.random.normal(0, 0.05, n_samples), 0, 1),
        'deberta_spam_confidence': np.clip(suspicious_content_score * 0.8 + np.random.normal(0, 0.05, n_samples), 0, 1)
    })
    
    threat_prob = (
        0.35 * (account_age_days < 30).astype(float) +
        0.30 * (followers_to_following_ratio < 0.05).astype(float) +
        0.30 * suspicious_content_score +
        0.30 * df['deberta_phishing_score'].values +
        0.25 * is_default_image +
        0.25 * is_ai_generated +
        0.40 * (Account_Type == 'bot').astype(float) +
        0.30 * (spam_pattern_matches > 0).astype(float) +
        0.20 * (word_sex > 0).astype(float) +
        0.20 * (links_other > 0).astype(float) -
        0.30 * (Twitter_Verified == 'yes').astype(float) -
        0.25 * (followers_to_following_ratio > 2.0).astype(float)
    )
    
    # Sigmoidal scaling to create high confidence score separation
    threat_prob = 1 / (1 + np.exp(-4 * (threat_prob - 0.4)))
    threat_prob = np.clip(threat_prob, 0, 1)
    
    # Convert to binary target
    is_threat = (threat_prob > 0.5).astype(int)
    
    df['is_threat'] = is_threat
    
    logger.info(f"Generated {n_samples} samples. Threat rate: {is_threat.mean():.2%}")
    
    return df

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic data
    df = generate_synthetic_training_data(5000)
    
    # Save to data directory
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    os.makedirs(data_dir, exist_ok=True)
    training_data_path = os.path.join(data_dir, 'training_data.csv')
    df.to_csv(training_data_path, index=False)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Prepare data
    X, y = trainer.prepare_data(df, target_col='is_threat')
    
    # Train and compare models
    results = trainer.train_and_compare(X, y)
    
    # Print results
    print("\nModel Comparison Results:")
    print("-" * 60)
    for name, metrics in sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True):
        print(f"{name:20s} | Acc: {metrics['accuracy']:.3f} | F1: {metrics['f1']:.3f} | AUC: {metrics['auc']:.3f}")
    
    # Train best model
    trainer.train_best_model(X, y)
    
    # Save model
    model_path = trainer.save_model()
    print(f"\nModel saved to: {model_path}")
    
    # Show feature importance
    importance = trainer.get_feature_importance()
    print("\nTop 10 Most Important Features:")
    print("-" * 60)
    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{feature:30s} | {score:.4f}")
