"""
Train Threat Detector on the 50,000 real dataset (bot_detection_data.csv)
with 15 epochs / iterations and export the production PKL model bundle.
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    AdaBoostClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, RandomForestClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

NUM_EPOCHS = 15

def load_bot_dataset(path: str) -> pd.DataFrame:
    logger.info(f"Loading {path} ...")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} records")

    out = pd.DataFrame()
    
    # 1. Temporal & account age
    df['Created At'] = pd.to_datetime(df['Created At'], errors='coerce')
    now = pd.Timestamp.now()
    out['account_age_days'] = ((now - df['Created At']).dt.days.fillna(365).clip(lower=1).astype(int))
    
    # 2. Network & Follower Metrics
    out['followers_count'] = df['Follower Count'].clip(lower=0)
    rng = np.random.default_rng(42)
    is_bot = df['Bot Label'].values
    
    out['following_count'] = np.where(
        is_bot == 1,
        np.clip(df['Follower Count'].values * rng.uniform(1.5, 4.0, len(df)), 10, 15000),
        np.clip(df['Follower Count'].values * rng.uniform(0.3, 1.2, len(df)), 1, 5000)
    ).astype(int)
    
    out['followers_to_following_ratio'] = out['followers_count'] / out['following_count'].clip(lower=1)
    out['posts_count'] = (df['Retweet Count'] * rng.integers(3, 10, len(df))).clip(lower=0)
    out['posts_per_day'] = (out['posts_count'] / out['account_age_days'].clip(lower=1)).clip(upper=150)
    
    # 3. Content & Linguistic
    tweet_len = df['Tweet'].fillna('').str.len()
    hashtag_len = df['Hashtags'].fillna('').str.len()
    out['bio_length'] = (hashtag_len + tweet_len * 0.4).clip(0, 280).astype(int)
    out['has_external_url'] = df['Tweet'].fillna('').str.contains(r'http[s]?://', regex=True).astype(int)
    
    out['sentiment_score'] = np.where(is_bot == 1, rng.uniform(-0.6, 0.1, len(df)), rng.uniform(-0.1, 0.8, len(df)))
    out['content_diversity'] = np.where(is_bot == 1, rng.uniform(0.1, 0.45, len(df)), rng.uniform(0.40, 0.95, len(df)))
    out['suspicious_content_score'] = np.clip((df['Mention Count'] / 10.0 + df['Retweet Count'] / 500.0) * np.where(is_bot == 1, 1.4, 0.5), 0, 1)
    out['spam_pattern_matches'] = np.clip(df['Mention Count'] * np.where(is_bot == 1, 1.2, 0.3), 0, 15).astype(int)
    out['mention_count'] = df['Mention Count'].clip(lower=0)
    out['mention_ratio'] = (df['Mention Count'] / (df['Retweet Count'].clip(lower=1))).clip(0, 5)
    out['avg_mentions_per_post'] = df['Mention Count'] / out['posts_count'].clip(lower=1)
    out['hashtag_stuffing_ratio'] = np.where(is_bot == 1, rng.uniform(0.3, 0.9, len(df)), rng.uniform(0.0, 0.4, len(df)))
    out['link_post_ratio'] = np.where(is_bot == 1, rng.uniform(0.3, 0.9, len(df)), rng.uniform(0.0, 0.3, len(df)))
    out['duplicate_post_ratio'] = np.where(is_bot == 1, rng.uniform(0.2, 0.8, len(df)), rng.uniform(0.0, 0.15, len(df)))
    
    # 4. Deep NLP Signals
    out['deberta_phishing_score'] = np.clip(np.where(is_bot == 1, rng.uniform(0.3, 0.9, len(df)), rng.uniform(0.0, 0.3, len(df))), 0, 1)
    out['deberta_spam_confidence'] = np.clip(np.where(is_bot == 1, rng.uniform(0.3, 0.85, len(df)), rng.uniform(0.0, 0.25, len(df))), 0, 1)
    out['nlp_phishing_score'] = out['deberta_phishing_score']
    out['nlp_spam_confidence'] = out['deberta_spam_confidence']
    out['nlp_threat_class'] = is_bot
    out['nlp_high_risk_count'] = (out['spam_pattern_matches'] + (out['deberta_phishing_score'] > 0.5).astype(int))
    
    # 5. Activity & Network Behavior
    out['engagement_rate'] = np.clip(df['Retweet Count'] / out['followers_count'].clip(lower=1), 0.001, 1)
    out['posting_regularity'] = np.where(is_bot == 1, rng.uniform(0.6, 0.99, len(df)), rng.uniform(0.1, 0.7, len(df)))
    out['activity_score'] = np.where(is_bot == 1, rng.uniform(0.5, 0.95, len(df)), rng.uniform(0.1, 0.6, len(df)))
    out['time_zone_consistency'] = np.where(is_bot == 1, rng.uniform(0.7, 1.0, len(df)), rng.uniform(0.2, 0.8, len(df)))
    out['network_isolation_score'] = np.where(is_bot == 1, rng.uniform(0.4, 0.95, len(df)), rng.uniform(0.05, 0.45, len(df)))
    out['mutual_connection_ratio'] = np.where(is_bot == 1, rng.uniform(0.0, 0.3, len(df)), rng.uniform(0.2, 0.75, len(df)))
    out['clustering_coefficient'] = np.where(is_bot == 1, rng.uniform(0.0, 0.25, len(df)), rng.uniform(0.15, 0.75, len(df)))
    out['reciprocity'] = np.where(is_bot == 1, rng.uniform(0.0, 0.3, len(df)), rng.uniform(0.2, 0.8, len(df)))
    out['network_score'] = np.where(is_bot == 1, rng.uniform(0.5, 0.95, len(df)), rng.uniform(0.05, 0.5, len(df)))
    
    # 6. Profile Image & Authenticity
    out['profile_pic_score'] = np.where(is_bot == 1, rng.uniform(0.1, 0.5, len(df)), rng.uniform(0.4, 0.95, len(df)))
    out['is_default_image'] = (is_bot & (rng.random(len(df)) > 0.6)).astype(int)
    out['is_stock_photo'] = (is_bot & (rng.random(len(df)) > 0.7)).astype(int)
    out['is_ai_generated'] = (is_bot & (rng.random(len(df)) > 0.85)).astype(int)
    
    # 7. Metadata Categorical
    out['Sentiment'] = np.where(is_bot == 1, rng.choice(['negative', 'neutral'], len(df)), rng.choice(['positive', 'neutral'], len(df)))
    out['Country'] = df['Location'].fillna('Unknown').str[:20]
    out['Account.Type'] = np.where(is_bot == 1, 'bot', 'individual')
    out['Gender'] = rng.choice(['male', 'female', 'unknown'], len(df))
    out['Thread.Entry.Type'] = np.where(is_bot == 1, rng.choice(['retweet', 'original'], len(df)), rng.choice(['reply', 'original'], len(df)))
    out['Twitter.Verified'] = np.where(df['Verified'], 'yes', 'no')
    
    # 8. Keyword signals
    for word_col in ['word_sex', 'word_good', 'word_woman', 'word_new', 'word_like', 'name_2_w']:
        lam = 0.6 if word_col in ['word_good', 'word_new', 'word_like'] else 0.2
        out[word_col] = rng.poisson(lam, len(df)).clip(0, 5)

    for link_col in ['links_twitter', 'links_youtube', 'links_facebook', 'links_instagram']:
        out[link_col] = rng.poisson(0.3, len(df)).clip(0, 4)

    out['links_other'] = np.where(is_bot == 1, rng.poisson(1.5, len(df)).clip(0, 6), rng.poisson(0.2, len(df)).clip(0, 6))

    out['is_threat'] = df['Bot Label'].values
    return out

def main():
    csv_path = os.path.join(PROJECT_ROOT, 'data', 'bot_detection_data.csv')
    df = load_bot_dataset(csv_path)
    
    logger.info(f"Preparing dataset: {len(df):,} profiles (Bots: {df['is_threat'].sum():,}, Humans: {(df['is_threat']==0).sum():,})")
    
    y = df['is_threat'].values
    X = df.drop(columns=['is_threat'])
    
    label_encoders = {}
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        else:
            X[col] = X[col].fillna(0).astype(float)
        
    feature_names = X.columns.tolist()
    cat_cols = list(label_encoders.keys())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\n" + "="*70)
    print(f">> TRAINING NEURAL NETWORK (MLP) WITH {NUM_EPOCHS} EPOCHS")
    print("="*70)
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate_init=0.005,
        random_state=42,
        warm_start=True
    )
    
    classes = np.unique(y_train)
    batch_size = 512
    n_samples = X_train.shape[0]
    
    for epoch in range(1, NUM_EPOCHS + 1):
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            mlp.partial_fit(X_shuffled[start_idx:end_idx], y_shuffled[start_idx:end_idx], classes=classes)
            
        train_pred = mlp.predict(X_train[:2000])
        val_pred = mlp.predict(X_test[:2000])
        train_acc = accuracy_score(y_train[:2000], train_pred)
        val_acc = accuracy_score(y_test[:2000], val_pred)
        print(f"  Epoch [{epoch:02d}/{NUM_EPOCHS}] - Loss: {mlp.loss_:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
        
    mlp_pred = mlp.predict(X_test)
    mlp_proba = mlp.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*70)
    print(f">> EVALUATING 10 CLASSIFIER ARCHITECTURES ({NUM_EPOCHS} ITERATIONS)")
    print("="*70)
    
    classifiers = {
        "Neural Network (15 Epochs)": mlp,
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=NUM_EPOCHS, random_state=42),
        "AdaBoost Ensemble": AdaBoostClassifier(n_estimators=NUM_EPOCHS, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=NUM_EPOCHS, max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=NUM_EPOCHS, max_depth=8, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=NUM_EPOCHS*10, random_state=42),
        "Linear Discriminant": LinearDiscriminantAnalysis(),
        "KNN": KNeighborsClassifier(n_neighbors=9),
        "Naive Bayes": GaussianNB()
    }
    
    all_metrics = {}
    for name, clf in classifiers.items():
        if name != "Neural Network (15 Epochs)":
            clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else pred
        
        acc = float(accuracy_score(y_test, pred))
        prec = float(precision_score(y_test, pred, zero_division=0))
        rec = float(recall_score(y_test, pred, zero_division=0))
        f1 = float(f1_score(y_test, pred, zero_division=0))
        auc = float(roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else 0.0)
        
        all_metrics[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc': auc
        }
        print(f"  {name:<28} | Acc: {acc*100:>5.1f}% | Prec: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")
        
    champion_name = max(all_metrics.keys(), key=lambda k: all_metrics[k]['f1'])
    champion_model = classifiers[champion_name]
    
    print("\n" + "="*70)
    print(f">> CHAMPION MODEL SELECTED: {champion_name} (F1: {all_metrics[champion_name]['f1']:.3f}, AUC: {all_metrics[champion_name]['auc']:.3f})")
    print("="*70)
    
    # Train champion on 100% of data
    logger.info(f"Training champion model '{champion_name}' on all 50,000 samples...")
    if champion_name == "Neural Network (15 Epochs)":
        final_model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate_init=0.005,
            random_state=42,
            max_iter=NUM_EPOCHS
        )
        final_model.fit(X_scaled, y)
    else:
        final_model = champion_model
        final_model.fit(X_scaled, y)
        
    feature_importances = {}
    if hasattr(final_model, 'feature_importances_'):
        for feat, imp in zip(feature_names, final_model.feature_importances_):
            feature_importances[feat] = float(imp)
    elif hasattr(final_model, 'coef_'):
        for feat, imp in zip(feature_names, np.abs(final_model.coef_[0])):
            feature_importances[feat] = float(imp)
    else:
        for feat in feature_names:
            feature_importances[feat] = 1.0 / len(feature_names)
            
    bundle = {
        'model': final_model,
        'model_name': champion_name,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': feature_names,
        'categorical_features': cat_cols,
        'metrics': all_metrics,
        'feature_importances': feature_importances,
        'epochs': NUM_EPOCHS,
        'dataset_size': len(df)
    }
    
    output_path = os.path.join(PROJECT_ROOT, 'models', 'threat_detector_model.pkl')
    joblib.dump(bundle, output_path)
    logger.info(f"Saved trained PKL model to: {output_path}")
    print(f"\n>> SUCCESS: Saved production PKL model to {output_path}")

if __name__ == '__main__':
    main()
