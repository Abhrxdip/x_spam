"""
Train Threat Detector with 15 Epochs on the 52,066 sample dataset
and export the production-ready pkl model for deployment.
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

def main():
    data_path = os.path.join(PROJECT_ROOT, 'data', 'training_data_merged.csv')
    if not os.path.exists(data_path):
        logger.error(f"Merged dataset not found at {data_path}. Running retrain ingestion...")
        from scripts.retrain_with_new_data import load_bot_detection_data
        existing_path = os.path.join(PROJECT_ROOT, 'data', 'training_data.csv')
        new_path = os.path.join(PROJECT_ROOT, 'data', 'bot_detection_data.csv')
        df_existing = pd.read_csv(existing_path)
        df_new = load_bot_detection_data(new_path)
        common_cols = [c for c in df_existing.columns if c in df_new.columns]
        df_merged = pd.concat([df_existing, df_new[common_cols]], ignore_index=True)
        n_min = df_merged['is_threat'].value_counts().min()
        df_balanced = pd.concat([
            df_merged[df_merged['is_threat']==0].sample(n_min, random_state=42),
            df_merged[df_merged['is_threat']==1].sample(n_min, random_state=42)
        ]).sample(frac=1, random_state=42).reset_index(drop=True)
        df_balanced.to_csv(data_path, index=False)
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset with {len(df):,} samples.")
    
    # Target and features
    y = df['is_threat'].values
    X = df.drop(columns=['is_threat'])
    
    # Categorical encoding
    label_encoders = {}
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        
    feature_names = X.columns.tolist()
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\n" + "="*70)
    print(f">> TRAINING NEURAL NETWORK (MLP) WITH {NUM_EPOCHS} EPOCHS")
    print("="*70)
    
    # Epoch-by-epoch training simulation with partial_fit
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
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            mlp.partial_fit(X_batch, y_batch, classes=classes)
            
        train_pred = mlp.predict(X_train[:2000])
        val_pred = mlp.predict(X_test[:2000])
        train_acc = accuracy_score(y_train[:2000], train_pred)
        val_acc = accuracy_score(y_test[:2000], val_pred)
        print(f"  Epoch [{epoch:02d}/{NUM_EPOCHS}] — Loss: {mlp.loss_:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
        
    mlp_test_pred = mlp.predict(X_test)
    mlp_test_proba = mlp.predict_proba(X_test)[:, 1]
    
    mlp_metrics = {
        'accuracy': float(accuracy_score(y_test, mlp_test_pred)),
        'precision': float(precision_score(y_test, mlp_test_pred, zero_division=0)),
        'recall': float(recall_score(y_test, mlp_test_pred, zero_division=0)),
        'f1': float(f1_score(y_test, mlp_test_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_test, mlp_test_proba))
    }
    
    print("\n" + "="*70)
    print("📊 BENCHMARKING CLASSIFIER FAMILIES (15 Iteration Stage Boosting)")
    print("="*70)
    
    classifiers = {
        "Neural Network (15 Epochs)": mlp,
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=NUM_EPOCHS, random_state=42),
        "AdaBoost (15 Stages)": AdaBoostClassifier(n_estimators=NUM_EPOCHS, random_state=42),
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
        
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else 0.0
        
        all_metrics[name] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'auc': float(auc)
        }
        print(f"  {name:<28} | Acc: {acc*100:>5.1f}% | Prec: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")
    
    # Pick Champion Model (HistGradientBoosting / AdaBoost / Gradient Boosting)
    champion_name = max(all_metrics.keys(), key=lambda k: all_metrics[k]['f1'])
    champion_model = classifiers[champion_name]
    print("\n" + "="*70)
    print(f"🏆 CHAMPION MODEL SELECTED: {champion_name} (F1: {all_metrics[champion_name]['f1']:.3f})")
    print("="*70)
    
    # Retrain champion on 100% of data
    logger.info(f"Training champion {champion_name} on 100% of data ({len(X_scaled):,} samples)...")
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
        
    # Feature importances
    feature_importances = {}
    if hasattr(final_model, 'feature_importances_'):
        for feat, imp in zip(feature_names, final_model.feature_importances_):
            feature_importances[feat] = float(imp)
    elif hasattr(final_model, 'coef_'):
        for feat, imp in zip(feature_names, np.abs(final_model.coef_[0])):
            feature_importances[feat] = float(imp)
    else:
        # Default equal weights if not directly accessible
        for feat in feature_names:
            feature_importances[feat] = 1.0 / len(feature_names)
            
    # Save model bundle
    bundle = {
        'model': final_model,
        'model_name': champion_name,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': feature_names,
        'categorical_features': categorical_cols,
        'metrics': all_metrics,
        'feature_importances': feature_importances,
        'epochs': NUM_EPOCHS
    }
    
    output_path = os.path.join(PROJECT_ROOT, 'models', 'threat_detector_model.pkl')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(bundle, output_path)
    logger.info(f"💾 Model PKL successfully created & saved to: {output_path}")
    print(f"\n✅ Created PKL bundle at: {output_path}")
    print(f"📦 File size: {os.path.getsize(output_path) / 1024:.1f} KB")

if __name__ == '__main__':
    main()
