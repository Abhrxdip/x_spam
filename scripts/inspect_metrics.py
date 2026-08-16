import joblib
import json

bundle = joblib.load('models/threat_detector_model.pkl')

print("Model Name:", bundle.get('model_name'))
print("Dataset Size:", bundle.get('dataset_size'))
print("Epochs:", bundle.get('epochs'))
print("Total Features:", len(bundle.get('feature_names', [])))
print("\n--- EXACT METRICS OF ALL TRAINED MODELS ---")
metrics = bundle.get('metrics', {})
for k, v in metrics.items():
    if isinstance(v, dict):
        acc = v.get('accuracy', 0)
        f1 = v.get('f1_score', 0)
        auc = v.get('roc_auc', 0)
        prec = v.get('precision', 0)
        rec = v.get('recall', 0)
        print(f"{k:25} | Accuracy: {acc*100:6.2f}% | F1: {f1:0.4f} | ROC-AUC: {auc:0.4f} | Precision: {prec:0.4f} | Recall: {rec:0.4f}")
    else:
        print(f"{k}: {v}")
