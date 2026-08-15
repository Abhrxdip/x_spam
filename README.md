# x_spam: Adaptive Social Engineering Defense Framework (ASEDF)

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.12-brightgreen.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-black.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange.svg)

An AI-powered multi-modal threat detection platform designed for **Smart India Hackathon (SIH)** to identify fake social media profiles, spam campaigns, crypto phishing, and botnets across X (Twitter), Instagram, and Facebook.

---

## 🌟 Key Features

1. **Multi-Modal Feature Fusion Engine**:
   - Analyzes 42 distinct features combining **Account Metrics**, **NLP Content Sentiment**, **Timeline Postings**, **Network Isolation Ratios**, and **Profile Picture Visual Cues**.

2. **Mention & Posting Spam Analysis**:
   - Flags `@username` tagging spam attacks (unsolicited mass tagging).
   - Detects external phishing link campaigns (`bit.ly`, `t.me`, `wa.me`).
   - Identifies copy-paste repetitive timeline postings and hashtag stuffing.

3. **13 ML Classifier Evaluation Leaderboard**:
   - Trains and compares 13 machine learning classifiers: *Gradient Boosting*, *Random Forest*, *HistGradientBoosting*, *AdaBoost*, *Extra Trees*, *Neural Network (MLP)*, *Support Vector Machine (SVC)*, *Decision Tree*, *Logistic Regression*, *Linear Discriminant*, *KNN*, *Gaussian Naive Bayes*, and *Quadratic Discriminant*.
   - **Active Champion**: Gradient Boosting with **98.9% Accuracy** and **0.997 ROC-AUC**.

4. **Live X/Twitter Profile Ingestion**:
   - Connects to public X/Twitter syndication feeds to pull live display names, follower counts, tweet timelines, and verification status for any profile link (e.g. `https://x.com/username`).

5. **Interactive Data Explorer Dashboard**:
   - Searchable, paginated 5,000+ benchmark dataset viewer with dynamic Chart.js visualizations, row feature inspection modal, and CSV export.

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10+
- `pip` package manager

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Abhrxdip/x_spam.git
cd x_spam

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Model Training Pipeline (Optional)

```bash
python scripts/train.py
```

### 3. Launch Web Application

```bash
python app.py
```

Open your browser at `http://127.0.0.1:5000`.

---

## 🖥️ Web App Endpoints

- `/` — Homepage & Single Profile Risk Analyzer
- `/data-explorer` — Interactive Dataset & Feature Explorer Dashboard
- `/model-info` — 13 ML Classifier Evaluation Matrix & Leaderboard
- `/train-model` — Live Retraining Interface
- `/batch` — Batch File Upload Analysis
- `/api/analyze` — REST API endpoint for real-time risk scoring

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
