<h1 align="center">🛡️ x_spam — Adaptive Social Engineering Defense Framework (ASEDF)</h1>

<p align="center">
  <b>Multi-Modal AI Threat Intelligence & Deep Natural Language Defense for Social Networks</b><br/>
  <i>Engineered for Smart India Hackathon (SIH)</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.13-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Transformers-5.15-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/DistilBERT-97.5%25_Acc-4B8BBE?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/ML_Champion-98.9%25_Acc-008080?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

---

## 📑 Table of Contents
1. [Executive Summary & Problem Statement](#-executive-summary--problem-statement)
2. [What The Framework Does](#-what-the-framework-does)
3. [Full System Architecture & Processing Pipeline](#-full-system-architecture--processing-pipeline)
4. [Deep Dive: System Modules & Engineering Mechanics](#-deep-dive-system-modules--engineering-mechanics)
   - [Module 1: Real-Time Data Ingestion & Reverse-Engineered GraphQL API](#module-1-real-time-data-ingestion--reverse-engineered-graphql-api)
   - [Module 2: 44-Feature Multi-Modal Extraction Engine](#module-2-44-feature-multi-modal-extraction-engine)
   - [Module 3: Fine-Tuned DistilBERT Social Engineering NLP Engine (97.5% Accuracy)](#module-3-fine-tuned-distilbert-social-engineering-nlp-engine-975-accuracy)
   - [Module 4: 13-Model Machine Learning Ensemble & Benchmark Leaderboard](#module-4-13-model-machine-learning-ensemble--benchmark-leaderboard)
   - [Module 5: Explainable AI (XAI) & Security Indicator Engine](#module-5-explainable-ai-xai--security-indicator-engine)
   - [Module 6: Web Presentation, Data Explorer & REST API](#module-6-web-presentation-data-explorer--rest-api)
5. [Mathematical & Algorithmic Foundations](#-mathematical--algorithmic-foundations)
6. [Complete Implemented Features Matrix](#-complete-implemented-features-matrix)
7. [Directory & File Structure](#-directory--file-structure)
8. [Installation & Execution Guide](#-installation--execution-guide)
9. [REST API Documentation](#-rest-api-documentation)
10. [Future SIH Development Roadmap](#-future-sih-development-roadmap)
11. [License & Acknowledgements](#-license--acknowledgements)

---

## 🎯 Executive Summary & Problem Statement

Modern social networks (such as X/Twitter, Instagram, and Facebook) are heavily weaponized by sophisticated threat actors deploying automated botnets, AI-driven crypto phishing, mass mention spam, fake identity impersonation, and romance/investment social engineering attacks.

### Why Traditional Moderation Fails:
1. **Scale**: Millions of synthetic accounts are registered daily using disposable virtual numbers and proxies.
2. **Evasion**: Scammers bypass simple word filters using homoglyphs, URL shorteners (`bit.ly`, `t.me`), zero-width spaces, and context-switching.
3. **Coordinated Inauthentic Behavior**: Isolated rules miss network-level and temporal anomalies (e.g. synchronized burst posting or follower-to-following imbalances).
4. **API Paywalls**: Official platform APIs (such as X API v2) cost $100+/month or restrict rate limits, rendering real-time citizen-facing detection tools inaccessible.

### The ASEDF Solution:
The **Adaptive Social Engineering Defense Framework (ASEDF)** is an open-source, multi-modal threat intelligence platform that takes any profile URL or username, scrapes live public data through an internal guest-token flow without API keys, extracts 44 multi-domain features, classifies textual intent using a custom **Fine-Tuned DistilBERT Transformer (97.5% Accuracy)**, and evaluates threat vectors using an **AdaBoost Ensemble Classifier (89.8% Accuracy, 0.9569 ROC-AUC)** trained with anti-shortcut regularization on real-world academic benchmarks (`Botwiki`, `Cresci-RTbust`, `Verified-2019`, `TwiBot-20`) — returning explainable security intelligence in under 3 seconds.

---

## 🔬 How ASEDF Analyzes Everything: Step-by-Step Technical Anatomy

ASEDF implements an end-to-end, multi-layered threat evaluation pipeline that dissects social media accounts across **5 distinct analytical phases**:

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ PHASE 1: Zero-Key Live Ingestion & Benchmark Batch Parser                   │
 │ • Reverse-Engineered Twitter GraphQL (Guest Token Activation)               │
 │ • TwiBot-20 / TwiBot-22 / Cresci Batch Parser (Multi-Encoding Fallback)     │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ PHASE 2: 44-Dimensional Multi-Modal Feature Extraction                      │
 │ ├─ Tier 1: Profile Identity & Longevity Metrics (Age, Follower Ratios)      │
 │ ├─ Tier 2: Linguistic Entropy & Keyword Density (TTR, Scam Regex Matches)   │
 │ ├─ Tier 3: Temporal Regularity & Burst Frequency (CV, Repetition, Links)   │
 │ ├─ Tier 4: Multi-Relational Graph & Network Topology (Isolation Index)     │
 │ └─ Tier 5: Profile Image & Visual Forensics (Default Avatar, Synthetic GAN) │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ PHASE 3: Fine-Tuned DistilBERT NLP Intent Classification                    │
 │ • Tokenizes timeline tweets and extracts contextual threat probabilities:   │
 │   [Crypto Scam (92.3% F1) | Phishing (94.1% F1) | Mention Spam (100% F1)]   │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ PHASE 4: 13-Model Machine Learning Ensemble & Classification                │
 │ • Standardizes 44-D vector -> Evaluates through Champion AdaBoost Ensemble  │
 │ • Generates Calibrated Threat Probability Score (0.0% - 100.0%)             │
 │ • Assigns Threat Category: Legitimate, Spam, Bot, Phishing, Fake Profile    │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ PHASE 5: Explainable AI (XAI) Forensic Reasoning & Remediation              │
 │ • Generates human-readable decision explanations ("Why High / Low Risk")    │
 │ • Displays 44-Feature Signal Matrix & Actionable SOC Containment Steps      │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

### 1️⃣ Phase 1: Real-Time Ingestion & Multi-Benchmark Parsing
*File: `src/utils/data_processor.py`*

1. **Zero-Key Guest Token Activation**:
   - Queries `https://api.twitter.com/1.1/guest/activate.json` to generate an ephemeral `x-guest-token`, enabling real-time live scraping without paywalled $100+/mo X API v2 subscriptions.
2. **GraphQL User & Timeline Ingestion**:
   - Dispatches authenticated GraphQL requests (`UserByScreenName` and `UserTweets`) extracting: `rest_id`, `name`, `screen_name`, `description`, `followers_count`, `friends_count`, `statuses_count`, `created_at`, `is_blue_verified`, avatar URLs, and up to 100 recent tweets with engagement metrics.
3. **Multi-Benchmark Batch Parser**:
   - Supports **TwiBot-20**, **TwiBot-22** (`public_metrics`, nested `user`, `recent_tweets`), and raw CSVs with automated multi-encoding decoding (`utf-8`, `utf-8-sig`, `latin1`, `cp1252`), completely eliminating unicode decode crashes.

---

### 2️⃣ Phase 2: 44-Feature Multi-Modal Extraction Engine
*File: `src/features/feature_extractor.py`*

The system extracts 44 granular signals grouped into 5 specialized domains:

| Category | Extracted Signals | Security Threat Significance |
|---|---|---|
| **Account Identity & Longevity** | `account_age_days`, `followers_count`, `following_count`, `posts_count`, `followers_to_following_ratio`, `posts_per_day`, `Twitter.Verified`, `Account.Type`, `Country`, `Gender` | Newly registered accounts (<30 days) with high followings (>2,000) and 0 followers represent classic automated bot creation scripts. |
| **Linguistic & Content Semantics** | `bio_length`, `has_external_url`, `sentiment_score`, `content_diversity`, `suspicious_content_score`, `spam_pattern_matches`, `word_sex`, `word_good`, `word_woman`, `word_new`, `word_like`, `name_2_w` | Measures bio complexity, lexical richness (Type-Token Ratio), keyword density (`airdrop`, `whitelist`, `free bonus`), and bait terms. |
| **Fine-Tuned DistilBERT NLP** | `nlp_phishing_score`, `nlp_spam_confidence`, `nlp_threat_class`, `nlp_high_risk_count`, `deberta_phishing_score`, `deberta_spam_confidence` | Neural transformer logits representing direct semantic threat intent (Crypto Scam, Phishing, Mention Spam, Social Engineering). |
| **Temporal & Behavioral Regularity** | `mention_count`, `mention_ratio`, `avg_mentions_per_post`, `hashtag_stuffing_ratio`, `link_post_ratio`, `duplicate_post_ratio`, `engagement_rate`, `posting_regularity`, `time_zone_consistency`, `activity_score`, `Thread.Entry.Type` | Captures mass `@tagging` attacks, duplicate copy-paste flood campaigns, high external link ratios (`bit.ly`, `t.me`), and unnatural posting interval regularity. |
| **Network & Graph Topology** | `network_isolation_score`, `mutual_connection_ratio`, `clustering_coefficient`, `reciprocity`, `network_score`, `profile_pic_score`, `is_default_image`, `is_stock_photo`, `is_ai_generated`, `links_twitter`, `links_youtube`, `links_facebook`, `links_instagram`, `links_other` | Identifies accounts isolated from the general social graph, default placeholder avatars, and external malicious redirection channels. |

---

### 3️⃣ Phase 3: Fine-Tuned DistilBERT Social Engineering Classifier
*Files: `scripts/finetune_nlp.py`, `src/features/nlp_classifier.py`*

Unlike primitive regex filters that fail when attackers rephrase sentences or use homoglyphs, ASEDF features a **fine-tuned Transformer Language Model (`distilbert-base-uncased`, 66M parameters)** trained on a multi-class social engineering threat corpus.

#### 5-Class Threat Taxonomy & Evaluation Metrics:

```
======================================================================
  DistilBERT Multi-Class Threat Classification Report (Test Set)
======================================================================
     Threat Category        Precision    Recall    F1-Score   Support
──────────────────────────────────────────────────────────────────────
 0.  Legitimate Organic       1.0000     1.0000     1.0000       17
 1.  Crypto Giveaway Scam     1.0000     0.8571     0.9231        7
 2.  Phishing / Credential    0.8889     1.0000     0.9412        8
 3.  Mass Mention Spam        1.0000     1.0000     1.0000        3
 4.  Social Engineering       1.0000     1.0000     1.0000        5
──────────────────────────────────────────────────────────────────────
     OVERALL ACCURACY                               0.9750       40
     Macro Average            0.9778     0.9714     0.9729       40
     Weighted Average         0.9778     0.9750     0.9748       40
======================================================================
```

---

### 4️⃣ Phase 4: Machine Learning Model Benchmark Leaderboard (13 Models Evaluated)
*Files: `scripts/build_real_benchmark_dataset.py`, `src/models/train_model.py`*

Trained and evaluated on **2,102 strictly balanced real-world profiles** (1,051 Threats / 1,051 Humans) from academic benchmarks (**Botwiki-2019**, **Cresci-RTbust-2019**, **Verified-2019**, and **TwiBot-20**) with shortcut neutralization (verified badge bias masked). All metrics evaluated on a 20% holdout test set (421 real accounts):

| Rank | Model Architecture | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | Architectural Evaluation & Analysis |
|---|---|---|---|---|---|---|---|
| 🥇 | **AdaBoost Ensemble (Champion)** | **89.79%** | **0.8584** | **0.9524** | **0.9029** | **0.9569** | **Optimal Production Champion**: Adaptive sequential boosting on boundary profiles; 0.9524 recall on real botnets with 0.9569 ROC-AUC. |
| 🥈 | **Decision Tree Classifier** | **89.55%** | 0.8705 | 0.9286 | 0.8986 | 0.9506 | Regularized decision tree capturing key threshold splits without memorization. |
| 🥉 | **Gradient Boosting Classifier** | **89.07%** | 0.8727 | 0.9143 | 0.8930 | 0.9596 | Highest ROC-AUC (0.9596); sequential residual error correction. |
| 4 | **Random Forest Classifier** | **88.84%** | 0.8498 | 0.9429 | 0.8939 | 0.9568 | Bagging ensemble averaging across 150 randomized trees with depth pruning. |
| 5 | **Histogram Gradient Boosting** | **88.12%** | 0.8704 | 0.8952 | 0.8826 | 0.9563 | Histogram binning with L2 regularization penalty ($L2=2.0$). |
| 6 | **K-Nearest Neighbors (KNN)** | **81.71%** | 0.8276 | 0.8000 | 0.8136 | 0.8851 | Distance-weighted metric clustering in normalized continuous feature space ($k=9$). |
| 7 | **Neural Network (MLP)** | **75.53%** | 0.7772 | 0.7143 | 0.7444 | 0.8345 | Multi-Layer Perceptron (64, 32) deep feature representation with early stopping. |
| 8 | **Logistic Regression** | **74.58%** | 0.6886 | 0.8952 | 0.7785 | 0.8892 | Regularized linear baseline with L2 penalty ($C=0.5$). |
| 9 | **Support Vector Machine (RBF)** | **72.92%** | 0.6678 | 0.9095 | 0.7702 | 0.8802 | Soft-margin hyperplane separation with radial basis function kernel ($C=0.8$). |
| 10 | **Extra Trees Classifier** | **65.32%** | 0.8333 | 0.3810 | 0.5229 | 0.7859 | High precision (0.8333) with conservative recall on edge-case profiles. |
| 11 | **Linear Discriminant (LDA)** | **63.66%** | 0.7767 | 0.3810 | 0.5112 | 0.8110 | Linear continuous feature projection; struggles with non-linear social behaviors. |
| 12 | **Naive Bayes (Gaussian)** | **60.57%** | 0.5591 | 0.9905 | 0.7148 | 0.9058 | Probabilistic model under independence assumption; near-zero false negatives. |

---

### 5️⃣ Phase 5: Explainable AI (XAI) Forensic Engine
*Files: `src/xai/shap_explainer.py`, `src/xai/counterfactual.py`, `src/xai/nlp_saliency.py`, `src/xai/lime_explainer.py`*

Rather than acting as an opaque black box, ASEDF implements a **4-Layer Explainable AI Subsystem**:

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ 1. SHAP & Permutation Attribution Engine                                    │
 │    • Decomposes: Baseline (50%) ──► Feature Shifts ──► Final Score (89.8%)  │
 │    • Signed contributions: e.g. Link Ratio (+19.6%), Age (-2.6%)           │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ 2. Counterfactual Remediation Engine ("What-If" Analysis)                   │
 │    • Computes minimal changes required to flip threat score to SAFE (<40%)  │
 │    • e.g. "Reduce external links to <20% and age account past 90 days"      │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ 3. DistilBERT Token-Level Saliency Heatmap                                  │
 │    • Extracts real-time Transformer attention weights from the last layer   │
 │    • Generates visual color heatmaps on suspicious tweet words              │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ 4. LIME Cross-Verification & Consensus                                      │
 │    • Independent local linear surrogate cross-verifies SHAP stability       │
 │    • Computes XAI Consensus Score (HIGH / MODERATE / LOW)                   │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

### Module 6: Web Presentation, Data Explorer & REST API
*Files: `app.py`, `templates/`*

The platform provides a modern, responsive Glassmorphism dashboard:
- **Risk Assessment & XAI Dossier (`/results`)**: Dynamic threat gauge, SHAP waterfall chart, counterfactual remediation panel, DistilBERT token heatmap, and SOC analyst containment steps.
- **Data Explorer (`/data-explorer`)**: Live interactive table exploring benchmark profiles with Chart.js distribution charts, feature modals, and CSV export.
- **Model Leaderboard (`/model-info`)**: Real-time evaluation matrix comparing all 13 algorithms across Accuracy, Precision, Recall, F1, and ROC-AUC.
- **Batch Processing (`/batch`)**: Upload CSV or JSON files to scan hundreds of profiles in bulk.
- **REST API (`/api/analyze`)**: Headless JSON interface for SIEM and SOC automation.

---

## 📐 Mathematical & Algorithmic Foundations

### 1. Cooperative Game Theory (Shapley Value Attribution)
$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} (v(S \cup \{i\}) - v(S))$$
*Calculates the exact marginal contribution of feature $i$ across all possible feature subsets $S$.*

### 2. Posting Regularity (Coefficient of Variation)
Automated bots often post at rigid, mechanical time intervals, whereas human posting behavior displays natural variance.
$$\Delta t_i = t_{i+1} - t_i$$
$$\mu_{\Delta t} = \frac{1}{N-1}\sum_{i=1}^{N-1} \Delta t_i, \quad \sigma_{\Delta t} = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N-1}(\Delta t_i - \mu_{\Delta t})^2}$$
$$CV = \frac{\sigma_{\Delta t}}{\mu_{\Delta t}}, \quad \text{Regularity Score} = \max\left(0, 1 - \min(1, CV)\right)$$
*A Regularity Score close to 1 indicates robotic periodicity.*

### 3. Lexical Diversity (Type-Token Ratio)
Spam bots repeatedly post limited vocabularies or duplicate phrases.
$$\text{Diversity} = \frac{|U_{\text{words}}|}{|T_{\text{words}}|}$$
*where $U_{\text{words}}$ is the set of unique words and $T_{\text{words}}$ is the total word count.*

### 4. Network Isolation Index
Measures how disconnected an account is from organic two-way follow relationships.
$$\text{Ratio} = \frac{\text{Followers}}{\text{Following} + 1}$$
$$\text{Isolation Score} = \begin{cases} 
1.0 & \text{if } \text{Ratio} < 0.01 \\
1.0 - \text{Ratio} & \text{if } 0.01 \le \text{Ratio} < 1.0 \\
0.0 & \text{if } \text{Ratio} \ge 1.0 
\end{cases}$$

---

## 📋 Complete Implemented Features Matrix

| Feature Domain | Implemented Capability | Source File | Status |
|---|---|---|---|
| **Data Ingestion** | Live X/Twitter Guest-Token GraphQL Ingestion (No API Key Required) | `src/utils/data_processor.py` | ✅ Production Ready |
| **Data Ingestion** | Official X API v2 Bearer Token Hook | `src/utils/data_processor.py` | ✅ Production Ready |
| **Data Ingestion** | Multi-Benchmark Ingestor (`Botwiki`, `Cresci`, `TwiBot-20/22`) | `scripts/build_real_benchmark_dataset.py` | ✅ Production Ready |
| **NLP AI** | Fine-Tuned DistilBERT Social Engineering Classifier (5 Classes) | `src/features/nlp_classifier.py` | ✅ 97.5% Accuracy |
| **NLP AI** | Automated DistilBERT Fine-Tuning Pipeline | `scripts/finetune_nlp.py` | ✅ Production Ready |
| **Feature Engine** | 44-Feature Multi-Modal Extraction Engine | `src/features/feature_extractor.py` | ✅ Production Ready |
| **Feature Engine** | Mention & Tagging Spam Detection Engine | `src/features/feature_extractor.py` | ✅ Production Ready |
| **Feature Engine** | Phishing URL & Shortener Scanner (`bit.ly`, `t.me`) | `src/features/feature_extractor.py` | ✅ Production Ready |
| **Machine Learning** | 13 ML Classifier Training & Evaluation Suite | `src/models/train_model.py` | ✅ Production Ready |
| **Machine Learning** | AdaBoost Ensemble Champion Model | `src/detector.py` | ✅ 89.8% Balanced Acc |
| **Explainable AI** | SHAP / Permutation Attribution Decomposition | `src/xai/shap_explainer.py` | ✅ Production Ready |
| **Explainable AI** | Counterfactual Remediation Engine ("What-If") | `src/xai/counterfactual.py` | ✅ Production Ready |
| **Explainable AI** | DistilBERT Token Saliency Attention Heatmap | `src/xai/nlp_saliency.py` | ✅ Production Ready |
| **Explainable AI** | LIME Local Surrogate Cross-Verification | `src/xai/lime_explainer.py` | ✅ Production Ready |
| **Web Interface** | Real-Time URL / Username Threat Analyzer | `app.py`, `templates/index.html` | ✅ Production Ready |
| **Web Interface** | Forensic Threat Report with Visual Gauges & XAI Dossier | `templates/results.html` | ✅ Production Ready |
| **Web Interface** | 5,000+ Record Dataset Explorer & Chart.js Hub | `templates/data_explorer.html` | ✅ Production Ready |
| **Web Interface** | 13-Model Benchmark Comparison Leaderboard | `templates/model_info.html` | ✅ Production Ready |
| **Web Interface** | Batch CSV / JSON Threat Processing Portal | `templates/batch.html` | ✅ Production Ready |
| **Integration** | REST API Endpoint (`/api/analyze`) | `app.py` | ✅ Production Ready |


---

## 📂 Directory & File Structure

```
x_spam/
├── app.py                          # Main Flask application and REST API routes
├── requirements.txt                # Production Python dependencies
├── .env.example                    # Environment variable configuration template
├── test_nlp.py                     # Integration test suite for fine-tuned NLP model
│
├── src/
│   ├── detector.py                 # Core UnifiedThreatDetector & XAI orchestrator
│   ├── features/
│   │   ├── feature_extractor.py    # 44-feature extraction engine
│   │   ├── nlp_classifier.py       # Fine-tuned DistilBERT 5-class neural inference
│   │   └── deberta_analyzer.py     # Backward-compatible transformer shim
│   ├── models/
│   │   └── train_model.py          # 13-model training, evaluation & metrics pipeline
│   └── utils/
│       ├── data_processor.py       # Reverse-engineered live X GraphQL data scraper
│       └── visualization.py        # Report generation and Chart.js plotting utilities
│
├── models/
│   ├── threat_detector_model.pkl   # Pre-trained HistGradientBoosting model
│   └── nlp_classifier/             # Fine-tuned DistilBERT model weights & config
│       ├── config.json             # Model architecture hyperparameters
│       ├── label_map.json          # 5-class ID to label mapping
│       ├── tokenizer.json          # Fast WordPiece tokenizer vocabulary
│       └── tokenizer_config.json   # Tokenizer configuration
│
├── scripts/
│   ├── finetune_nlp.py             # DistilBERT supervised fine-tuning pipeline
│   ├── train.py                    # Tabular ML model training entry point
│   └── train_kaggle_dataset.py     # Kaggle benchmark training script
│
├── data/
│   └── training_data.csv           # 5,000+ profile multi-modal benchmark dataset
│
└── templates/
    ├── index.html                  # Main analyzer search interface
    ├── results.html                # Threat report dashboard & XAI indicators
    ├── data_explorer.html          # Interactive 5,000-record dataset explorer
    ├── model_info.html             # 13-model benchmark comparison leaderboard
    ├── batch.html                  # Bulk CSV / JSON scan interface
    └── train.html                  # Live model retraining web interface
```

---

## 🚀 Installation & Execution Guide

### Prerequisites
- Python 3.10, 3.11, or 3.12
- `git` version control
- Recommended: 4GB+ RAM

### 1. Clone & Setup Environment
```bash
# Clone the repository
git clone https://github.com/Abhrxdip/x_spam.git
cd x_spam

# Create and activate virtual environment (optional but recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run NLP Integration Verification
```bash
python test_nlp.py
```

### 3. (Optional) Re-Train Models from Scratch
```bash
# Fine-tune DistilBERT NLP classifier (takes ~10-15 mins on CPU)
python scripts/finetune_nlp.py

# Train 13 Tabular ML models & export champion model
python scripts/train.py
```

### 4. Launch the Web Platform
```bash
python app.py
```
Open **`http://127.0.0.1:5000`** in your browser.

---

## 📡 REST API Documentation

### Endpoint: `POST /api/analyze`

Analyze any social profile programmatically.

#### Request Headers:
`Content-Type: application/json`

#### Request Body:
```json
{
  "profile_url": "https://x.com/elonmusk",
  "platform": "twitter"
}
```

#### Successful Response (`200 OK`):
```json
{
  "is_threat": false,
  "threat_type": "legitimate",
  "probability": 0.0004,
  "indicators": [
    {
      "type": "unverified_high_followers",
      "severity": "low",
      "description": "High follower count without verification",
      "value": 241344977
    }
  ],
  "recommendations": [
    "No immediate action required",
    "Profile exhibits normal behavioral patterns"
  ],
  "profile_data": {
    "username": "elonmusk",
    "display_name": "Elon Musk",
    "followers_count": 241344977,
    "following_count": 1389,
    "posts_count": 107120,
    "verified": true,
    "creation_date": "Tue Jun 02 20:12:29 +0000 2009"
  }
}
```

---

## 🔮 Future SIH Development Roadmap

```
[Phase 1: Current Release]  ✅ Live GraphQL Scraping + 44 Features + Fine-Tuned DistilBERT + HistGBM
[Phase 2: Graph Intelligence] 🔄 Graph Neural Network (GNN) for Coordinated Botnet Detection
[Phase 3: Multi-Platform]     🔄 Instagram & Facebook Public Graph Ingestion Modules
[Phase 4: Visual Forensics]   🔄 CNN / FaceForensics++ DeepFake Avatar Detection
[Phase 5: SOC Tooling]        🔄 Chrome Browser Extension + Automated PDF Forensics Reports
```

1. **Graph Neural Networks (PyG / DGL)**: Build account-to-account interaction graphs to uncover hidden bot syndicates that share identical followers and amplify coordinated narratives.
2. **DeepFake Profile Image Detection**: Integrate a lightweight EfficientNet/ResNet model to detect synthetic GAN-generated human faces (StyleGAN artifacts in pupil reflections and ear symmetry).
3. **VirusTotal & SafeBrowsing Real-Time Feed**: Auto-scan every URL extracted from post timelines against global threat databases.
4. **Browser Extension**: Real-time safety badge injected directly into X/Twitter web interface.

---

## 📜 License & Acknowledgements

- **License**: Distributed under the **MIT License**. See `LICENSE` for details.
- **Dataset**: Built using academic social media threat benchmarks and augmented with real-world social engineering samples.
- **Frameworks**: Built using [PyTorch](https://pytorch.org/), [Hugging Face Transformers](https://huggingface.co/), [Scikit-Learn](https://scikit-learn.org/), and [Flask](https://flask.palletsprojects.com/).

<p align="center">
  <b>Built for Smart India Hackathon (SIH)</b><br/>
  <a href="https://github.com/Abhrxdip/x_spam">GitHub Repository</a>
</p>
