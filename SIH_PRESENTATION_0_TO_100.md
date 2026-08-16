# 🏆 Smart India Hackathon (SIH) — 0 to 100 Project Presentation Guide
## Adaptive Social Engineering Defense Framework (ASEDF)
### *"The Truecaller & Gmail Spam Filter for X (Twitter)"*

---

## 📑 Quick Navigation
1. [Elevator Pitch (30 Seconds)](#1-elevator-pitch-30-seconds)
2. [Problem Statement & The "Why Now"](#2-problem-statement--the-why-now)
3. [Product Analogy & Positioning](#3-product-analogy--positioning)
4. [Complete End-to-End Technical Architecture (0 to 100)](#4-complete-end-to-end-technical-architecture-0-to-100)
5. [The 5 Core Analytical Modules](#5-the-5-core-analytical-modules)
6. [Machine Learning & Anti-Overfitting Benchmark (13 Models)](#6-machine-learning--anti-overfitting-benchmark-13-models)
7. [The 4-Layer Explainable AI (XAI) Subsystem](#7-the-4-layer-explainable-ai-xai-subsystem)
8. [Live Demonstration Script (Step-by-Step for Judges)](#8-live-demonstration-script-step-by-step-for-judges)
9. [Key Novelties & Competitive Differentiators](#9-key-novelties--competitive-differentiators)
10. [Defense & Judge Q&A Cheat Sheet (Harsh Questions Answered)](#10-defense--judge-qa-cheat-sheet-harsh-questions-answered)

---

## 1. ⚡ Elevator Pitch (30 Seconds)

> *"Respected Jury, just like **Gmail** automatically analyzes incoming emails to block spam, and **Truecaller** scans phone numbers in real time to alert you about fraudulent calls — **our platform, ASEDF, is the real-time Spam and Threat Identifier for X (Twitter)**.*
> 
> *When an account or tweet arrives, our system ingests live public data with **zero API cost**, extracts **44 multi-modal behavioral signals**, classifies semantic phishing intent using a custom **Fine-Tuned DistilBERT Transformer (97.5% Accuracy)**, and evaluates threat probability via an **AdaBoost Ensemble Classifier (89.8% Accuracy, 0.957 ROC-AUC)**.*
> 
> *Most importantly, we don't give a black-box score: our **4-Layer Explainable AI (XAI)** provides exact mathematical **SHAP attribution**, **Counterfactual remediation ('what-if' analysis)**, and **DistilBERT token attention heatmaps** in under 3 seconds."*

---

## 2. 🚨 Problem Statement & The "Why Now"

Social networks have evolved from simple chat rooms into critical national infrastructure for news, finance, and governance. However:
1. **Financial Fraud**: ₹10,000+ Crores are lost annually to automated crypto giveaway scams, impersonation bots, and phishing redirect shortlinks.
2. **Coordinated Astroturfing**: Political botnets hijack public sentiment through synchronized burst-tweeting and mass `@mention` attacks.
3. **The API Paywall Barrier**: Official X API v2 access costs $100 to $5,000/month, making cyber protection inaccessible to everyday citizens and law enforcement agencies.
4. **Black-Box AI Skepticism**: Existing security tools output a percentage (e.g. *"85% Bot"*) without evidence, making it impossible for SOC analysts or judges to trust or take legal action.

---

## 3. 💡 Product Analogy & Positioning

| Platform | What It Scans | How It Warns | Our Direct Equivalent |
|---|---|---|---|
| 📧 **Gmail** | Headers, Sender Domain, Body Text, Attachments | ⚠️ *"This message seems dangerous"* | **ASEDF Tweet & Bio Inspector** |
| 📞 **Truecaller** | Caller ID, Call Duration, Telemarketer Reports | 🚨 *"Spam / Fraud Caller"* | **ASEDF Account Longevity & Graph Score** |
| 🛡️ **ASEDF (Our System)** | **44 Multi-Modal Signals, DistilBERT NLP, Graph Isolation, Shortlinks** | 🚨 **Threat Classification + 4-Layer XAI Forensic Dossier** | **The Unified Social Threat Shield** |

---

## 4. 🏗️ Complete End-to-End Technical Architecture (0 to 100)

```
 [USER ENTERS USERNAME OR POST URL] (e.g., @cryptopump_bot22)
                 │
                 ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ LAYER 1: ZERO-KEY DATA INGESTION ENGINE (src/utils/data_processor.py)       │
 │ • Activates ephemeral x-guest-token via https://api.twitter.com/1.1/guest   │
 │ • Dispatches authenticated GraphQL queries: UserByScreenName + UserTweets   │
 │ • Extracts: Metadata, follower counts, bio, avatar URL, 100 recent posts   │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ LAYER 2: 44-DIMENSIONAL FEATURE EXTRACTION (src/features/feature_extractor) │
 │ ├─ Identity & Longevity: Account age, followers-to-following ratio, p/day  │
 │ ├─ Temporal Regularity: Timestamp variance (CV), duplicate text ratio       │
 │ ├─ Social Graph Topology: Network isolation score, reciprocal follow index  │
 │ └─ Visual Forensics: AI-generated profile pic flags, default avatar check   │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ LAYER 3: FINE-TUNED DISTILBERT NLP CLASSIFIER (src/features/nlp_classifier) │
 │ • 66M parameter Transformer model fine-tuned on social engineering corpus   │
 │ • Evaluates timeline posts across 5 threat classes:                         │
 │   [0: Legitimate | 1: Crypto Scam | 2: Phishing | 3: Spam | 4: Social Eng]  │
 │ • Test Set Performance: 97.5% Accuracy | 0.975 F1-Score                     │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ LAYER 4: 13-MODEL MACHINE LEARNING ENGINE (src/models/train_model.py)       │
 │ • Trained on 2,102 strictly balanced 50/50 real-world benchmark profiles    │
 │ • Champion Model: AdaBoost Ensemble Classifier (89.8% Acc, 0.957 ROC-AUC)   │
 │ • Anti-Shortcut Regularization (neutralizes verified badge cheat codes)     │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ LAYER 5: 4-LAYER EXPLAINABLE AI (XAI) SUBSYSTEM (src/xai/)                  │
 │ ├─ 1. SHAP Decomposition: Exact Shapley values (Baseline 50% ──► 89.8%)    │
 │ ├─ 2. Counterfactual Engine: "What changes would make this account SAFE?"  │
 │ ├─ 3. DistilBERT Token Saliency: Attention weight heatmaps on tweet words   │
 │ └─ 4. LIME Cross-Verification: Independent local linear surrogate consensus │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
 [DYNAMIC GLASSMORPHISM FORENSIC DASHBOARD & REST API OUTPUT] (/results)
```

---

## 5. 🔬 The 5 Core Analytical Modules

### Module 1: Zero-Key GraphQL Scraper
* **The Breakthrough**: Doesn't require a $100/mo X API key. Uses the official web client guest-token handshake protocol to fetch live public profiles and recent timeline posts safely in real time.

### Module 2: 44-Feature Multi-Modal Extraction Engine
* Translates raw JSON into 44 normalized numeric and semantic signals across 5 domains:
  - **Identity**: `account_age_days`, `followers_count`, `followers_to_following_ratio`.
  - **Linguistic**: `bio_length`, `suspicious_content_score`, `spam_pattern_matches`.
  - **Behavioral**: `posts_per_day`, `posting_regularity` (Coefficient of Variation $\sigma/\mu$), `duplicate_post_ratio`, `link_post_ratio`, `mention_ratio`.
  - **Network Graph**: `network_isolation_score`, `mutual_connection_ratio`.
  - **Visual Forensics**: `is_default_image`, `is_ai_generated`.

### Module 3: Fine-Tuned DistilBERT Social Engineering Engine
* Trained on multi-class cyber threat corpora.
* Computes exact softmax probabilities across 5 threat classes:
  - `Crypto Giveaway Scam` (92.3% F1)
  - `Credential Phishing` (94.1% F1)
  - `Mass Mention Spam` (100% F1)
  - `Social Engineering / Advance-Fee Fraud` (100% F1)

### Module 4: Anti-Shortcut Machine Learning Classifier
* Evaluated against 13 distinct algorithms on real academic datasets (**Botwiki**, **Cresci-RTbust**, **Verified-2019**, **TwiBot-20**).
* **Champion Model**: **AdaBoost Ensemble (89.8% Test Accuracy, 0.957 ROC-AUC, 0.952 Recall)**.

### Module 5: 4-Layer Explainable AI Subsystem
* Explains the mathematical reasoning behind every classification, turning opaque AI into actionable evidence.

---

## 6. 📊 Machine Learning & Anti-Overfitting Benchmark (13 Models)

Trained on **2,102 strictly balanced real-world benchmark profiles (1,051 Threats / 1,051 Humans)** with verified-badge shortcut neutralization. Evaluated on a 20% holdout test set (421 real profiles):

| Rank | Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Why Selected / Rejected |
|---|---|---|---|---|---|---|---|
| 🥇 | **AdaBoost Ensemble (Champion)** | **89.79%** | **0.8584** | **0.9524** | **0.9029** | **0.9569** | **Production Champion**: Adaptive sequential boosting on boundary edge cases; highest recall (95.2%). |
| 🥈 | **Decision Tree** | **89.55%** | 0.8705 | 0.9286 | 0.8986 | 0.9506 | High interpretability; pruned rules capture threshold splits. |
| 🥉 | **Gradient Boosting** | **89.07%** | 0.8727 | 0.9143 | 0.8930 | **0.9596** | Highest ROC-AUC (0.9596); sequential residual gradient correction. |
| 4 | **Random Forest** | **88.84%** | 0.8498 | 0.9429 | 0.8939 | 0.9568 | Bagging ensemble averaging across 150 randomized trees. |
| 5 | **Histogram Gradient Boosting** | **88.12%** | 0.8704 | 0.8952 | 0.8826 | 0.9563 | Histogram binning with L2 regularization penalty ($L2=2.0$). |
| 6 | **K-Nearest Neighbors (KNN)** | **81.71%** | 0.8276 | 0.8000 | 0.8136 | 0.8851 | Distance metric clustering in continuous feature space ($k=9$). |
| 7 | **Neural Network (MLP)** | **75.53%** | 0.7772 | 0.7143 | 0.7444 | 0.8345 | Multi-Layer Perceptron (64, 32) deep feature representation. |
| 8 | **Logistic Regression** | **74.58%** | 0.6886 | 0.8952 | 0.7785 | 0.8892 | Regularized linear baseline with L2 penalty ($C=0.5$). |
| 9 | **Support Vector Machine (RBF)** | **72.92%** | 0.6678 | 0.9095 | 0.7702 | 0.8802 | Soft-margin hyperplane with radial basis kernel ($C=0.8$). |
| 10 | **Extra Trees** | **65.32%** | 0.8333 | 0.3810 | 0.5229 | 0.7859 | Extremely randomized trees; conservative recall. |
| 11 | **Linear Discriminant (LDA)** | **63.66%** | 0.7767 | 0.3810 | 0.5112 | 0.8110 | Linear continuous projection; struggles with non-linear threats. |
| 12 | **Naive Bayes (Gaussian)** | **60.57%** | 0.5591 | 0.9905 | 0.7148 | 0.9058 | Probabilistic model under feature independence assumption. |

---

## 7. 🧠 The 4-Layer Explainable AI (XAI) Subsystem

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ 1️⃣  SHAP WATERFALL (Cooperative Game Theory)                               │
 │ • Decomposes: Baseline Prior (50%) ──► Marginal Contributions ──► Score     │
 │ • Red Bars: Threat Amplifiers (e.g. Followers Count +19.6%)                 │
 │ • Green Bars: Safety Anchors (e.g. Organic Ratio -2.6%)                     │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ 2️⃣  COUNTERFACTUAL REMEDIATION ("What-If" Analysis)                        │
 │ • Simulates greedy feature perturbations to answer:                         │
 │   "What minimum actions would flip this account to SAFE (<40%)?"            │
 │ • e.g., Step 1: Reduce external redirect links below 20% (Actionable)       │
 │         Step 2: Age account past 90 days (Requires Time)                    │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ 3️⃣  DISTILBERT TOKEN SALIENCY HEATMAP                                       │
 │ • Extracts real-time Transformer attention weights from the last layer.     │
 │ • Renders colored background heatmaps on suspicious tweet words:            │
 │   [FREE] [AIRDROP] [claim] [your] [USDT] [t.me/scampump]                    │
 │    CRIT    CRIT     HIGH    NEUT    MED        CRIT                         │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ 4️⃣  LIME CROSS-VERIFICATION CONSENSUS                                      │
 │ • Fits independent local linear surrogates in a perturbed neighbourhood.    │
 │ • Computes XAI Consensus Score (Agreement % between SHAP and LIME).         │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. 🎬 Live Demonstration Script (Step-by-Step for Judges)

### Step 1: Open the Dashboard
Navigate to `http://127.0.0.1:5000`. Show the clean, dark-mode Glassmorphism interface.

### Step 2: Test a Known Clean Account (`@elonmusk` or `@coolcoder56`)
1. Enter the username and click **Analyze Profile**.
2. Point out:
   - **Low Risk Verdict (Green)** with < 35% probability.
   - **Executive Summary**: Shows positive authenticity signals (organic longevity, balanced follow ratio).
   - **SHAP Waterfall**: Green bars pulling the score down below the baseline.

### Step 3: Test a Threat Account or Paste a Scam Tweet
1. Scan a crypto-bot or suspicious profile with high link ratios.
2. Point out:
   - **High Risk Alert (Red Badge)**: Classified as *Crypto Scam / Phishing*.
   - **Detected Indicators**: Shows high external redirect ratio, copy-pasted duplicate text, and low account age.
   - **SHAP Waterfall**: Red bars showing exact mathematical blame (+% contributions).
   - **Counterfactual Panel**: Show the judge the exact remediation steps that would flip the score.
   - **Token Heatmap**: Show the highlighted scam tokens (`airdrop`, `guaranteed`, `claim now`, `t.me/`).

### Step 4: Show the Data Explorer & Model Leaderboard
1. Click **Data Explorer** (`/data-explorer`): Show 5,000+ benchmark profiles with live Chart.js distribution charts and CSV export.
2. Click **Model Info** (`/model-info`): Show the 13-algorithm comparative benchmark table.

---

## 9. 🌟 Key Novelties & Competitive Differentiators

| Feature | Typical Hackathon Project | Our ASEDF Platform |
|---|---|---|
| **Data Ingestion** | Uses dummy hardcoded JSON or broken Tweepy | **Zero-Key Reverse-Engineered GraphQL Live Ingestion** |
| **NLP Intelligence** | Simple regex keyword counting | **Fine-Tuned 66M-Parameter DistilBERT Transformer (97.5% Acc)** |
| **Model Rigor** | 1 single model trained on synthetic data | **13-Model Benchmark on 2,102 Real Academic Profiles** |
| **Dataset Bias** | Overfits to verified badges (shortcut cheating) | **Strictly Balanced 50/50 Anti-Shortcut Regularization** |
| **Explainability** | Opaque black-box score (no explanation) | **4-Layer XAI: SHAP + Counterfactuals + Token Saliency + LIME** |
| **API & Enterprise** | UI only | **Full REST API (`/api/analyze`) ready for SIEM & SOC integration** |

---

## 10. 🛡️ Defense & Judge Q&A Cheat Sheet (Harsh Questions Answered)

#### Q1: "How do you bypass Twitter's $100/mo API paywall?"
> **Answer**: *"We reverse-engineered the official X web client guest activation protocol (`api.twitter.com/1.1/guest/activate.json`). Our system requests an ephemeral `x-guest-token` and authenticates directly against official GraphQL endpoints (`UserByScreenName` and `UserTweets`). It requires zero API keys, costs ₹0, and runs in real time."*

#### Q2: "How do you know your model isn't just memorizing or overfitting?"
> **Answer**: *"We specifically tackled the **'Verified Badge Shortcut Bias'**. In academic datasets like Verified-2019, all humans have verified badges and all bots don't. A lazy model simply splits on `is_verified`. We neutralized the verified flag during training, balanced the dataset 50/50 with 1,051 threats and 1,051 humans across 4 academic archives (Botwiki, Cresci, Verified, TwiBot), and applied L2 regularization. Our champion AdaBoost achieves an authentic, generalized 89.8% test accuracy with 0.957 ROC-AUC."*

#### Q3: "Why did you use DistilBERT instead of regular BERT or GPT-4?"
> **Answer**: *"DistilBERT retains 97% of BERT's language understanding while being 60% faster and 40% smaller (66M parameters). It executes inference in under 30 milliseconds on standard CPU without requiring expensive GPU infrastructure, achieving **97.5% Accuracy** on social engineering classification. GPT-4 API calls introduce 2–3 second network latency and recurring API costs."*

#### Q4: "What makes your Explainable AI (XAI) better than simple feature importances?"
> **Answer**: *"Feature importance is global — it only tells you what the model looks at across the whole dataset. Our system provides **local, per-prediction XAI across 4 layers**: (1) **SHAP Shapley values** decomposing the exact percentage shift from baseline for this specific profile, (2) **Counterfactuals** showing what minimal changes would make it safe, (3) **DistilBERT token attention heatmaps** highlighting the exact words triggering the alert, and (4) **LIME cross-verification** confirming the surrogate consensus."*

#### Q5: "How does this scale to an enterprise SOC or Cyber Cell?"
> **Answer**: *"Our system includes a headless REST API (`/api/analyze`) and batch processing portal (`/batch`). A cyber crime unit or SIEM platform can ingest thousands of suspicious accounts via CSV/JSON, auto-flag high-risk bot clusters, and export forensic PDF/JSON dossiers for evidence filing."*
