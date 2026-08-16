# 🌐 Deployment Guide — Adaptive Social Engineering Defense Framework (ASEDF)

This guide walks you through deploying your Flask + PyTorch + Scikit-Learn application to the cloud for hackathon presentations and production.

---

## 🚀 Option 1: Free Cloud Deployment on Render.com (Recommended for Hackathons)

[Render.com](https://render.com) connects directly to your GitHub repository and automatically deploys your code for free.

### Step-by-Step Instructions:
1. Go to **[https://dashboard.render.com](https://dashboard.render.com)** and sign in with GitHub.
2. Click **New +** $\rightarrow$ **Web Service**.
3. Select your repository: **`Abhrxdip/x_spam`**.
4. Configure the service settings:
   - **Name**: `x-threat-detector` (or your preferred name)
   - **Region**: Singapore / Oregon / Frankfurt
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt && pip install gunicorn shap lime`
   - **Start Command**: `gunicorn app:app --timeout 120 --workers 2 --bind 0.0.0.0:$PORT`
   - **Instance Type**: `Free`
5. Click **Create Web Service**.
6. Render will build the environment and provide you with a live HTTPS link (e.g. `https://x-threat-detector.onrender.com`).

---

## 🤗 Option 2: Free Cloud Deployment on Hugging Face Spaces (Best for PyTorch/Transformers)

Hugging Face Spaces offers a generous free tier with **16GB RAM and 2 vCPUs**, making it ideal for PyTorch / DistilBERT transformer models.

### Step-by-Step Instructions:
1. Go to **[https://huggingface.co/spaces](https://huggingface.co/spaces)** and log in.
2. Click **Create new Space**.
3. Settings:
   - **Space Name**: `asedf-social-threat-shield`
   - **Space SDK**: Select **Docker** (Blank).
   - **License**: MIT
   - **Visibility**: Public
4. Clone the space repository locally or push your code to the Hugging Face Git remote:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/asedf-social-threat-shield
   git push space main
   ```
5. Hugging Face will build the Docker container using our [`Dockerfile`](file:///c:/Users/abhra/OneDrive/Desktop/ai/unified_detector/Dockerfile) and launch your live app!

---

## ⚡ Option 3: Instant 10-Second Live Public URL via Ngrok (For Live Pitch / Demos)

If you are presenting to judges right now from your laptop and want a secure public HTTPS URL without waiting for cloud builds:

### Step-by-Step Instructions:
1. Install Ngrok (if not already installed):
   ```powershell
   winget install ngrok
   ```
2. Start your local server (if not already running):
   ```powershell
   python app.py
   ```
3. In a separate terminal, start the Ngrok tunnel:
   ```powershell
   ngrok http 5000
   ```
4. Ngrok will output a public HTTPS link:
   ```text
   Forwarding: https://abc123-x-threat.ngrok-free.app -> http://localhost:5000
   ```
5. You can open this link on your phone, tablet, or send it to judges during the presentation.

---

## 🐳 Option 4: Production Docker Container (On-Premise / AWS / GCP)

To run the containerized application on any Linux server, VPS, AWS EC2, or Google Cloud VM:

### Build and Run with Docker:
```bash
# 1. Build the Docker image
docker build -t asedf-detector:latest .

# 2. Run the container on port 5000
docker run -d -p 5000:5000 --name asedf-app asedf-detector:latest

# 3. View live logs
docker logs -f asedf-app
```

---

## 🔑 Environment Variables

When deploying to cloud providers (Render / Railway / AWS), set the following environment variables:

| Variable | Recommended Value | Purpose |
|---|---|---|
| `FLASK_SECRET_KEY` | `your_secret_production_key_here` | Session encryption |
| `FLASK_DEBUG` | `False` | Disables debug mode in production |
| `PORT` | `5000` | Application port |
