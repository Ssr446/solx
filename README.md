<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" />
  
  <h1>🛡️ SOLX</h1>
  <h3>Sarcasm & Online harm Locator with eXplainability</h3>
  <p><em>Tamil Digital Safety • AI-Powered Tanglish Text Analysis</em></p>
  
  <p>
    <a href="https://solx-seven.vercel.app/index.html"><strong>Explore the Landing Page</strong></a> ·
    <a href="https://solx-seven.vercel.app/solx_app.html"><strong>Test the Live Dashboard</strong></a>
  </p>
</div>

<hr />

## 🌟 Overview
**SOLX** is a culturally-aware AI moderation system designed to detect toxicity, cyberbullying, and sarcasm natively within complex code-mixed **"Tanglish"** (Tamil-English) text. 

Unlike traditional English-only models, SOLX understands the cultural nuances, pop-culture sarcasm, and extreme spelling chaos of modern South Indian digital communication.

## ✨ Key Features
- **Explainable AI (XAI) Cultural Context Engine:** Generates transparent, human-readable rationales explaining exactly why specific regional slang or pop-culture phrases were flagged (e.g., explaining movie references used sarcastically).
- **Phonetic Normalization:** An advanced preprocessing layer that normalizes dozens of spelling variations (e.g., *"enna"*, *"yenna"*, *"ena"*) into a unified root form before classification.
- **Multi-Label Detection:** Handles intersectional online harms, predicting independent confidence scores for Toxicity, Sarcasm, and Cyberbullying simultaneously.
- **Hybrid Architecture:** Seamlessly integrates a highly-optimized machine learning pipeline (`LinearSVC` + `Char N-grams`) with a rule-based knowledge guardrail system.

## 🛠️ Architecture & Tech Stack
* **Backend Pipeline:** Python, Scikit-learn, FastAPI, Uvicorn
* **Frontend Dashboard:** Pure HTML, CSS (Glass-morphism), Vanilla JS
* **Deployment:**
  * **API:** Hosted on Render (Web Service)
  * **Dashboard:** Hosted on Vercel (Static Site)

## 🚀 Local Setup & Installation

If you wish to run the AI pipeline and dashboard locally:

### 1. Start the Backend API
```bash
# Navigate to the backend directory
cd backend

# Install the exact dependencies (requires Python 3.12+)
pip install -r requirements.txt

# Start the FastAPI server
python main.py
```
*The server will start running at `http://localhost:8000`.*

### 2. Launch the Frontend
Because the frontend is pure HTML/JS, no build step is required! 
Simply open `index.html` or `solx_app.html` in your web browser. 

*(Note: To test local changes, ensure you update the `fetch()` URL in the JavaScript files to point back to `localhost:8000` instead of the live Render URL).*

## 👥 Team Inferno
Built for the NLP Hackathon 2026.
* **Y Roshni** 
* **Sai Shivaram LN** 
* **Archana Ramesh**
