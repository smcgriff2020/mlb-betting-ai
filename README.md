# ⚾ MLB Sports Betting AI System

A powerful MLB prediction and analysis engine combining machine learning, statistical modeling, and GPT-4o insights. View stat projections (hits, runs, walks, pitch counts, etc.), choose game matchups, and get expert GPT writeups in real-time.

---

## 🔧 Features

- 🧠 Predict **winner** using logistic regression
- 📈 Predict **advanced stats**: hits, walks, strikeouts, pitch count, batting average, etc.
- 💬 GPT-powered **analysis** of predictions
- 📊 `/ask` route for **custom stat Q&A**
- 🔄 Auto-refresh predictions every minute
- 🌐 Web UI with team dropdowns, tabbed results, and analysis
- 📦 Built with Flask, scikit-learn, OpenAI, and TheSportsDB

---

## 🚀 Setup Instructions

### 1. Clone and create environment

```bash
git clone https://github.com/yourname/mlb-betting-ai.git
cd mlb-betting-ai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
