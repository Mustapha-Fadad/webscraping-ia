# 🤖 AI Community Manager

This project is part of the internship *Agent IA Community Manager avec Web Scraping*.  
It collects and analyzes data from different sources (news, forums, social media, trends), stores it in a database, and generates insights + recommendations for community management.  

---

## 📂 Project Structure
```

ai-community-manager/
│── ai\_agent/          # Orchestrator & AI logic
│── scrapers/          # Scrapers for news, forums, social, trends
│── utils/             # Cleaning, analytics, storage
│── config/            # Settings & environment variables
│── main.py            # Main entry point
│── .env.template      # Example of environment variables (safe to share)
│── README.md          # This file

````

---

## ⚙️ Installation

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/ai-community-manager.git
cd ai-community-manager
````

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

1. Copy `.env.template` → `.env`

```bash
cp .env.template .env
```

2. Fill in your real API credentials:

```ini
TWITTER_API_KEY=...
TWITTER_API_SECRET=...
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
LINKEDIN_CLIENT_ID=...
LINKEDIN_CLIENT_SECRET=...
DATABASE_URL=sqlite:///data/community_manager.db
TRENDING_KEYWORDS="AI,artificial intelligence,machine learning,data science,python"
```

⚠️ **Never commit your real `.env` file.** Only use `.env.template`.

---

## ▶️ Run the project

```bash
python main.py
```

Logs will be saved in the `logs/` folder.
Analysis results and recommendations are stored in the database and exported as JSON/CSV.

---

## 📊 Current Features

* ✅ Web scraping (news, forums, social, trends)
* ✅ Data cleaning & enrichment
* ✅ SQLite database storage
* ✅ Basic analytics & keyword trends
* ✅ AI-driven recommendations (simulated, extendable with GPT-4/Hugging Face)
* ✅ Logging system
