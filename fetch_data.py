# fetch_data.py

import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()
SPORTSDATAIO_API_KEY = os.getenv("SPORTSDATAIO_API_KEY")
LIVE_SCORE_URL = f"https://api.sportsdata.io/v4/mlb/scores/json/GamesByDate/{{date}}?key={SPORTSDATAIO_API_KEY}"

def fetch_live_games(date):
    """Fetch live MLB games for a given date (format: YYYY-MM-DD)"""
    url = LIVE_SCORE_URL.format(date=date)
    response = requests.get(url)

    if response.status_code != 200:
        print(f"❌ Error fetching live games: {response.status_code} - {response.text}")
        return []

    games = response.json()
    os.makedirs("data", exist_ok=True)
    with open(f"data/live_games_{date}.json", "w") as f:
        json.dump(games, f, indent=2)
    print(f"✅ Fetched {len(games)} live games for {date}")
    return games

if __name__ == "__main__":
    import datetime
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    fetch_live_games(today)
if __name__ == "__main__":
    fetch_live_games()  # or fetch_season_events() if you're still using that

    # Trigger preprocessing
    try:
        print("⚙️ Running preprocess.py...")
        os.system("python preprocess.py")
    except Exception as e:
        print(f"[Error running preprocess.py] {e}")

    # OPTIONAL: Also retrain the model after preprocessing
    try:
        print("⚙️ Running train_model.py...")
        os.system("python train_model.py")
    except Exception as e:
        print(f"[Error running train_model.py] {e}")
