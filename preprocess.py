# preprocess.py

import json
import pandas as pd
import os

def preprocess():
    json_path = "data/games.json"
    if not os.path.exists(json_path):
        print("❌ data/games.json not found. Run fetch_data.py first.")
        return

    with open(json_path, "r") as f:
        events = json.load(f)

    rows = []

    # Dynamically find all stat keys that start with 'int' and contain numeric values
    stat_keys = set()
    for game in events:
        for k, v in game.items():
            if k.startswith("int") and v not in [None, "", "null"]:
                try:
                    float(v)
                    stat_keys.add(k)
                except:
                    continue

    stat_keys = list(stat_keys)

    for game in events:
        row = {
            "HomeTeam": game.get("strHomeTeam"),
            "AwayTeam": game.get("strAwayTeam"),
            "Winner": None,
            "HomeScore": None,
            "AwayScore": None
        }

        # Add all numeric stats
        for key in stat_keys:
            try:
                val = game.get(key)
                row[key] = int(val) if val is not None else 0
            except:
                row[key] = 0

        # Calculate winner based on final score
        try:
            home_score = int(game.get("intHomeScore"))
            away_score = int(game.get("intAwayScore"))
            row["HomeScore"] = home_score
            row["AwayScore"] = away_score
            row["Winner"] = 1 if home_score > away_score else 0
        except:
            continue  # skip rows without scores

        rows.append(row)

    df = pd.DataFrame(rows)

    # Drop any still-missing values in key fields
    df = df.dropna(subset=["Winner", "HomeTeam", "AwayTeam"])

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/clean_games.csv", index=False)
    print(f"✅ Processed {len(df)} games and saved to data/clean_games.csv")

if __name__ == "__main__":
    preprocess()
