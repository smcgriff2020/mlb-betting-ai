import joblib
import pandas as pd
import sys
import os
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found in environment variables.")

openai.api_key = OPENAI_API_KEY

def predict(home_team, away_team):
    try:
        model_bundle = joblib.load("models/model.pkl")
    except FileNotFoundError:
        print("❌ Model file not found. Run train_model.py first.")
        return

    winner_model = model_bundle["winner_model"]
    regression_models = model_bundle["regression_models"]
    feature_columns = model_bundle["feature_columns"]

    input_data = {col: 0 for col in feature_columns}
    home_col = f"HomeTeam_{home_team}"
    away_col = f"AwayTeam_{away_team}"

    if home_col not in input_data or away_col not in input_data:
        print(f"⚠️ One or both teams not found in training data.")
        return

    input_data[home_col] = 1
    input_data[away_col] = 1
    input_df = pd.DataFrame([input_data])

    # Predict winner
    winner_pred = winner_model.predict(input_df)[0]
    winner = home_team if winner_pred == 1 else away_team

    # Predict all stats
    stat_preds = {}
    for stat, model in regression_models.items():
        stat_preds[stat] = model.predict(input_df)[0]

    print(f"🏆 Predicted Winner: {winner}")
    print("📊 Stat Predictions:")
    for stat, val in stat_preds.items():
        print(f"  {stat}: {val:.2f}")

    # GPT explanation
    stat_summary = ", ".join(f"{k} = {v:.2f}" for k, v in stat_preds.items())
    prompt = (
        f"Predicting MLB game: {home_team} (home) vs {away_team} (away). "
        f"Model predicts winner: {winner}. Predicted stats: {stat_summary}. "
        "Explain why the model might make these predictions using baseball analytics reasoning."
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a baseball analytics expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=500,
        )
        explanation = response.choices[0].message.content.strip()
        print("\n🤖 GPT Analysis:")
        print(explanation)
    except Exception as e:
        print(f"❌ GPT error: {e}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        predict(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python predict.py 'Home Team Name' 'Away Team Name'")
