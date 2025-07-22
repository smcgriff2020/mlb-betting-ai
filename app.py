from flask import Flask, render_template, request
import joblib
import pandas as pd
import openai
import os
import json
from dotenv import load_dotenv

# Load API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

app = Flask(__name__)

# Global model cache
model_data = None

def load_models():
    global model_data
    if model_data is None:
        model_data = joblib.load("models/model.pkl")

def get_teams():
    load_models()
    teams = set()
    for col in model_data["feature_columns"]:
        if col.startswith("HomeTeam_"):
            teams.add(col.replace("HomeTeam_", ""))
        elif col.startswith("AwayTeam_"):
            teams.add(col.replace("AwayTeam_", ""))
    return sorted(teams)

def predict_game(home_team, away_team):
    load_models()
    winner_model = model_data["winner_model"]
    regression_models = model_data["regression_models"]
    feature_columns = model_data["feature_columns"]

    input_data = {col: 0 for col in feature_columns}
    home_col = f"HomeTeam_{home_team}"
    away_col = f"AwayTeam_{away_team}"

    if home_col not in input_data or away_col not in input_data:
        return {"error": f"One or both teams ('{home_team}', '{away_team}') not found in training data."}

    input_data[home_col] = 1
    input_data[away_col] = 1
    input_df = pd.DataFrame([input_data])

    winner_pred = winner_model.predict(input_df)[0]
    winner = home_team if winner_pred == 1 else away_team

    stats_predictions = {}
    for stat, model in regression_models.items():
        pred_val = model.predict(input_df)[0]
        stats_predictions[stat] = float(pred_val)

    stats_str = ", ".join(f"{k}={v:.2f}" for k, v in stats_predictions.items())
    prompt = (
        f"Analyze the upcoming MLB game between {home_team} (home) and {away_team} (away). "
        f"The model predicts the winner is {winner}. "
        f"Predicted stats include: {stats_str}. "
        "Explain why based on historical data, roster performance, and recent trends."
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful baseball analytics assistant who explains MLB predictions using advanced insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=600,
        )
        gpt_output = response.choices[0].message.content.strip()
    except Exception as e:
        gpt_output = f"OpenAI API error: {e}"

    return {
        "winner": winner,
        "stats": stats_predictions,
        "analysis": gpt_output
    }

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    home_team = ""
    away_team = ""
    teams = get_teams()

    if request.method == "POST":
        home_team = request.form.get("home_team")
        away_team = request.form.get("away_team")

        if not home_team or not away_team:
            error = "Please select both home and away teams."
        else:
            result = predict_game(home_team, away_team)
            if "error" in result:
                error = result["error"]
            else:
                prediction = result

    return render_template("index.html",
                           prediction=prediction,
                           error=error,
                           home_team=home_team,
                           away_team=away_team,
                           teams=teams)

@app.route("/ask", methods=["GET", "POST"])
def ask():
    answer = None
    if request.method == "POST":
        question = request.form.get("question")
        if question:
            try:
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful MLB analytics assistant."},
                        {"role": "user", "content": question}
                    ],
                    temperature=0.7,
                    max_tokens=600
                )
                answer = response.choices[0].message.content.strip()
            except Exception as e:
                answer = f"OpenAI API error: {e}"
    return render_template("ask.html", answer=answer)

@app.route("/live_data", methods=["GET"])
def live_data():
    import datetime
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    path = f"data/live_games_{today}.json"

    if not os.path.exists(path):
        return {"error": "No live data yet."}

    with open(path, "r") as f:
        data = json.load(f)

    return {"games": data}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
