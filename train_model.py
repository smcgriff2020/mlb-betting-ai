import os
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression, PoissonRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

def train():
    data_path = "data/clean_games.csv"
    model_path = "models/model.pkl"

    if not os.path.exists(data_path):
        print("❌ data/clean_games.csv not found. Run preprocess.py first.")
        return

    # Load and validate data
    df = pd.read_csv(data_path)
    if "Winner" not in df.columns:
        print("❌ 'Winner' column is missing in the dataset.")
        return

    # One-hot encode teams
    df_encoded = pd.get_dummies(df, columns=["HomeTeam", "AwayTeam"])

    # Separate inputs (X) and outputs (Y)
    team_cols = [col for col in df_encoded.columns if col.startswith("HomeTeam_") or col.startswith("AwayTeam_")]
    stat_targets = [col for col in df_encoded.columns if col.startswith("int")]

    X = df_encoded[team_cols]  # Base features: team one-hot encodings
    y_winner = df_encoded["Winner"]

    # Train classification model
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_winner, test_size=0.2, random_state=42)
    winner_model = LogisticRegression(max_iter=1000)
    winner_model.fit(X_train_clf, y_train_clf)
    acc = accuracy_score(y_test_clf, winner_model.predict(X_test_clf))
    print(f"✅ Winner prediction accuracy: {acc:.2f}")

    # Train regression models for each stat
    regression_models = {}

    for stat in stat_targets:
        y = df_encoded[stat]
        X_stat = df_encoded[team_cols]  # Only teams as predictors

        X_train, X_test, y_train, y_test = train_test_split(X_stat, y, test_size=0.2, random_state=42)

        if "Hits" in stat or "Runs" in stat:
            model = PoissonRegressor(max_iter=300)
        elif "Strikeouts" in stat or "Pitches" in stat:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = LinearRegression()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"✅ {stat} regression MSE: {mse:.2f}")
        regression_models[stat] = model

    # Save model bundle
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "winner_model": winner_model,
        "regression_models": regression_models,
        "feature_columns": list(X.columns)
    }, model_path)

    print(f"✅ All models saved to {model_path}")

if __name__ == "__main__":
    train()
