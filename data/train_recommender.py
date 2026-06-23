"""
data/train_recommender.py
Trains the LearningRecommender model ensemble (Random Forest, XGBoost)
on the synthetic Personalized Learning Recommendation System dataset.
"""
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

MODEL_DIR = Path("./data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_recommender():
    data_path = Path("./data/personalized_recommendation_dataset.csv")
    if not data_path.exists():
        raise FileNotFoundError("Synthetic dataset personalized_recommendation_dataset.csv not found.")
        
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with {len(df)} attempts.")
    
    # 1. Compute student-level average profile (student_avg_score, student_avg_time, student_avg_accuracy)
    student_profiles = df.groupby("student_id").agg({
        "score": "mean",
        "time_spent_mins": "mean",
        "accuracy": "mean"
    }).rename(columns={
        "score": "student_avg_score",
        "time_spent_mins": "student_avg_time",
        "accuracy": "student_avg_accuracy"
    }).reset_index()
    
    # Merge profiles back into attempts
    df_merged = df.merge(student_profiles, on="student_id")
    
    # 2. Encode categorical features
    subjects = [
        "Mathematics", "Physics", "Chemistry", "Biology", "Computer Science",
        "Engineering", "Economics", "Statistics", "History", "Literature",
        "Psychology", "Sociology", "Philosophy", "Political Science", "Geography",
        "Astronomy", "Environmental Science", "Art History", "Music Theory", "Linguistics"
    ]
    subject_to_code = {sub: idx for idx, sub in enumerate(subjects)}
    df_merged["subject_code"] = df_merged["subject"].map(subject_to_code).fillna(-1)
    
    difficulty_map = {"easy": 1, "medium": 2, "hard": 3}
    df_merged["difficulty_code"] = df_merged["difficulty"].map(difficulty_map).fillna(2)
    
    # 3. Define features X and target y
    features = [
        "student_avg_score",
        "student_avg_time",
        "student_avg_accuracy",
        "subject_code",
        "difficulty_code"
    ]
    
    X = df_merged[features]
    y = df_merged["score"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Split data into train size: {len(X_train)} and test size: {len(X_test)}")
    
    # 4. Train RandomForestRegressor and XGBRegressor
    print("Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    print("Training XGBoost Regressor...")
    xgb_model = XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    
    # Evaluate
    rf_pred = rf_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)
    ensemble_pred = (rf_pred + xgb_pred) / 2.0
    
    rf_mae = np.mean(np.abs(y_test - rf_pred))
    xgb_mae = np.mean(np.abs(y_test - xgb_pred))
    ensemble_mae = np.mean(np.abs(y_test - ensemble_pred))
    
    print(f"Evaluation results:")
    print(f"  Random Forest MAE: {rf_mae:.2f}")
    print(f"  XGBoost MAE: {xgb_mae:.2f}")
    print(f"  Ensemble MAE: {ensemble_mae:.2f}")
    
    # Save models and subject code dictionary
    joblib.dump({
        "rf_model": rf_model,
        "xgb_model": xgb_model,
        "subject_to_code": subject_to_code,
    }, MODEL_DIR / "recommender.joblib")
    
    print("Successfully trained and saved Recommender ensemble models to recommender.joblib")

if __name__ == "__main__":
    train_recommender()
