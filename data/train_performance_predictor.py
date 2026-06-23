"""
data/train_performance_predictor.py
Trains the PerformancePredictor model ensemble (XGBoost, Gradient Boosting, Random Forest)
on augmented student Portuguese/Math grades dataset.
"""
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier

# Add parent directory to path so we can import backend packages if needed
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import build_student_features and _compute_trend
from backend.services.ml.analytics_service import build_student_features

MODEL_DIR = Path("./data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_and_augment_data():
    mat_path = Path("./data/models/student/student-mat.csv")
    por_path = Path("./data/models/student/student-por.csv")
    
    if not mat_path.exists() or not por_path.exists():
        raise FileNotFoundError("Student CSV files not found under data/models/student/")
        
    df_mat = pd.read_csv(mat_path, sep=";")
    df_por = pd.read_csv(por_path, sep=";")
    
    # Add subject indicator
    df_mat["is_math"] = 1
    df_por["is_math"] = 0
    
    df = pd.concat([df_mat, df_por], ignore_index=True)
    print(f"Loaded {len(df)} original student records.")
    
    # Data Augmentation: Generate perturbed copies of students
    augmented_students = []
    np.random.seed(42)
    
    # 5x augmentation factor
    for idx, row in df.iterrows():
        # Keep original
        augmented_students.append(row.to_dict())
        
        for _ in range(5):
            new_row = row.copy().to_dict()
            # Perturb G1, G2, G3 slightly
            new_row["G1"] = int(np.clip(row["G1"] + np.random.normal(0, 0.6), 0, 20))
            new_row["G2"] = int(np.clip(row["G2"] + np.random.normal(0, 0.6), 0, 20))
            new_row["G3"] = int(np.clip(row["G3"] + np.random.normal(0, 0.6), 0, 20))
            # Perturb studytime (1-4)
            if np.random.rand() < 0.2:
                new_row["studytime"] = int(np.clip(row["studytime"] + np.random.choice([-1, 1]), 1, 4))
            # Perturb absences
            new_row["absences"] = int(np.clip(row["absences"] + np.random.randint(-2, 3), 0, 93))
            augmented_students.append(new_row)
            
    # Oversample/augment low performers specifically to define the fail/pass classifier boundary
    for _ in range(1500):
        low_g1 = np.random.randint(1, 7)
        low_g2 = np.random.randint(1, 7)
        low_g3 = np.random.randint(1, 7)  # All below 8 (i.e., final score < 40%)
        augmented_students.append({
            "G1": low_g1,
            "G2": low_g2,
            "G3": low_g3,
            "studytime": np.random.randint(1, 3),
            "failures": np.random.randint(2, 5),
            "absences": np.random.randint(5, 25),
            "is_math": np.random.choice([0, 1])
        })
            
    df_aug = pd.DataFrame(augmented_students)
    print(f"Augmented dataset to {len(df_aug)} student profiles.")
    return df_aug

def simulate_attempts_for_student(student_row):
    """
    Simulates quiz attempts for a student based on their background characteristics.
    Outputs a list of dicts that can be passed to build_student_features.
    """
    g1 = student_row["G1"]
    g2 = student_row["G2"]
    studytime = student_row["studytime"]
    failures = student_row["failures"]
    
    # Map grades (0-20) to score percentages (0-100)
    g1_pct = (g1 / 20.0) * 100.0
    g2_pct = (g2 / 20.0) * 100.0
    
    attempts = []
    topics = ["Kinematics", "Newton's Laws", "Work-Energy Theorem", "Rotational Motion", 
              "Gravitation", "Thermodynamics", "Electrostatics", "Current Electricity"]
              
    # Random number of attempts between 6 and 12
    num_attempts = np.random.randint(6, 13)
    
    for i in range(num_attempts):
        topic = np.random.choice(topics)
        # Earlier attempts reflect G1 performance, later reflect G2
        phase_pct = g1_pct if i < (num_attempts / 2) else g2_pct
        
        # Random difficulty choice to create variance in features
        difficulty = np.random.choice(["easy", "medium", "hard"], p=[0.25, 0.50, 0.25])
        
        if difficulty == "easy":
            base_score = min(phase_pct + 12.0, 100.0)
            base_time = 18.0
        elif difficulty == "hard":
            base_score = max(phase_pct - 12.0, 0.0)
            base_time = 75.0
        else:
            difficulty = "medium"
            base_score = phase_pct
            base_time = 40.0
            
        # Adjust score and time spent based on study time and failures
        score_adj = (studytime - 2) * 3.0 - failures * 5.0
        time_adj = (studytime - 2) * 5.0 - failures * 2.0
        
        score = base_score + score_adj + np.random.normal(0, 6.0)
        score = np.clip(score, 0.0, 100.0)
        
        time_spent = base_time + time_adj + np.random.normal(0, 4.0)
        time_spent = np.clip(time_spent, 5.0, 180.0)
        
        attempts.append({
            "topic": topic,
            "score": round(score, 1),
            "max_score": 100.0,
            "difficulty": difficulty,
            "time_spent_mins": round(time_spent, 1),
            "attempted_at": f"2024-01-{i+1:02d}T10:00:00"
        })
        
    return attempts

def train_models():
    df_aug = load_and_augment_data()
    
    X_list = []
    y_scores = []
    
    print("Simulating quiz attempts and engineering features for training...")
    for idx, row in df_aug.iterrows():
        attempts = simulate_attempts_for_student(row)
        features_df = build_student_features(attempts)
        if not features_df.empty:
            X_list.append(features_df.iloc[0])
            # Target is G3 (final grade) scaled to 100
            y_scores.append((row["G3"] / 20.0) * 100.0)
            
    X = pd.DataFrame(X_list)
    y_scores = np.array(y_scores)
    y_pass = (y_scores >= 40.0).astype(int)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define models
    xgb_reg = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    gb_reg = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        random_state=42
    )
    rf_reg = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    
    # Pass classifier (predict probability of passing)
    pass_clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        verbosity=0
    )
    
    print("Training models...")
    xgb_reg.fit(X_scaled, y_scores)
    gb_reg.fit(X_scaled, y_scores)
    rf_reg.fit(X_scaled, y_scores)
    pass_clf.fit(X_scaled, y_pass)
    
    # Evaluate on training data (cross validation can be printed but let's do a quick train set evaluation)
    pred_xgb = xgb_reg.predict(X_scaled)
    pred_gb = gb_reg.predict(X_scaled)
    pred_rf = rf_reg.predict(X_scaled)
    pred_ensemble = (pred_xgb + pred_gb + pred_rf) / 3.0
    
    mae = np.mean(np.abs(y_scores - pred_ensemble))
    rmse = np.sqrt(np.mean((y_scores - pred_ensemble) ** 2))
    
    print(f"Training set evaluation:")
    print(f"  Ensemble MAE: {mae:.2f}%")
    print(f"  Ensemble RMSE: {rmse:.2f}%")
    
    # Save the models
    joblib.dump(xgb_reg, MODEL_DIR / "score_predictor_xgb.joblib")
    joblib.dump(gb_reg, MODEL_DIR / "score_predictor_gb.joblib")
    joblib.dump(rf_reg, MODEL_DIR / "score_predictor_rf.joblib")
    joblib.dump(pass_clf, MODEL_DIR / "pass_classifier.joblib")
    joblib.dump(scaler, MODEL_DIR / "feature_scaler.joblib")
    print("Successfully trained and saved all Performance Prediction models.")

if __name__ == "__main__":
    train_models()
