"""
backend/services/ml/analytics_service.py
XGBoost/RF-based personalized learning recommendations and performance prediction.
"""
from __future__ import annotations
import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor, XGBClassifier

from backend.core.logging import get_logger
from backend.models.schemas import LearningRecommendation, PerformancePrediction

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

MODEL_DIR = Path("./data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ─── Feature Engineering ──────────────────────────────────────────────────────

def build_student_features(quiz_attempts: list[dict]) -> pd.DataFrame:
    """
    Build feature matrix from raw quiz attempt records.
    Each attempt: {topic, score, max_score, difficulty, time_spent_mins, attempted_at}
    """
    if not quiz_attempts:
        return pd.DataFrame()

    df = pd.DataFrame(quiz_attempts)
    df["pct_score"] = df["score"] / df["max_score"].replace(0, 100) * 100
    df["attempted_at"] = pd.to_datetime(df["attempted_at"])
    df = df.sort_values("attempted_at")

    # Difficulty encoding
    diff_map = {"easy": 1, "medium": 2, "hard": 3}
    df["difficulty_enc"] = df["difficulty"].map(diff_map).fillna(2)

    # Per-student aggregate features
    features = {
        "avg_score": df["pct_score"].mean(),
        "score_std": df["pct_score"].std(ddof=0),
        "min_score": df["pct_score"].min(),
        "max_score_val": df["pct_score"].max(),
        "num_attempts": len(df),
        "avg_time_spent": df["time_spent_mins"].mean(),
        "hard_attempt_ratio": (df["difficulty_enc"] == 3).mean(),
        "easy_attempt_ratio": (df["difficulty_enc"] == 1).mean(),
        "unique_topics": df["topic"].nunique(),
        "score_trend": _compute_trend(df["pct_score"].values),
        "recent_avg": df["pct_score"].tail(5).mean(),
        "consistency": 1 - (df["pct_score"].std(ddof=0) / (df["pct_score"].mean() + 1e-6)),
    }

    return pd.DataFrame([features])


def _compute_trend(scores: np.ndarray) -> float:
    """Linear regression slope of scores over time."""
    if len(scores) < 2:
        return 0.0
    x = np.arange(len(scores))
    try:
        slope = np.polyfit(x, scores, 1)[0]
        return float(slope)
    except Exception:
        return 0.0


def get_topic_weaknesses(quiz_attempts: list[dict], threshold: float = 60.0) -> list[str]:
    """Return topics where average performance is below threshold."""
    if not quiz_attempts:
        return []
    df = pd.DataFrame(quiz_attempts)
    df["pct_score"] = df["score"] / df["max_score"].replace(0, 100) * 100
    topic_avg = df.groupby("topic")["pct_score"].mean()
    return topic_avg[topic_avg < threshold].sort_values().index.tolist()


# ─── Model Training ───────────────────────────────────────────────────────────

class PerformancePredictor:
    """Ensemble-based (XGBoost, Gradient Boosting, Random Forest) exam score and pass probability predictor."""

    SCORE_XGB_PATH = MODEL_DIR / "score_predictor_xgb.joblib"
    SCORE_GB_PATH = MODEL_DIR / "score_predictor_gb.joblib"
    SCORE_RF_PATH = MODEL_DIR / "score_predictor_rf.joblib"
    PASS_MODEL_PATH = MODEL_DIR / "pass_classifier.joblib"
    SCALER_PATH = MODEL_DIR / "feature_scaler.joblib"

    def __init__(self):
        self.xgb_model: XGBRegressor | None = None
        self.gb_model: GradientBoostingRegressor | None = None
        self.rf_model: RandomForestRegressor | None = None
        self.pass_model: XGBClassifier | None = None
        self.scaler: StandardScaler | None = None
        self._load_or_init_models()

    def _load_or_init_models(self) -> None:
        if self.SCORE_XGB_PATH.exists() and self.SCORE_GB_PATH.exists() and self.SCORE_RF_PATH.exists():
            self.xgb_model = joblib.load(self.SCORE_XGB_PATH)
            self.gb_model = joblib.load(self.SCORE_GB_PATH)
            self.rf_model = joblib.load(self.SCORE_RF_PATH)
            self.pass_model = joblib.load(self.PASS_MODEL_PATH)
            self.scaler = joblib.load(self.SCALER_PATH)
            logger.info("Loaded pretrained prediction models (XGB, GB, RF)")
        else:
            self._init_fresh_models()
            logger.info("Initialised fresh prediction models")

    def _init_fresh_models(self) -> None:
        self.xgb_model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        self.gb_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        )
        self.rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            random_state=42
        )
        self.pass_model = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
            verbosity=0,
        )
        self.scaler = StandardScaler()

    def train(self, X: pd.DataFrame, y_scores: np.ndarray) -> dict:
        """Train both score regressors and pass classifier."""
        if len(X) < 5:
            logger.warning("Insufficient data for training", samples=len(X))
            return {"status": "insufficient_data"}

        X_scaled = self.scaler.fit_transform(X)
        y_pass = (y_scores >= 40).astype(int)

        self.xgb_model.fit(X_scaled, y_scores)
        self.gb_model.fit(X_scaled, y_scores)
        self.rf_model.fit(X_scaled, y_scores)
        self.pass_model.fit(X_scaled, y_pass)

        # Save models
        joblib.dump(self.xgb_model, self.SCORE_XGB_PATH)
        joblib.dump(self.gb_model, self.SCORE_GB_PATH)
        joblib.dump(self.rf_model, self.SCORE_RF_PATH)
        joblib.dump(self.pass_model, self.PASS_MODEL_PATH)
        joblib.dump(self.scaler, self.SCALER_PATH)

        return {"status": "trained"}

    def predict(self, features: pd.DataFrame) -> dict:
        """Predict exam score and pass probability."""
        if self.scaler is None or self.xgb_model is None or self.gb_model is None or self.rf_model is None:
            return {"predicted_score": 70.0, "pass_probability": 0.75, "status": "default"}

        try:
            X_scaled = self.scaler.transform(features)
            pred_xgb = float(np.clip(self.xgb_model.predict(X_scaled)[0], 0, 100))
            pred_gb = float(np.clip(self.gb_model.predict(X_scaled)[0], 0, 100))
            pred_rf = float(np.clip(self.rf_model.predict(X_scaled)[0], 0, 100))
            
            # Ensemble predicted score
            predicted_score = (pred_xgb + pred_gb + pred_rf) / 3.0
            
            pass_prob = float(self.pass_model.predict_proba(X_scaled)[0][1])

            # Ensure logical coherence by calibrating the classifier's pass probability
            # with a score-based sigmoid curve. This prevents high pass probability on low exam scores.
            pass_prob_calibrated = 1.0 / (1.0 + np.exp(-(predicted_score - 40.0) / 10.0))
            pass_prob = 0.5 * pass_prob + 0.5 * pass_prob_calibrated

            # Feature importances from XGBoost model
            feat_names = features.columns.tolist()
            importances = dict(zip(feat_names, self.xgb_model.feature_importances_.tolist()))

            return {
                "predicted_score": round(predicted_score, 1),
                "pass_probability": round(pass_prob, 3),
                "feature_importances": importances,
                "status": "ok",
            }
        except Exception as e:
            logger.warning("Prediction failed, using heuristic", error=str(e))
            return _heuristic_prediction(features)


class LearningRecommender:
    """Random Forest and XGBoost ensemble personalized learning recommender."""

    MODEL_PATH = MODEL_DIR / "recommender.joblib"

    SUBJECT_TOPIC_ORDER = {
        "Physics": [
            "Units and Measurements", "Kinematics", "Newton's Laws",
            "Work Energy Power", "Rotational Motion", "Gravitation",
            "Thermodynamics", "Waves", "Electrostatics", "Current Electricity",
            "Electromagnetic Induction", "Optics", "Modern Physics",
        ],
        "Chemistry": [
            "Atomic Structure", "Periodic Table", "Chemical Bonding",
            "States of Matter", "Thermodynamics", "Equilibrium",
            "Electrochemistry", "Organic Chemistry Basics",
        ],
        "Mathematics": [
            "Sets and Functions", "Trigonometry", "Algebra",
            "Coordinate Geometry", "Calculus Basics", "Differential Calculus",
            "Integral Calculus", "Vectors", "Probability",
        ],
        "Biology": [
            "Cell Biology", "Genetics", "Evolution", "Physiology",
            "Ecology", "Biochemistry"
        ],
        "Computer Science": [
            "Data Structures", "Algorithms", "Operating Systems",
            "DBMS", "Networks", "Machine Learning"
        ],
        "Economics": [
            "Microeconomics", "Macroeconomics", "Statistics",
            "Game Theory", "International Trade"
        ],
        "Engineering": [
            "Circuits", "Signals & Systems", "Control Systems",
            "Thermodynamics", "Fluid Mechanics", "Material Science"
        ]
    }

    def __init__(self):
        self.rf_model: RandomForestRegressor | None = None
        self.xgb_model: XGBRegressor | None = None
        self.subject_to_code: dict[str, int] = {}
        self._init_model()

    def _init_model(self) -> None:
        if self.MODEL_PATH.exists():
            try:
                data = joblib.load(self.MODEL_PATH)
                self.rf_model = data.get("rf_model")
                self.xgb_model = data.get("xgb_model")
                self.subject_to_code = data.get("subject_to_code", {})
                logger.info("Loaded pretrained recommender ensemble models")
            except Exception as e:
                logger.error("Failed to load recommender models, using fallback", error=str(e))
                self._init_fallback()
        else:
            self._init_fallback()

    def _init_fallback(self) -> None:
        self.rf_model = None
        self.xgb_model = None
        self.subject_to_code = {}

    def generate_recommendation(
        self,
        student_id: str,
        quiz_attempts: list[dict],
        subject: str = "Physics",
    ) -> LearningRecommendation:
        weak_topics = get_topic_weaknesses(quiz_attempts, threshold=65.0)

        # 1. Compute student average profile
        if quiz_attempts:
            df = pd.DataFrame(quiz_attempts)
            df["pct_score"] = df["score"] / df["max_score"].replace(0, 100) * 100
            
            student_avg_score = float(df["pct_score"].mean())
            student_avg_time = float(df["time_spent_mins"].mean())
            student_avg_accuracy = float((df["score"] / df["max_score"]).mean())
        else:
            student_avg_score = 65.0
            student_avg_time = 45.0
            student_avg_accuracy = 0.65

        # 2. Get subjects and curriculum topics
        curriculum = self.SUBJECT_TOPIC_ORDER.get(subject, ["General"])
        sub_code = self.subject_to_code.get(subject, 0)

        predicted_topic_scores = {}
        
        # 3. Predict expected performance for each curriculum topic using the trained models
        if self.rf_model is not None and self.xgb_model is not None:
            try:
                for topic in curriculum:
                    features = pd.DataFrame([{
                        "student_avg_score": student_avg_score,
                        "student_avg_time": student_avg_time,
                        "student_avg_accuracy": student_avg_accuracy,
                        "subject_code": sub_code,
                        "difficulty_code": 2  # Predict for medium difficulty
                    }])
                    
                    pred_rf = float(self.rf_model.predict(features)[0])
                    pred_xgb = float(self.xgb_model.predict(features)[0])
                    # Ensemble predicted score
                    predicted_score = (pred_rf + pred_xgb) / 2.0
                    predicted_topic_scores[topic] = predicted_score
            except Exception as e:
                logger.warning("Recommender inference failed, falling back to heuristics", error=str(e))
                predicted_topic_scores = {}

        # 4. Filter weak topics and determine recommended study order
        if predicted_topic_scores:
            recommended_order = sorted(curriculum, key=lambda t: predicted_topic_scores.get(t, 65.0))
            predicted_weak = [t for t, score in predicted_topic_scores.items() if score < 65.0]
            weak_topics = list(set(weak_topics + predicted_weak))
            revision_priorities = [t for t in recommended_order if t in weak_topics][:3]
            
            weak_predicted_scores = [predicted_topic_scores[t] for t in weak_topics if t in predicted_topic_scores]
            if weak_predicted_scores:
                predicted_improvement = min(15.0, np.mean([max(0.0, 75.0 - s) for s in weak_predicted_scores]) * 0.8)
            else:
                predicted_improvement = 5.0
        else:
            if quiz_attempts:
                df = pd.DataFrame(quiz_attempts)
                df["pct_score"] = df["score"] / df["max_score"].replace(0, 100) * 100
                topic_avg = df.groupby("topic")["pct_score"].mean().to_dict()
            else:
                topic_avg = {}
            weak_topics_sorted = sorted(weak_topics, key=lambda t: topic_avg.get(t, 0))
            
            studied_topics = set(topic_avg.keys())
            not_started = [t for t in curriculum if t not in studied_topics]
            recommended_order = weak_topics_sorted[:5] + not_started[:3]
            revision_priorities = weak_topics_sorted[:3]
            predicted_improvement = min(15.0, len(weak_topics) * 2.5)

        return LearningRecommendation(
            student_id=student_id,
            weak_topics=weak_topics,
            recommended_order=recommended_order,
            revision_priorities=revision_priorities,
            predicted_improvement=round(predicted_improvement, 1),
        )


# ─── Prediction Heuristic Fallback ────────────────────────────────────────────

def _heuristic_prediction(features: pd.DataFrame) -> dict:
    row = features.iloc[0]
    avg = float(row.get("avg_score", 60))
    trend = float(row.get("score_trend", 0))
    recent = float(row.get("recent_avg", avg))
    predicted = np.clip(recent + trend * 2, 0, 100)
    pass_prob = float(1 / (1 + np.exp(-(predicted - 40) / 10)))
    return {
        "predicted_score": round(float(predicted), 1),
        "pass_probability": round(pass_prob, 3),
        "feature_importances": {},
        "status": "heuristic",
    }
