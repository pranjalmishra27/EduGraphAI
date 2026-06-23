"""
tests/test_analytics.py
Unit tests for the ML analytics service (feature engineering + predictions).
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from backend.services.ml.analytics_service import (
    build_student_features,
    get_topic_weaknesses,
    PerformancePredictor,
    LearningRecommender,
    _compute_trend,
)


def make_attempts(scores: list[float], topics: list[str] | None = None,
                  difficulty: str = "medium") -> list[dict]:
    if topics is None:
        topics = [f"Topic{i}" for i in range(len(scores))]
    base = datetime(2024, 1, 1)
    return [
        {
            "topic": topics[i % len(topics)],
            "score": s,
            "max_score": 100.0,
            "difficulty": difficulty,
            "time_spent_mins": 30.0,
            "attempted_at": (base + timedelta(days=i)).isoformat(),
        }
        for i, s in enumerate(scores)
    ]


class TestFeatureEngineering:

    def test_build_features_basic(self):
        attempts = make_attempts([70, 80, 60, 90, 75])
        features = build_student_features(attempts)
        assert not features.empty
        assert "avg_score" in features.columns
        assert "score_trend" in features.columns
        assert "num_attempts" in features.columns
        assert features["num_attempts"][0] == 5

    def test_build_features_empty(self):
        features = build_student_features([])
        assert features.empty

    def test_avg_score_correct(self):
        attempts = make_attempts([60, 80, 100])
        features = build_student_features(attempts)
        assert abs(features["avg_score"][0] - 80.0) < 0.01

    def test_trend_improving(self):
        scores = np.array([40, 50, 60, 70, 80])
        trend = _compute_trend(scores)
        assert trend > 0

    def test_trend_declining(self):
        scores = np.array([80, 70, 60, 50, 40])
        trend = _compute_trend(scores)
        assert trend < 0

    def test_trend_single_score(self):
        trend = _compute_trend(np.array([75.0]))
        assert trend == 0.0


class TestTopicWeaknesses:

    def test_identifies_weak_topics(self):
        attempts = make_attempts(
            [30, 35, 40, 90, 85],
            topics=["Thermodynamics", "Thermodynamics", "Thermodynamics", "Kinematics", "Kinematics"],
        )
        weak = get_topic_weaknesses(attempts, threshold=60.0)
        assert "Thermodynamics" in weak
        assert "Kinematics" not in weak

    def test_no_weak_topics(self):
        attempts = make_attempts([80, 90, 85, 95])
        weak = get_topic_weaknesses(attempts, threshold=60.0)
        assert len(weak) == 0

    def test_empty_attempts(self):
        assert get_topic_weaknesses([]) == []


class TestPerformancePredictor:

    def test_predict_returns_valid_score(self):
        predictor = PerformancePredictor()
        attempts = make_attempts([70, 75, 80, 65, 85, 78, 72, 88])
        features = build_student_features(attempts)
        result = predictor.predict(features)
        assert 0 <= result["predicted_score"] <= 100
        assert 0 <= result["pass_probability"] <= 1

    def test_high_performer_low_risk(self):
        predictor = PerformancePredictor()
        attempts = make_attempts([88, 90, 92, 85, 95, 87, 91, 89])
        features = build_student_features(attempts)
        result = predictor.predict(features)
        assert result["pass_probability"] >= 0.6

    def test_low_performer_risk(self):
        predictor = PerformancePredictor()
        attempts = make_attempts([25, 30, 20, 35, 28, 32, 22, 27])
        features = build_student_features(attempts)
        result = predictor.predict(features)
        assert result["pass_probability"] <= 0.6


class TestLearningRecommender:

    def test_recommends_weak_topics_first(self):
        recommender = LearningRecommender()
        attempts = make_attempts(
            [30, 35, 90, 85],
            topics=["Thermodynamics", "Thermodynamics", "Kinematics", "Kinematics"],
        )
        rec = recommender.generate_recommendation("stu_001", attempts, "Physics")
        assert "Thermodynamics" in rec.weak_topics
        assert len(rec.recommended_order) > 0

    def test_empty_attempts_no_crash(self):
        recommender = LearningRecommender()
        rec = recommender.generate_recommendation("stu_002", [], "Physics")
        assert rec.student_id == "stu_002"
        assert isinstance(rec.weak_topics, list)
        assert isinstance(rec.recommended_order, list)

    def test_predicted_improvement_positive(self):
        recommender = LearningRecommender()
        attempts = make_attempts(
            [20, 25, 30] * 3,
            topics=["Topic A", "Topic B", "Topic C"] * 3,
        )
        rec = recommender.generate_recommendation("stu_003", attempts, "Physics")
        assert rec.predicted_improvement >= 0
