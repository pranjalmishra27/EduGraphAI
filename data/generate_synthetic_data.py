"""
data/generate_synthetic_data.py
Script to generate synthetic student attempt data for the Personalized Learning Recommendation System.
Generates data for 10,000 students and 20 subjects.
"""
import os
import numpy as np
import pandas as pd

def generate_dataset(num_students=10000, attempts_per_student=10):
    np.random.seed(42)
    
    # 20 subjects
    subjects = [
        "Mathematics", "Physics", "Chemistry", "Biology", "Computer Science",
        "Engineering", "Economics", "Statistics", "History", "Literature",
        "Psychology", "Sociology", "Philosophy", "Political Science", "Geography",
        "Astronomy", "Environmental Science", "Art History", "Music Theory", "Linguistics"
    ]
    
    print(f"Generating synthetic data for {num_students} students, each with {attempts_per_student} attempts across {len(subjects)} subjects...")
    
    # Pre-generate student traits to keep student behavior consistent
    # Latent ability: higher means higher score/accuracy
    student_abilities = np.random.normal(loc=0.0, scale=1.0, size=num_students)
    # Speed trait: positive means student takes more time than average
    student_speed_traits = np.random.normal(loc=0.0, scale=1.0, size=num_students)
    
    records = []
    
    for student_idx in range(num_students):
        student_id = f"STU_{student_idx + 1:05d}"
        ability = student_abilities[student_idx]
        speed_trait = student_speed_traits[student_idx]
        
        # Choose a subset of subjects this student is studying (e.g. 3 to 6 subjects)
        num_student_subjects = np.random.randint(3, 7)
        student_subjects = np.random.choice(subjects, size=num_student_subjects, replace=False)
        
        for attempt in range(attempts_per_student):
            # Select a random subject from the student's list
            subject = np.random.choice(student_subjects)
            
            # Difficulty: easy (30%), medium (50%), hard (20%)
            difficulty = np.random.choice(["easy", "medium", "hard"], p=[0.3, 0.5, 0.2])
            
            # Base parameters depending on difficulty
            if difficulty == "easy":
                base_time = 20.0
                base_score = 78.0
                difficulty_factor = 1.0
            elif difficulty == "medium":
                base_time = 45.0
                base_score = 63.0
                difficulty_factor = 2.0
            else:  # hard
                base_time = 80.0
                base_score = 48.0
                difficulty_factor = 3.0
                
            # Time spent: affected by difficulty, student speed trait, and some random variation
            time_spent = base_time + (speed_trait * 10.0) + np.random.normal(0.0, 5.0)
            time_spent = max(5.0, min(180.0, time_spent)) # Clip between 5 and 180 minutes
            
            # Score: affected by student ability, difficulty, and some random variation
            # Slower speed trait might slightly increase score (taking more time to think)
            score = base_score + (ability * 12.0) + (speed_trait * 2.0) + np.random.normal(0.0, 8.0)
            score = max(0.0, min(100.0, score)) # Clip between 0 and 100
            
            # Accuracy: highly correlated with score and difficulty
            # accuracy = score/100 adjusted with some small noise
            accuracy = (score / 100.0) + np.random.normal(0.0, 0.04)
            accuracy = max(0.0, min(1.0, accuracy))
            
            records.append({
                "student_id": student_id,
                "subject": subject,
                "difficulty": difficulty,
                "time_spent_mins": round(time_spent, 1),
                "score": round(score, 1),
                "accuracy": round(accuracy, 3)
            })
            
    df = pd.DataFrame(records)
    
    # Save directory
    os.makedirs("./data", exist_ok=True)
    output_path = "./data/personalized_recommendation_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"Successfully generated and saved dataset to {output_path}")
    print(f"Total rows: {len(df)}")
    print(f"Unique students: {df['student_id'].nunique()}")
    print(f"Unique subjects: {df['subject'].nunique()}")
    print("\nSample Data:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    generate_dataset()
