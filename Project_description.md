# Project Description: EduGraphAI 🎓

EduGraphAI is a multimodal educational intelligence platform designed to maximize student success through AI-guided personalized learning. Below is a detailed description of each core feature of the platform.

---

## 1. Document Library 📁
The **Document Library** serves as the central file storage and ingestion hub for the platform.
* **Multimodal Uploads**: Supports PDF textbooks, PowerPoint lecture notes, plain text summaries, and PNG/JPG images of diagrams or homework sheets.
* **Intelligent Ingestion Pipeline**: When a document is uploaded, the FastAPI backend processes it through a multimodal pipeline:
  * Text content is extracted, cleaned, and structured into semantic chunks.
  * Mathematical equations are identified and translated into valid LaTeX syntax.
  * Embedded diagrams, flowcharts, and figures are automatically isolated and cataloged with auto-generated captions.
* **Vector Store Ingestion**: Generates high-dimensional vector embeddings for all extracted chunks and stores them in **ChromaDB** for semantic search.
* **Knowledge Graph Extraction**: Core concepts and their inter-dependencies are extracted from the text and saved to the database.

---

## 2. Ask EduGraphAI 💬
The **Ask EduGraphAI** feature is a context-aware Retrieval-Augmented Generation (RAG) chat engine.
* **Targeted Answers**: Students can ask questions directly related to their uploaded textbooks and lectures.
* **Source Citations**: To prevent AI hallucinations and maintain academic integrity, every answer includes a "Sources" tab showing the exact passages, page numbers, and filenames used to synthesize the response.
* **LaTeX Formula Integration**: Automatically pulls relevant formulas from the documents, presenting them in LaTeX formatting with comprehensive variable explanations.
* **Diagram Extraction**: Surfaces figures and diagrams from the textbook that visually support the written explanation.
* **Confidence Scoring**: Shows a confidence percentage badge reflecting the relevance score of the retrieved source documents.

---

## 3. Knowledge Graph 🕸️
The **Knowledge Graph** visualizes concepts and their dependencies as an interactive web.
* **Interactive Visualization**: Renders an interactive 2D physics-based network using Pyvis.
* **Difficulty Mapping**: Concept nodes are color-coded based on difficulty level (Green for Beginner, Yellow for Intermediate, and Red for Advanced). Hovering over a node displays its summary description.
* **Learning Path Finder**: Users can input a target concept (e.g. "Electromagnetic Induction"), and the system traverses the graph (Neo4j or local SQLite) to return a sequential, step-by-step prerequisite learning path.

---

## 4. Handwriting Evaluator ✍️
The **Handwriting Evaluator** helps students evaluate their step-by-step solutions to physics, chemistry, or mathematics problems.
* **Image OCR & Analysis**: Ingests images of handwritten worksheets, extracts steps using Gemini Vision, and compares them against optional model answers.
* **Grading Schema**: Rates solutions out of 10 points.
* **Error Classification**: Segments errors into three tabs:
  * *Conceptual Errors*: Mistakes in physics laws, theorem applications, or qualitative reasoning.
  * *Calculation Errors*: Math mistakes, sign errors, or arithmetic failures.
  * *Formula Errors*: Wrong formulas used or variables plugged in incorrectly.
* **Actionable Feedback**: Generates detailed, step-by-step hints and advice on how to fix the errors.

---

## 5. Exam Intelligence 📊
**Exam Intelligence** allows students to upload past exam papers (PDFs) to prepare strategically.
* **Priority Topic Identification**: Analyzes questions to determine high-priority topics ranked by frequency.
* **Topic Frequency Charts**: Renders interactive bar charts showing topic prevalence.
* **Derivation Finder**: Catalogues all required mathematical or physical derivations identified in the test paper.
* **Numerical Inventory**: Extracts and groups the kinds of calculation problems asked in the test, helping students focus on key problem-solving patterns.

---

## 6. Student Analytics & Predictor 📈
**Student Analytics** acts as the central dashboard tracking performance and guiding study paths using machine learning.
* **Quiz Logger**: Logs scores, max marks, question difficulty, and time spent on study topics.
* **Performance Dashboard**: Renders interactive Plotly charts of score trends, topic breakdown averages, and recent quiz scores.
* **Ensemble Score Predictor**: Runs a machine learning ensemble of XGBoost, Gradient Boosting, and Random Forest models to predict final exam scores.
* **Pass Probability & Risk Level**: Displays a calibrated pass probability along with risk levels (Low, Medium, High) and lists weak topics causing student risk.
* **Personalized Recommendations**: Analyzes the student's profile via Random Forest + XGBoost models to generate a prioritized recommended study order, weak topics list, and revision checklist.

### 📊 Machine Learning Model Details & Metrics

#### Model A: Personalized Learning Recommendation System
* **Purpose**: Recommends a prioritized study path and identifies conceptual vulnerabilities.
* **Algorithm**: Ensemble of **Random Forest Regressor** and **XGBoost Regressor** (averaged predictions).
* **Feature Space**: Student profile average score, average time spent per quiz, average accuracy, subject code, and quiz difficulty.
* **Training Dataset**: Trained on 100,000 synthetic student interaction attempts.
* **Evaluation Metrics**:
  | Model | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | R² Score |
  |---|---|---|---|
  | Random Forest Regressor | **6.44** | **8.07** | **0.787** |
  | XGBoost Regressor | **6.36** | **7.98** | **0.791** |
  | **Ensemble (Average)** | **6.37** | **8.00** | **0.791** |

#### Model B: Student Performance Prediction System
* **Purpose**: Forecasts final exam scores, predicts pass probabilities, and assesses student failure risk.
* **Algorithm**: Ensemble of **XGBoost Regressor**, **Gradient Boosting Regressor**, and **Random Forest Regressor** for score forecasting, paired with an **XGBoost Classifier** for pass/fail classification.
* **Feature Space**: Quiz performance averages, consistency score, topic breadth, attempt difficulties, and score trend.
* **Training Dataset**: Original Portuguese/Math grades dataset augmented to **7,764 records** using Gaussian perturbations to oversample tail performance distributions (defining a clear pass/fail boundary).
* **Evaluation Metrics**:
  | Metric | Value |
  |---|---|
  | **Ensemble MAE** (Regressor) | **6.09%** |
  | **Ensemble RMSE** (Regressor) | **8.35%** |
  | **Classifier Accuracy** | **93.74%** |
  | **Classifier Precision** | **94.38%** |
  | **Classifier Recall** | **96.92%** |
  | **Classifier F1-Score** | **95.63%** |
  | **Classifier Calibration** | Sigmoid score-based calibration applied |
