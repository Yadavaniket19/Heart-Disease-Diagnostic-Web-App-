# ❤️ Heart Disease Diagnostic Web App

An end-to-end Machine Learning web application designed to predict and diagnose the likelihood of heart disease based on patient health symptoms, clinical parameters, and medical test indicators.

---

## 📌 Project Overview
Cardiovascular diseases (CVDs) are the leading cause of mortality globally. Early detection and risk assessment can significantly improve clinical outcomes. 

This project analyzes patient diagnostic data, trains classification machine learning models to identify risk patterns, and exports production-ready pipelines (`scaler.pkl`, `feature_columns.pkl`, `best_model.pkl`) to power an interactive diagnostic tool.

---

## 📊 Dataset & Features
The model is trained on a comprehensive clinical symptoms dataset, utilizing key diagnostic indicators:

- **Demographics:** Age, Gender
- **Vitals & Clinical Measures:** Resting Blood Pressure, Serum Cholesterol, Fasting Blood Sugar, Resting ECG results
- **Cardiac Function:** Maximum Heart Rate Achieved, Exercise Induced Angina, ST Depression (Oldpeak), Slope of Peak Exercise ST Segment
- **Target:** Diagnosis result (Presence or Absence of Heart Disease risk)

---

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn (Logistic Regression, Decision Trees, Random Forest, SVC)
- **Model Serialization:** Joblib
- **Web App / Deployment:** Streamlit

---

## 🚀 Workflow & Project Pipeline

1. **Exploratory Data Analysis (EDA):**
   - Assessing distributions, missing values, and medical correlations.
   - Identifying critical indicators like cholesterol, resting BP, and ST depression levels.
2. **Data Preprocessing & Feature Engineering:**
   - Encoding categorical features using `LabelEncoder`.
   - Feature normalization and standardization with `StandardScaler`.
   - Feature alignment saved as `feature_columns.pkl`.
3. **Model Training & Evaluation:**
   - Benchmarking multiple classifiers (Logistic Regression, Random Forest, Support Vector Classifier).
   - Evaluating models using Accuracy, Precision, Recall, Confusion Matrix, and ROC-AUC.
4. **Export & Deployment:**
   - Serializing the top-performing model as `best_model.pkl` alongside `scaler.pkl` for real-time inference.

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Yadavaniket19/Heart-Disease-Diagnostic-Web-App-.git](https://github.com/Yadavaniket19/Heart-Disease-Diagnostic-Web-App-.git)
   cd Heart-Disease-Diagnostic-Web-App-
   ```
### 2.Create and activate a virtual environment:
```bash
python -m venv env
# On Windows:
env\Scripts\activate
# On macOS/Linux:
source env/bin/activate
```

### 3.Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit
```

### 4.Run the Streamlit Web Application:
```bash
streamlit run app.py
```

### 📁 Repository Structure
```
├── app.py                                            # Streamlit web application
├── best_model.pkl                                    # Trained machine learning model
├── scaler.pkl                                        # Fitted StandardScaler object
├── feature_columns.pkl                               # Extracted feature column names
├── Health Symptoms and Disease Prediction Dataset.ipynb # Research & EDA notebook
└── README.md                                         # Project documentation
```

---

## 👤 Author
- **Aniket Yadav**
- GitHub: [@Yadavaniket19](https://github.com/Yadavaniket19)

---
⭐️ If you found this project helpful, please consider giving it a star!
