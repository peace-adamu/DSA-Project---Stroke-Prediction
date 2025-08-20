# Stroke Prediction App 🧠

[![Streamlit App](https://img.shields.io/badge/Live-App-green)](https://dsa-project---stroke-prediction.streamlit.app)

A machine learning app that predicts stroke risk based on user input...

# Digital SkillUp Africa Stroke Risk Prediction App — Project Documentation
<img width="2962" height="1030" alt="DSA Logo C1" src="https://github.com/user-attachments/assets/f39d7289-0d63-47e5-857a-6adcdd64ab08" />


##  Table of Contents
- [Acknowledgments](#acknowledgments)
-  [Project Preview](#project-preview.)
- [Project Objective](#project-objective)
- [Project Significance](#project-significance)
- [Methodology](#methodology)
- [Prerequisites](#Prerequisities)
- [Discussion of Result](#discussion-of-result)
- [Conclusion](#conclusion)

## Acknowledgments
I would like to express my deepest gratitude to the following individuals and organizations for their support and guidance throughout this project:
First and foremost, I acknowledge the Digital SkillUp African organization for providing the platform and resources necessary to develop my skills in Machine learning and Artifical Intelligent. Their commitment to empowering Africa in technology is truly inspiring.
I would also like to extend my sincere appreciation to my tutors, Mr. Oluwole Olajide and Mr. Blessing, their expertise, patience, and dedication were instrumental in my success. Also their guidances and feedbacks were invaluable, and I am grateful for the opportunity to learn from them.


## Project Preview
Stroke is one of the leading causes of disability and death worldwide. Early detection and risk assessment can be life-saving, yet access to diagnostic tools and clinical evaluations is often limited. This project explores how artificial intelligence (AI) can bridge that gap by developing a stroke risk prediction system based on basic health metrics and demographic information.
The goal was to build a lightweight, accessible, and user-friendly tool powered by machine learning to assist in early stroke risk screening as my DSA CapStone project.
The Stroke Risk Prediction App is an interactive AI-powered health assistant built with Streamlit, leveraging machine learning to estimate the probability of stroke in a patient based on medical and lifestyle inputs.
Users provide data such as age, BMI, glucose level, heart disease status, hypertension history, and lifestyle habits. The app processes these inputs using a pre-trained XGBoost model, outputs:
Stroke risk probability (percentage), a clear conclusion (“likely” / “unlikely” to be experiencing stroke), a user-friendly interface for both non-technical and medical audiences, and this ensures fast, consistent, and reliable health insights for educational and preventive purposes.


## Objectives:
1. To gather patient information from medical imaging data and health information systems, such as demographics, medical history, and vital signs.
2. To make sure the gathered data is appropriate for machine learning models by pre-processing it.
3. In order to forecast the risk of developing hypertension or stroke and identify these problems early, machine learning models are being developed utilizing a variety of techniques, such as logistic regression, random forest, and support vector machines.
4. To evaluate the effectiveness of machine learning models built with a variety of metrics, including area under the curve (AUC), sensitivity, specificity, and precision.
5. To evaluate the effectiveness of the created machine learning models against more conventional approaches to identifying and treating stroke and hypertension, such as taking blood pressure readings by hand and employing imaging technologies.
6. To carry out further analysis, such as statistical analysis, cross-validation, and parameter adjustment, in order to enhance the generalization and robustness of the model.
7. To incorporate Explainable AI methods for better model interpretability and clinical transparency, such as SHAP (SHapley Additive Explanations) and LIME (Local Interpretable Model-agnostic Explanations)..
8. To determine any restrictions or difficulties in creating and applying machine learning models for patient hypertension and stroke detection and prediction.
9. To offer suggestions for next studies and the development of machine learning models for use in medical settings.


## Project Significance
Health awareness: Enables individuals to assess their risk factor awareness in seconds.

Educational tool: Helps students and professionals understand the real-world application of machine learning in healthcare.

Preventive insight: Though not a diagnostic tool, it can encourage timely medical consultation.

Automation in healthcare: Shows how AI can supplement professional workflows for faster screening
3. Dataset Overview


## Methodology
1: Data Acquisition: Kaggle Stroke Prediction Dataset - 
- Total Records: 5,110
- Target Variable: stroke (0 = No Stroke, 1 = Stroke)
- Key Features: Age, BMI, Average Glucose Level, Hypertension (Yes/No), Heart Disease (Yes/No), Gender, Work Type, Marital Status, Residence Type, Smoking Status

##### Note: The dataset was originally imbalanced, with a significantly lower number of stroke cases compared to non-stroke cases.

2: Handled missing values, normalized formats, and encoded categorical variables.
3: Data Preprocessing
- Handled missing values in the bmi field using median imputation.
- One-hot encoded categorical variables (gender, work_type, etc.) using OneHotEncoder with drop='first'.
- Used SMOTE (Synthetic Minority Oversampling Technique) to handle class imbalance.
- Scaled continuous features like age, avg_glucose_level, and bmi using StandardScaler.
- Split dataset into 80% training and 20% testing using a stratified split to preserve class distribution.
4: Model Development
- Multiple machine learning algorithms were evaluated, including:
- Logistic Regression
- Decision Tree
- XGBoost

5:  Evaluation Metrics
- Performance was measured using accuracy, precision, recall, and F1-score.
- This tight generalization gap indicates minimal overfitting and strong model stability.
6: Model Interpretation
- To ensure explainability:
- SHAP (SHapley Additive exPlanations) was used to visualize global and individual feature importance.
- LIME (Local Interpretable Model-agnostic Explanations) was used to explain predictions for single cases.
7: Web App Deployment with Streamlit


## Prerequisites
1. Before running the app locally, ensure:
2. Python 3.8 or newer
3. Installed libraries:
```
pip install streamlit pandas numpy scikit-learn xgboost joblib
```
4. Files in your working directory:
  - stroke_app.py (main app)
  - xgboost_stroke_model.pkl (trained ML model)
  - ohe_encoder.pkl (one-hot encoder)
  - model_features.pkl (list of features in training order)


## Discussion of result
Table 1.0: Cross-Validation Performance of ML Models for Stroke Prediction
| Model                   | Train Accuracy | Train F1  | Train Precision | Train Recall | Test Accuracy | Test F1  | Test Precision | Test Recall |
|-------------------------|---------------:|----------:|----------------:|-------------:|--------------:|---------:|---------------:|------------:|
| Logistic Regression     | 0.7940         | 0.8017    | 0.7710          | 0.8350       | 0.7906        | 0.8008   | 0.7710         | 0.8330      |
| Decision Tree           | 1.0000         | 1.0000    | 1.0000          | 1.0000       | 0.9434        | 0.9446   | 0.9351         | 0.9542      |
| Random Forest           | 1.0000         | 1.0000    | 1.0000          | 1.0000       | 0.9727        | 0.9726   | 0.9884         | 0.9572      |
| K-Nearest Neighbors     | 0.9141         | 0.9196    | 0.8626          | 0.9845       | 0.8981        | 0.9074   | 0.8391         | 0.9878      |
| Support Vector Machine  | 0.7707         | 0.7844    | 0.7386          | 0.8363       | 0.7623        | 0.7764   | 0.7399         | 0.8167      |
| Gradient Boosting       | 0.9680         | 0.9671    | 0.9927          | 0.9428       | 0.9630        | 0.9624   | 0.9872         | 0.9389      |
| XGBoost                 | 0.9990         | 0.9990    | 1.0000          | 0.9979       | 0.9702        | 0.9701   | 0.9812         | 0.9593      |

The findings establish a preliminary benchmark for choosing the most appropriate model by taking into account not just statistical performance but also interpretability and robustness. Conducting further analysis with SHAP and LIME interpretation methods for these models will aid in revealing how each model makes its predictions and the trustworthiness of these results in practical situations.

As displayed in Table 1.0, the Random Forest, Decision Tree, and XGBoost models all achieved flawless classification performance, reaching 100% across all assessed metrics. This outcome suggests either an outstanding model fit or possible data leakage or overfitting, which necessitates careful scrutiny.
On the other hand, the XG Boost, while not perfect, achieved very high scores across all metrics, with:
| Metric     | Train Score | Test Score | Gap    |
|------------|------------:|-----------:|-------:|
| Accuracy   | 0.9990      | 0.9676     | 0.0317 |
| Precision  | 1.0000      | 0.9782     | 0.0218 |
| Recall     | 0.9979      | 0.9564     | 0.0422 |
| F1 Score   | 0.9990      | 0.9671     | 0.0321 | 

The XG Boost model showed strong generalization capabilities, exhibiting a high recall rate that suggests few false negatives, an especially important factor in medical screening tasks like predicting hypertension. These results are consistent with earlier research that highlights GBM’s ability to capture intricate, non-linear relationships in structured data with significant predictive accuracy. - Final Selection: The XGBoost Classifier outperformed others in terms of accuracy, recall, and generalization. It was optimized with: L1/L2 regularization, Reduced max depth, Cross-validation with 10 Stratified folds. 

To ensure explainability:
- SHAP (SHapley Additive exPlanations) was used to visualize global and individual feature importance.
- LIME (Local Interpretable Model-agnostic Explanations) was used to explain predictions for single cases.
    - Top Influential Features:
    - Average Glucose Level
    - BMI
    - Heart Disease
    - Age
    - Smoking Status
