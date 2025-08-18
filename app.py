import streamlit as st
import pandas as pd
import joblib

# === Load artifacts ===
model = joblib.load('xgboost_stroke_model.pkl')
ohe = joblib.load('ohe_encoder.pkl')
features = joblib.load('model_features.pkl')  # your golden column order

st.set_page_config(page_title="Stroke Predictor", layout="centered")
st.title("Stroke Risk Prediction App")
st.markdown(
    """
    ## Stroke Risk Prediction App  
    Welcome to your **AI-powered health companion**.  
    This tool uses a machine learning model trained on patient health records to estimate the likelihood of a stroke,  
    based on key medical and lifestyle factors you provide.

    **How it works:**  
    - Input your details in the form below.  
    - Our model processes them against patterns it learned from real-world medical data.  
    - You’ll get a **risk probability** and a clear **conclusion**.

    ⚠️ *Note:* This app is for **educational and informational purposes only** and is **not** a substitute for professional medical advice.  
    Always consult a qualified healthcare provider for concerns about your health.
    """
)


# === Input form ===
with st.form(key='input_form'):
    gender = st.selectbox("Gender", ["Male", "Female"])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.selectbox(
        "Work Type",
        ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
    )
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    smoking_status = st.selectbox(
        "Smoking Status",
        ["formerly smoked", "never smoked", "smokes", "Unknown"]
    )

    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    avg_glucose = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, step=0.1)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

    submit = st.form_submit_button("🔍 Predict Stroke Risk")

# === Prediction logic ===
if submit:
    # Map yes/no to binary
    hypertension_val = 1 if hypertension == "Yes" else 0
    heart_disease_val = 1 if heart_disease == "Yes" else 0

    # Build categorical and numeric DataFrames
    cat_df = pd.DataFrame(
        [[gender, ever_married, work_type, residence_type, smoking_status]],
        columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    )
    num_df = pd.DataFrame(
        [[age, hypertension_val, heart_disease_val, avg_glucose, bmi]],
        columns=['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
    )

    # One-hot encode categoricals
    encoded_cats = ohe.transform(cat_df).toarray()
    encoded_df = pd.DataFrame(
        encoded_cats,
        columns=ohe.get_feature_names_out(cat_df.columns)
    )

    # Merge numeric + encoded categoricals
    final_df = pd.concat([num_df, encoded_df], axis=1)

    # Fill any missing expected features with zeros
    for col in features:
        if col not in final_df.columns:
            final_df[col] = 0

    # Reorder to match training exactly
    final_df = final_df[features]

    # Predict
    pred = model.predict(final_df)[0]
    prob = model.predict_proba(final_df)[0][1]

    # Display result
    if pred == 1:
        st.error(f"⚠️ High Stroke Risk — Probability: {prob:.2%}")
    else:
        st.success(f"✅ Low Stroke Risk — Probability: {prob:.2%}")

    # Display result
    if pred == 1:
        st.error(f"⚠️ High Stroke Risk — Probability: {prob:.2%}")
        st.write("🩺 **Conclusion:** The patient is likely to experience a stroke.")
    else:
        st.success(f"✅ Low Stroke Risk — Probability: {prob:.2%}")
        st.write("🩺 **Conclusion:** The patient is unlikely to experience a stroke.")
