import streamlit as st
import pandas as pd
import joblib
import time

# -----------------------------------------
# 1. Page Configuration & Custom CSS
# -----------------------------------------
st.set_page_config(page_title="Treatment Advisor", page_icon="💊", layout="centered")

st.markdown("""
    <style>
    /* Clean background */
    .stApp { background-color: #f0f4f8; }
    
    /* Big blue primary button */
    .stButton>button {
        width: 100%;
        background-color: #007BFF;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        padding: 12px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #0056b3; }
    
    /* Prescription Card Style */
    .rx-card {
        background-color: white;
        border-left: 8px solid #28a745; /* Green side border */
        padding: 25px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-top: 20px;
    }
    .rx-header { color: #28a745; font-size: 18px; font-weight: bold; text-transform: uppercase; letter-spacing: 1px;}
    .rx-medicine { color: #1e293b; font-size: 24px; font-weight: 600; margin-top: 10px; line-height: 1.4;}
    
    /* Warning Box */
    .medical-warning {
        font-size: 14px; color: #856404; background-color: #fff3cd; 
        padding: 15px; border-radius: 8px; margin-top: 30px; border: 1px solid #ffeeba;
    }
    </style>
""", unsafe_allow_html=True)


# -----------------------------------------
# 2. Load ML Components
# -----------------------------------------
@st.cache_resource
def load_components():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns

try:
    model, scaler, feature_columns = load_components()
except Exception as e:
    st.error("⚠️ Could not load the AI models. Please ensure .pkl files are in the folder.")
    st.stop()


# -----------------------------------------
# 3. Simple User Interface
# -----------------------------------------
st.title("💊 Smart Treatment Advisor")
st.write("Select your symptoms below to see the recommended medicine or treatment plan.")

# Clean symptom selector
selected_symptoms = st.multiselect(
    "What symptoms are you experiencing?", 
    options=feature_columns,
    placeholder="Type or click to select symptoms..."
)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------
# 4. Processing & Result Display
# -----------------------------------------
if st.button("Find Recommended Medicine"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom from the list above.")
    else:
        with st.spinner("Consulting the AI knowledge base..."):
            time.sleep(1) # Brief pause for UX
            
            # Prepare data exactly how the model expects it
            input_data = {col: 0 for col in feature_columns}
            for symptom in selected_symptoms:
                input_data[symptom] = 1
                
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            
            # Get the treatment prediction
            prediction = model.predict(input_scaled)
            recommended_treatment = prediction[0]
            
            # Display the elegant Prescription Card
            st.markdown(f"""
                <div class="rx-card">
                    <div class="rx-header">🩺 Recommended Action / Medication</div>
                    <div class="rx-medicine">{recommended_treatment}</div>
                </div>
            """, unsafe_allow_html=True)

# Medical Disclaimer at the bottom
st.markdown("""
    <div class="medical-warning">
        <b>🛑 Important Medical Disclaimer:</b><br>
        This application uses an AI model for educational and informational purposes only. It is <b>not</b> a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider before taking any medication.
    </div>
""", unsafe_allow_html=True)