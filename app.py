import streamlit as st
import torch
import joblib
import numpy as np
from model import LeadScoringNet

# --- Safe Transform Helper (Matches validated inference.py) ---
def safe_transform(encoder, value):
    """Safely retrieves integer index, bypassing scikit-learn internal mapping errors."""
    if encoder.classes_.size == 0:
        return 0
    val_str = str(value)
    if val_str not in encoder.classes_:
        # Fallback to 'Unknown' or the first available class
        val_str = 'Unknown' if 'Unknown' in encoder.classes_ else str(encoder.classes_[0])
    
    indices = np.where(encoder.classes_ == val_str)[0]
    return int(indices[0])

# --- Page Configuration ---
st.set_page_config(page_title="Lead Scorer AI", page_icon="🚀", layout="wide")
st.title("🎯 Marketing Lead Scoring Dashboard")
st.markdown("""
This AI model predicts the likelihood of a lead converting into a customer based on 
behavioral data and profile attributes.
""")

# --- Load Model & Processing Assets ---
@st.cache_resource
def load_scoring_assets():
    # Load the joblib assets created by lead_processor.py
    assets = joblib.load('processing_assets.joblib')
    encoders = assets['encoders']
    scaler = assets['scaler']
    cat_cols = assets['cat_cols']
    num_cols = assets['num_cols']
    
    # Reconstruct the architecture to match the saved weights
    emb_dims = [(len(encoders[col].classes_), min(50, (len(encoders[col].classes_) + 1) // 2)) 
                for col in cat_cols]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeadScoringNet(emb_dims, len(num_cols)).to(device)
    
    # Load the weights generated during the Best ROC-AUC epoch
    model.load_state_dict(torch.load('lead_model.pth', map_location=device))
    model.eval()
    
    return model, assets, device

try:
    model, assets, device = load_scoring_assets()

    # --- Sidebar for Input ---
    st.sidebar.header("User Input Parameters")
    
    user_input = {}
    
    # Numerical Inputs (Behavioral)
    st.sidebar.subheader("Behavioral Data")
    for col in assets['num_cols']:
        user_input[col] = st.sidebar.number_input(f"Enter {col}", value=0.0)

    # Categorical Inputs (Profile)
    st.sidebar.subheader("Lead Profile")
    for col in assets['cat_cols']:
        options = list(assets['encoders'][col].classes_)
        user_input[col] = st.sidebar.selectbox(f"Select {col}", options=options)

    # --- Main Display and Prediction ---
    if st.sidebar.button("Generate Lead Score"):
        # 1. Transform Categorical Inputs
        cat_values = [safe_transform(assets['encoders'][col], user_input[col]) for col in assets['cat_cols']]
        
        # 2. Transform Numerical Inputs
        num_values = [float(user_input[col]) for col in assets['num_cols']]
        num_scaled = assets['scaler'].transform([num_values])
        
        # 3. Predict
        cat_tensor = torch.tensor([cat_values], dtype=torch.long).to(device)
        num_tensor = torch.tensor(num_scaled, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = model(cat_tensor, num_tensor)
            probability = torch.sigmoid(logits).item()

        # --- Visualizing Results ---
        st.subheader("Prediction Results")
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.metric(label="Conversion Probability", value=f"{probability:.2%}")
            st.progress(probability)

        with col_res2:
            if probability >= 0.80:
                st.success("### 🔥 HOT LEAD\nPriority: **High**. Assign to senior sales rep immediately.")
            elif probability >= 0.40:
                st.warning("### ⚡ WARM LEAD\nPriority: **Medium**. Enrollment in automated nurture sequence recommended.")
            else:
                st.error("### ❄️ COLD LEAD\nPriority: **Low**. Keep in long-term awareness mailing list.")

    else:
        st.info("Adjust the parameters in the sidebar and click 'Generate Lead Score' to see results.")

except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.warning("Please ensure 'lead_model.pth' and 'processing_assets.joblib' are in the project folder.")