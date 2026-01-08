import torch
import joblib
import numpy as np
from model import LeadScoringNet

def safe_transform(encoder, value):
    """
    Safely retrieves the integer index for a categorical value.
    Bypasses scikit-learn's internal _map_to_integer to avoid indexing errors.
    """
    # 1. Handle empty encoders (safety for zero-variance columns)
    if encoder.classes_.size == 0:
        return 0
        
    val_str = str(value)
    
    # 2. Check if the value exists in trained classes
    # If not, fall back to 'Unknown'. If 'Unknown' isn't there, take the first class.
    if val_str not in encoder.classes_:
        if 'Unknown' in encoder.classes_:
            val_str = 'Unknown'
        else:
            val_str = str(encoder.classes_[0])
    
    # 3. Manual index lookup: stable across all Python 3.x versions
    indices = np.where(encoder.classes_ == val_str)[0]
    return int(indices[0])

def predict_lead_quality(lead_data, model_path='lead_model.pth', assets_path='processing_assets.joblib'):
    """
    Loads model and assets to score a single lead profile.
    """
    # Load transformation assets
    try:
        assets = joblib.load(assets_path)
    except FileNotFoundError:
        raise FileNotFoundError("Assets not found. Ensure you ran main.py successfully.")

    encoders = assets['encoders']
    scaler = assets['scaler']
    cat_cols = assets['cat_cols']
    num_cols = assets['num_cols']
    
    # Reconstruct architecture based on the processed categorical features
    emb_dims = [(len(encoders[col].classes_), min(50, (len(encoders[col].classes_) + 1) // 2)) 
                for col in cat_cols]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeadScoringNet(emb_dims, len(num_cols)).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Process Categorical Stream
    cat_values = []
    for col in cat_cols:
        val = lead_data.get(col, 'Unknown')
        cat_values.append(safe_transform(encoders[col], val))
    
    # Process Numerical Stream
    num_values = []
    for col in num_cols:
        num_values.append(float(lead_data.get(col, 0)))
    
    num_scaled = scaler.transform([num_values])

    # Convert to Tensors and Predict
    cat_tensor = torch.tensor([cat_values], dtype=torch.long).to(device)
    num_tensor = torch.tensor(num_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(cat_tensor, num_tensor)
        score = torch.sigmoid(logits).item()

    return score

if __name__ == "__main__":
    # Test lead dictionary with core features
    test_lead = {
        'Lead Origin': 'Landing Page Submission',
        'Lead Source': 'Google',
        'Do Not Email': 'No',
        'Do Not Call': 'No',
        'TotalVisits': 5.0,
        'Total Time Spent on Website': 1100.0,
        'Page Views Per Visit': 3.5,
        'Last Activity': 'SMS Sent',
        'What is your current occupation': 'Working Professional',
        'Last Notable Activity': 'SMS Sent'
    }

    try:
        probability = predict_lead_quality(test_lead)
        print(f"\n--- Lead Scoring Result ---")
        print(f"Conversion Probability: {probability:.2%}")
        
        if probability > 0.75:
            print("Status: 🔥 HOT LEAD")
        elif probability > 0.40:
            print("Status: ⚡ WARM LEAD")
        else:
            print("Status: ❄️ COLD LEAD")
            
    except Exception as e:
        print(f"Inference failed: {e}")