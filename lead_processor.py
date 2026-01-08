import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

class LeadDataset(Dataset):
    """
    Standard Dataset for dual-stream Tabular data.
    """
    def __init__(self, cat_data, num_data, labels=None):
        self.cat_data = torch.tensor(cat_data, dtype=torch.long)
        self.num_data = torch.tensor(num_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

    def __len__(self):
        return len(self.num_data)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.cat_data[idx], self.num_data[idx], self.labels[idx]
        return self.cat_data[idx], self.num_data[idx]

def get_processed_data(file_path, batch_size=64):
    """
    Cleans data and filters out zero-variance columns to prevent IndexError during inference.
    """
    # 1. Load and Replace 'Select'
    df = pd.read_csv(file_path)
    df = df.replace('Select', np.nan)
    
    # 2. Systematic Drop of high-null and redundant ID columns
    cols_to_drop = [
        'Prospect ID', 'Lead Number', 'How did you hear about X Education', 
        'Lead Profile', 'Lead Quality', 'Asymmetrique Activity Index', 
        'Asymmetrique Profile Index', 'Asymmetrique Activity Score', 
        'Asymmetrique Profile Score', 'City', 'Specialization', 'Tags'
    ]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    
    target = 'Converted'
    num_cols = ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']
    
    # 3. Categorical Stream with Variance Filtering
    # We only encode columns that have actual information (more than 1 unique value)
    potential_cat = [c for c in df.columns if c not in num_cols + [target]]
    cat_cols = []
    encoders = {}
    
    for col in potential_cat:
        # Impute and cast to string to prevent mixed-type errors in Scikit-Learn
        df[col] = df[col].fillna('Unknown').astype(str)
        
        unique_vals = df[col].unique()
        if len(unique_vals) > 1:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
            cat_cols.append(col)
        else:
            # Dropping columns with zero variance as they break the model.py logic
            df.drop(columns=[col], inplace=True)
    
    # 4. Numerical Stream Imputation & Scaling
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
        
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # 5. Act Precisely: Save assets using Joblib
    assets = {
        'encoders': encoders, 
        'scaler': scaler, 
        'cat_cols': cat_cols, 
        'num_cols': num_cols
    }
    joblib.dump(assets, 'processing_assets.joblib')
    
    # 6. Train/Val Split
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    # Embedding Dimensions: [Vocabulary Size, Embedding Vector Size]
    emb_dims = [(len(encoders[col].classes_), min(50, (len(encoders[col].classes_) + 1) // 2)) 
                for col in cat_cols]
    
    # Create Loaders
    train_ds = LeadDataset(train_df[cat_cols].values, train_df[num_cols].values, train_df[target].values)
    val_ds = LeadDataset(val_df[cat_cols].values, val_df[num_cols].values, val_df[target].values)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, emb_dims, len(num_cols)