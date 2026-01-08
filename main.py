import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from lead_processor import get_processed_data
from model import LeadScoringNet
import joblib

# --- Configuration ---
CONFIG = {
    'file_path': 'data/lead-scoring.csv',
    'batch_size': 64,
    'learning_rate': 0.001,
    'epochs': 15,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'model_save_path': 'lead_model.pth'
}

def run_training():
    print(f"--- Lead Scoring Training Pipeline ---")
    print(f"Target Device: {CONFIG['device']}")

    # 1. Ground Objectively: Load and Process Data
    # emb_dims maps each categorical feature to its optimal embedding dimension
    train_loader, val_loader, emb_dims, n_num = get_processed_data(CONFIG['file_path'], CONFIG['batch_size'])

    # 2. Analyze Logically: Initialize Tabular DNN
    model = LeadScoringNet(emb_dims, n_num).to(CONFIG['device'])
    
    # Using BCEWithLogitsLoss for numerical stability (handles sigmoid internally)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-5)

    # 3. Explore Systematically: Training Loop
    best_auc = 0.0
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0.0
        
        for x_cat, x_num, y in train_loader:
            x_cat, x_num, y = x_cat.to(CONFIG['device']), x_num.to(CONFIG['device']), y.to(CONFIG['device'])
            
            optimizer.zero_grad()
            logits = model(x_cat, x_num)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        # 4. Validate Rigorously: Performance Evaluation
        model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for x_cat, x_num, y in val_loader:
                logits = model(x_cat.to(CONFIG['device']), x_num.to(CONFIG['device']))
                # Sigmoid converts logits to [0, 1] probability
                probs = torch.sigmoid(logits)
                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(y.cpu().numpy())

        auc_score = roc_auc_score(val_targets, val_preds)
        avg_train_loss = train_loss / len(train_loader)

        print(f"Epoch [{epoch+1:02d}/{CONFIG['epochs']}] | Loss: {avg_train_loss:.4f} | ROC-AUC: {auc_score:.4f}")

        # Save the best model state based on validation performance
        if auc_score > best_auc:
            best_auc = auc_score
            torch.save(model.state_dict(), CONFIG['model_save_path'])

    print(f"\n--- Training Complete ---")
    print(f"Best ROC-AUC Achieved: {best_auc:.4f}")
    print(f"Model serialized to: {CONFIG['model_save_path']}")

if __name__ == "__main__":
    run_training()