# AI-Driven Marketing Lead Scoring

An end-to-end deep learning solution designed to identify "Hot Leads" with high conversion potential using the **Kaggle Lead Scoring** [dataset](https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset). This project implements a **Dual-Stream Neural Network** in PyTorch, utilizing **Entity Embeddings** for categorical features and numerical normalization for behavioral tracking.

## 📈 Project Performance

Based on the final training execution:

- **Best Validation ROC-AUC:** **0.8914**

- **Final Training Loss:** **0.3606**

- **Inference Confidence:** **93.20%** (Sample "Hot Lead" test)

- **Optimization:** The model reached peak discriminative power by Epoch 11, successfully ranking potential converters significantly higher than non-converters.

## 🛠️ Step-by-Step Methodology

### 1. Step 1 (Objective)

The objective was to transform raw CRM data into a prioritized sales queue. We framed this as a **Binary Classification** problem: predicting whether a lead will "Convert" (1) or "Not Convert" (0) based on their origin, source, and website behavior.

### 2. Step 2 (Data Engineering)

- **Variance Filtering:** Automatically pruned zero-variance or single-value columns that provide no predictive power to ensure model stability.
- **Entity Embedding Prep:** Categorical features (like _Lead Source_) were mapped to integers and prepared for high-dimensional vector representation.
- **Robust Normalization:** Behavioral metrics like _Total Time Spent on Website_ were scaled using `StandardScaler` to ensure uniform gradient descent.

- **Safe Serialization:** Utilized `joblib` for asset persistence to bypass common Python 3.13 string-mapping errors during deployment.

### 3. Step 3 (Architecture)

The architecture utilizes a **Dual-Stream DNN**:

- **Categorical Stream:** Individual **Embedding Layers** for each categorical feature to capture complex relationships between lead sources and conversion rates.
- **Numerical Stream:** Dense layers to process continuous behavioral metrics.
- **Classification Head:** A multi-layer perceptron (MLP) with **BCEWithLogitsLoss** for stable, high-precision probability output.

### 4. Step 4 (Evaluation)

- **ROC-AUC Metric:** Prioritized the Area Under the Receiver Operating Characteristic curve over simple accuracy to account for class imbalance in marketing data.

- **Safe-Transform Logic:** Implemented a robust inference fallback mechanism to handle "Unknown" or new categorical values without system crashes.

### 5. Step 5 (Interactive Deployment)

Created a **Streamlit Dashboard** that provides:

- **Real-Time Scoring:** Instant conversion probability for manual lead entry.
- **Priority Tiering:** Automatic classification into **Hot**, **Warm**, or **Cold** leads to guide sales team outreach.

## 📂 File Structure

- `data/`: Contains the `lead-scoring.csv`.
- `lead_processor.py`: Advanced cleaning, variance filtering, and `joblib` serialization logic.
- `model.py`: PyTorch architecture featuring dynamic Embedding layers.
- `main.py`: Orchestrator for the training loop and ROC-AUC validation.

- `inference.py`: Production-ready script with robust "Safe-Transform" logic.

- `app.py`: Interactive Streamlit dashboard for end-users.

## 🚀 Getting Started

1. **Clone the Repository:**

```bash
git clone https://github.com/devopspower/marketing-lead-scoring.git

```

2. **Install Dependencies:**

```bash
pip install -r requirements.txt

```

3. **Run the Training Pipeline:**

```bash
python main.py

```

4. **Launch the Dashboard:**

```bash
streamlit run app.py

```

## 💡 Business Insights

- **Behavioral Correlation:** The model identified that _Total Time Spent on Website_ and _SMS Activity_ are the strongest predictors of a "Hot Lead".

- **Sales Efficiency:** By focusing on leads with a >80% probability, sales teams can increase conversion rates while reducing time spent on uninterested prospects.
