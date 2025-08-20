# 💳 Credit Card Fraud Detection Dashboard  

An interactive **Streamlit dashboard** for detecting fraudulent transactions using multiple ML models.  

## 🚀 Features
- Upload your own dataset (`creditcard.csv`) or use preloaded models  
- Multiple algorithms: Logistic Regression, Random Forest, LightGBM  
- Side-by-side comparison of accuracy, precision, recall & F1-score  
- Real-time fraud prediction for new transactions  
- Visual performance metrics  

## 📂 Project Structure
├── fraud_detection.py # Streamlit App
├── train_models.py # Model Training Script
├── log_reg.pkl # Trained Logistic Regression Model
├── random_forest.pkl # Trained Random Forest Model
├── lgbm.pkl # Trained LightGBM Model
├── scaler.pkl # StandardScaler
└── README.md

csharp
Copy
Edit

## 📊 Dataset
This project uses the **[Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)**.  

⚠️ Due to GitHub file size limits, the dataset is **not included in this repo**.  
Please download it from Kaggle and place it in the project root as:  
