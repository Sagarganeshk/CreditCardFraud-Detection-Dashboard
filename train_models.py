import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import time

# Load dataset
df = pd.read_csv("creditcard.csv")

# Option 1: Downsample majority class (Non-fraud) for faster training
fraud_df = df[df["Class"] == 1]
non_fraud_df = df[df["Class"] == 0].sample(n=len(fraud_df) * 5, random_state=42)  # 1:5 ratio
df_balanced = pd.concat([fraud_df, non_fraud_df]).sample(frac=1, random_state=42)

X = df_balanced.drop("Class", axis=1)
y = df_balanced["Class"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Track training times
train_times = {}

# Logistic Regression
start = time.time()
log_reg = LogisticRegression(max_iter=500, class_weight="balanced")
log_reg.fit(X_train, y_train)
train_times["Logistic Regression"] = round(time.time() - start, 2)

# Random Forest
start = time.time()
rf = RandomForestClassifier(
    n_estimators=50,  # reduce trees for speed
    max_depth=10,     # limit depth
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
train_times["Random Forest"] = round(time.time() - start, 2)

# LightGBM
start = time.time()
lgbm = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    scale_pos_weight=10,  # handle imbalance
    n_jobs=-1,
    random_state=42
)
lgbm.fit(X_train, y_train)
train_times["LightGBM"] = round(time.time() - start, 2)

# Save models + scaler
joblib.dump(log_reg, "log_reg.pkl")
joblib.dump(rf, "random_forest.pkl")
joblib.dump(lgbm, "lightgbm.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Models trained and saved successfully!")
print("\n⏱ Training Times (seconds):")
for k, v in train_times.items():
    print(f"{k}: {v}s")
