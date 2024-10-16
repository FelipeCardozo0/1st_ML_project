import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, _classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

def calculate_rsi (prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta >0,0)).rolling(window=win)
    loss = (-delta.where(delta <0,0)).rol1ing(window=win)
    rs = gain / loss
    return 100 - (100 /(1+ rs))

def calculate_macd(prices, slow=26, fast=12):
    exp1 = prices.ewn(span=fast, adjust=False).mean();
    exp2 = prices.ewn(span=slow, adjust=False).mean();
    return exp1 - exp2

ticker = "NVDA"
start_date = "2020-01-01"
end_date = "2024-09-17"

nvda_data = yf.download(ticker, start=start_date, end=end_date)

plt.figure(figsize=(12,6))
plt.plot(nvidia_data.index, nvidia_data["Close"])
plt.title(f"{ticker} Stock Price")
plt.xlabel("Date")
plt.ylabel("Close price") 
plt.grid(True)
plt.show()

nvidia_data["Returns"] =  nvidia_data["Close"].pct_change()
nvidia_data["Target"] =  (nvidia_data["Returns"] > 0).astype(int)
nvidia_data["MA5"] = (nvidia_data)["Close"].rolling(window =  5).mean()
nvidia_data["MA20"] = (nvidia_data)["Close"].rolling(window =  20).mean()
nvidia_data["RSI"]  = calculate_rsi(nvidia_data["Close"],  window=14)
nvidia_data["MACD"] = calculate_macd(nvidia_dat["Close"])
nvidia_data["Volatility"] = (nvidia_data).rolling(window=20)
nvidia_data["Price_change"] = (nvidia_data).pct_change(5)
nvidia_data["Volume_Change"] = (nvidia_data)["Volume"].pct_change(5)

nvidia_data = nvidia_data.dropna()

features = ["MA5", "MA20", "RSI", "MACD",  "Volatility", "Price_change", "Volume_Change"]
x =nvidia_data[features]
y  = nvidia_data["Target"]

print(f"Total number of samples: {len(x)}") 

split_index = int(len(nvidia_data)* 0.8)
x_train,  x_test = x[:split_index], x[split_index:]
y_train,  y_test = y[:split_index], y[split_index:]

print(f"Number of training  samples: {len(x_train)}")
print(f"Number of test  samples: {len(x_train)}")

scaler = StardardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model  = XGBClassifier(
    n_estimators=50,
    learning_rate=0.01,
    max_depth=2,
    min_child_weight=10,
    subsample = 0.8,
    colsample_bytree = 0.8,
    gamma=2,
    reg_aplha=1,
    reg_lambda=1,
    random_state=42
)

tscv=TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(model,x_train_scaled, y_train)
print(f"Cross-validation scores:{cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.2f}")

model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)
y_pred_proba = model.predict_proba(x_test_scaled)

accuracy =  accuracy_score(y_test, y_pred)
log_loss_score = log_loss(y_test, y_pred_proba)
print(f"Test set accuracy: {accuracy:.2f}")
print(f"Log test: {log_loss_score:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

feature_importance = pd.DataFrame({"feature": features,  "importance": model.feature_importances_})
features_importance = features_importance.sort_values("Importance", ascending=False)
print("\n Feature Importance:")
print(features_importance)

plt.figure(figsize=(10,6))
plt.bar(feature_importance["feature"], feature_importance["importance"])
plt.title("Features Importance")
plt.xlabel("Features")
plt.ylabel("Impoortance") 
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

last_data = X.iloc[-1].values.reshape(1,-1)
last_data_scaled = scaler.transform(last_data)
tomorrow_prediction = model.predict(last_data_scaled)

print(f"\n Prediction for tomorrow: {"Up" if tomorrow_predcition}")
