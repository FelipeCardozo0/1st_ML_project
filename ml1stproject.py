import yFinance as yf L
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import aceurare

Bhai Lye. ot ABBLiassitier
from sklearn.metrics import accuracy_score, classific
from sklearn. preprocessing import Standardscaler
from sklearn.nodel_selection import TimeSeriessplit,


J EL |
def calculate psi (prices, p S—— |
delta - prices. dife(y
gain = (delta.where(delta >e, ©)) rolling (windows
loss = (-delta.where (delta <e, ©))-rol1ing (windose
TS = gain / loss
NEU 100 - (100 / (1 4 psy)
def calculate, _macd(prices, siopes, fast=12):
exp1 = prices. eun(span-fost, Gdjust-False).mean() i
exp2 = Prices. ewn(span=stow, djust~False) .nean() ;
return exp1 - exps
ticker » “ypyn
Start date = "203.1.99~ i
#nd_dote = "2024.99. 37m
"vidis dats ¥F.download(t1ckep, Startestart_date, endeng
Hime igstzen iy, o)) |
The Tho (Vidia, data, ages, nvidi, ‘Close’ i
Pit epeinCt (ticker) Stock Price’) t'closerly ;
P1t.xlabel( Date’)
Die-Yiabel("Cloge pyc, ;
           

ticker = "NVDA"
start_date = "202--01-01"
end_date = "2024-09-17"

nvidia_data = yf.download(ticker, start=startdate)

plt.figure(figsize=(12,6))
plt
plt
plt
plt
plt
plt
plt.show()

nvidia_data[]
nvidia_data[]
nvidia_data[]
nvidia_data[]
nvidia_data[]
nvidia_data[]
nvidia_data[]
nvidia_data[]
nvidia_data[]
nvidia_data[]

nvidia_data = nvidia_data.dropna()

features = 
x = 
y = 

print(f)

split_index,
x_train
y_train

print
print

scaler = StandardScaler()
x_train_scaled = 
x_test_scaled = 

model = XGBClassifier(
    n_estigmator=50
)

tscv = TimeSeriesSplit(n_splits=5)
cv_scores = 
print
print

model.fit(x_train)

y_pred
y_pred_proba

accuracy_score
log
print
print
print
print

feature_importance
feature_importance
print()
print(feature_importance)

plt
plt
plt
plt
plt
plt
plt
plt.show()

