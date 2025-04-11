pip install yfinance matplotlib seaborn scikit-learn tensorflow


import yfinance as yf
import pandas as pd

#fetching data from aspecific stock(eg: tesla-TSLA)

ticker = "TSLA"
data = yf.download(ticker, start='2015-01-01', end='2025-01-01')

#save to csv

data.to_csv(f'{ticker}_stock_data.csv')

#display the data
print(data.head())

#printing the shape of data
print("shape of data: ", data.shape)

print("\ninfo about data:\n ", data.info())

print("\nhead data: \n", data.head())  #to get the data in proper form we have used the \n

print("\nteil data:\n ", data.tail())


print("\nstatical summary: \n", data.describe())

print(data.isnull().sum())

print("duplicated values: ",data.duplicated().sum())

data.index = pd.to_datetime(data.index)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
sns.boxplot(data['Close'])
plt.title('Closing Price Distribution')
plt.legend()
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(14,6))
plt.plot(data['Close'], label=f'{ticker} closing price', color='blue')
plt.title(f'{ticker} stock price over time')
plt.xlabel('data')
plt.ylabel('price')
plt.legend() #upper right corner comes from this
plt.show()


import seaborn as sns

# Correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

plt.figure(figsize=(14,6))
plt.plot(data["Volume"], label=f'{ticker} volume',color="orange")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.title(f'{ticker}treding volume over time')
plt.legend()
plt.show()

#7 days moving average
data['MA_7'] = data['Close'].rolling(window=7).mean()

#30 days moving average
data['MA_30'] = data['Close'].rolling(window=30).mean()

#plotting the moving averages
plt.figure(figsize=(14,7))
plt.plot(data['Close'], label='Close prise', color='blue')
plt.plot(data['MA_7'], label='7-day MA', color='orange')
plt.plot(data['MA_30'], label='30-day MA', color ='green')
plt.title(f'{ticker} stock price over time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

#daily returns
data['Daily_Returns'] = data['Close'].pct_change()

#plot daily returns
plt.figure(figsize=(14,6))
plt.plot(data['Daily_Returns'], label='Daily Returns', color='purple')
plt.title(f'{ticker} daily returns')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()
plt.show()

def calculate_rsi(data, window=14):
   delta = data['Close'].diff()
   gain = delta.where(delta > 0, 0)
   loss = -delta.where(delta <0, 0)

   avg_gain = gain.rolling(window=window).mean()
   avg_loss = loss.rolling(window=window).mean()

   rs = avg_gain / avg_loss
   rsi = 100 - (100/ (1 + rs))

   return rsi

data['RSI'] = calculate_rsi(data)

   #plot rsi
plt.figure(figsize=(14,7))
plt.plot(data['RSI'], label='RSI', color='red')
plt.axhline(70, linestyle='--', alpha=0.5, color='grey')#overbought level
plt.axhline(30, linestyle='--', alpha=0.5, color='grey')#oversold level
plt.title(f'{ticker} RSI')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend()
plt.show()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data['close_scaled'] = scaler.fit_transform(data[['Close']])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data['Close_Standardized'] = scaler.fit_transform(data[['Close']])

features = ['MA_7', 'MA_30', 'Daily_Returns', 'RSI', 'Volume']
target = 'Close'

X = data[features]
y = data[target]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

import numpy as np

# Fill NaNs with median values
X_train_scaled = np.nan_to_num(X_train_scaled, nan=np.nanmedian(X_train_scaled))
X_test_scaled = np.nan_to_num(X_test_scaled, nan=np.nanmedian(X_test_scaled))
y_train = np.nan_to_num(y_train, nan=np.nanmedian(y_train))


from sklearn.linear_model import LinearRegression

# Initialize and train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict stock prices
y_pred = model.predict(X_test_scaled)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

# Print results
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R² Score: {r2}')


plt.figure(figsize=(14,7))
plt.plot(y_test.values, label='Actual Prices', color='blue')
plt.plot(y_pred, label='Predicted Prices', color='red', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Actual vs Predicted Stock Prices')
plt.legend()
plt.show()

# Importing necessary libraries
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Define features and target
features = ['MA_7', 'MA_30', 'Daily_Returns', 'RSI', 'Volume']
target = 'Close'

# Extract features and labels
X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fill missing values if any
X_train.fillna(X_train.median(), inplace=True)
X_test.fillna(X_test.median(), inplace=True)
y_train.fillna(y_train.median(), inplace=True)
y_test.fillna(y_test.median(), inplace=True)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the SVR model
svr_model = SVR(kernel='rbf')  # You can also try 'linear' or 'poly'
svr_model.fit(X_train_scaled, y_train)

# Predicting
y_pred_svr = svr_model.predict(X_test_scaled)

# Evaluate the model
mae_svr = mean_absolute_error(y_test, y_pred_svr)
mse_svr = mean_squared_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mse_svr)
r2_svr = r2_score(y_test, y_pred_svr)

print(f'SVR Model - MAE: {mae_svr}, MSE: {mse_svr}, RMSE: {rmse_svr}, R² Score: {r2_svr}')

# Plotting predictions
plt.figure(figsize=(14,7))
plt.plot(y_test.values, label='Actual Prices', color='blue')
plt.plot(y_pred_svr, label='Predicted Prices (SVR)', color='green', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Actual vs Predicted Stock Prices - SVR')
plt.legend()
plt.show()


pip install xgboost

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb_model.fit(X_train_scaled, y_train)


y_pred_xgb = xgb_model.predict(X_test_scaled)

mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = mse_xgb ** 0.5
r2_xgb = r2_score(y_test, y_pred_xgb)

print("XGBoost Model:")
print(f"MAE: {mae_xgb}")
print(f"MSE: {mse_xgb}")
print(f"RMSE: {rmse_xgb}")
print(f"R² Score: {r2_xgb}")

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
plt.plot(y_test.values, label='Actual Prices', color='blue')
plt.plot(y_pred_xgb, label='Predicted Prices (XGBoost)', color='orange')
plt.title(f'Actual vs Predicted Stock Prices ({ticker}) - XGBoost')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()

import joblib
joblib.dump(xgb_model, 'xgboost_stock_model.pkl')

loaded_model = joblib.load('xgboost_stock_model.pkl')

from xgboost import plot_importance
plot_importance(xgb_model)
plt.show()