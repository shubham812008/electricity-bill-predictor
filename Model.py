
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load daily electricity data
df = pd.read_csv("daily_electricity.csv")

# 2. Prepare columns
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Daily_Consumption_kWh"] = pd.to_numeric(
    df["Daily_Consumption_kWh"], errors="coerce"
)

df = df.dropna()
df = df.sort_values("Date")
df = df.drop_duplicates(subset="Date")
df = df.reset_index(drop=True)

# 3. Create previous-day features
for lag in range(1, 8):
    df[f"Previous_Day_{lag}"] = (
        df["Daily_Consumption_kWh"].shift(lag)
    )

df = df.dropna().reset_index(drop=True)

# 4. Select input features and target
features = [f"Previous_Day_{i}" for i in range(1, 8)]

X = df[features]
y = df["Daily_Consumption_kWh"]

# 5. Split chronologically: first 80% train, last 20% test
split = int(len(df) * 0.8)

if split < 1 or split >= len(df):
    raise ValueError("Not enough data to train and test.")

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# 6. Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Predict test data
predictions = model.predict(X_test)

# 8. Evaluate model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("Model Training Completed!")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"MAE: {mae:.3f} kWh")
print(f"RMSE: {rmse:.3f} kWh")
print(f"R2 Score: {r2:.3f}")


# 9. Save predictions
results = pd.DataFrame({
    "Date": df["Date"].iloc[split:].values,
    "Actual_kWh": y_test.values,
    "Predicted_kWh": predictions
})
results.to_csv("model_predictions.csv", index=False)

# 10. Save trained model
joblib.dump(model, "electricity_model.pkl")

# 11. Plot actual vs predicted consumption
plt.figure(figsize=(12, 5))
plt.plot(results["Date"], results["Actual_kWh"],
         label="Actual")
plt.plot(results["Date"], results["Predicted_kWh"],
         label="Predicted")
plt.xlabel("Date")
plt.ylabel("Daily Consumption (kWh)")
plt.title("Electricity Consumption: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.savefig("prediction_graph.png")
plt.show()

# 12. Predict the next day using the latest seven days
latest_seven = df["Daily_Consumption_kWh"].tail(7).values[::-1]
next_day_prediction = model.predict([latest_seven])[0]

print(f"Predicted next-day consumption: "
      f"{next_day_prediction:.2f} kWh")

print("\nFiles saved:")
print("electricity_model.pkl")
print("model_predictions.csv")
print("prediction_graph.png")
