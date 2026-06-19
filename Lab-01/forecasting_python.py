import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_excel("YT_Channel_view_forecast_Data.xlsx")

# Example: use row number as X
X = [[i] for i in range(len(df))]
y = df.iloc[:, 1]  # second column

# Train model
model = LinearRegression()
model.fit(X, y)

# Predictions
pred = model.predict(X)

# R² score
r2 = model.score(X, y)
print("R² Score:", r2)

# Plot
plt.plot(y, label="Actual")
plt.plot(pred, label="Trend Line")
plt.legend()
plt.title("YouTube Channel Forecast")
plt.savefig("forecast_plot.png")
plt.show()

print("Equation:")
print("y =", model.coef_[0], "* x +", model.intercept_)
