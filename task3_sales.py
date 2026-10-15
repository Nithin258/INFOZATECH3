"""Task 3 - Sales Prediction using Advertising Data (InfozaTech Data Science Internship).

Usage: python task3_sales.py [path/to/Advertising.csv]
Expected columns: TV, Radio, Newspaper, Sales (default: data/Advertising.csv).
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load
path = sys.argv[1] if len(sys.argv) > 1 else "data/Advertising.csv"
df = pd.read_csv(path)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
df.columns = [c.strip().title() if c.lower() != "tv" else "TV" for c in df.columns]
channels = ["TV", "Radio", "Newspaper"]
print(df.describe().round(2), "\n")
print("Missing values:", int(df.isna().sum().sum()))

# 2. Correlations
corr = df.corr()
print("\nCorrelation with Sales:\n", corr["Sales"].drop("Sales").round(3), "\n")
sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f")
plt.title("Correlation matrix"); plt.savefig("outputs/sales_correlation.png", dpi=150, bbox_inches="tight"); plt.close()
fig, ax = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for a, c in zip(ax, channels):
    sns.regplot(data=df, x=c, y="Sales", ax=a, scatter_kws={"alpha": .6})
    a.set_title(f"{c} spend vs Sales")
plt.tight_layout(); plt.savefig("outputs/sales_scatter.png", dpi=150); plt.close()

# 3. Model
X, y = df[channels], df["Sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)

# 4. Evaluate + channel impact
pred = model.predict(X_test)
r2 = r2_score(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
cv_r2 = cross_val_score(LinearRegression(), X, y, cv=5, scoring="r2").mean()
print(f"R2 (test): {r2:.3f} | RMSE: {rmse:.3f} | MAE: {mae:.3f} | 5-fold CV R2: {cv_r2:.3f}")
coef = pd.Series(model.coef_, index=channels)
print("\nCoefficient (sales units per unit of spend):\n", coef.round(4))
std = LinearRegression().fit(StandardScaler().fit_transform(X), y)
std_coef = pd.Series(std.coef_, index=channels)
print("\nStandardised coefficient (comparable impact):\n", std_coef.round(3))
top = std_coef.abs().idxmax()
print(f"\nBiggest impact channel: {top}")
plt.figure(figsize=(5, 4)); plt.scatter(y_test, pred, alpha=.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.xlabel("Actual sales"); plt.ylabel("Predicted sales"); plt.title("Actual vs predicted")
plt.savefig("outputs/sales_actual_vs_pred.png", dpi=150, bbox_inches="tight"); plt.close()

# 5. Insights (generated from the fitted model)
print("\nBudget insights:")
for c in channels:
    verdict = "worth investing in" if std_coef[c] > 0.1 else "little measurable effect"
    print(f" - {c}: +1 unit of spend -> {coef[c]:+.3f} sales; {verdict}")
