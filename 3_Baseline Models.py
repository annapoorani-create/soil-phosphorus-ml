# ------------------
# Boilerplate
# ------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

st.title("Baseline Models")
df = pd.read_csv(
    'HedleyP_database_20140708.xls',
    encoding='latin1',
    header=25,
)

# --- Compute labile_pi (same logic as your notebook/script) ---
df["labile_pi"] = -9999
mask = (df["Resin_Pi"] != -9999) & (df["Bicarbonate_Pi"] != -9999)
df.loc[mask, "labile_pi"] = df.loc[mask, "Resin_Pi"] + df.loc[mask, "Bicarbonate_Pi"]

hdf = df.copy()
hdf = hdf.replace(-9999, np.nan)

hdf.columns = hdf.columns.astype(str).str.strip()

# replacing NaN with avergae value by soil type

# make a copy so we don't mess with original data
hdf_avg = hdf.copy()

# identify columns with missing values
columns_with_NaN = []
for column in hdf_avg.columns:
    if hdf_avg[column].isna().any():
        columns_with_NaN.append(column)

# for each column with a missing value, put in averages
for column in columns_with_NaN:
    hdf_avg[column] = hdf_avg[column].fillna(
        hdf_avg.groupby("Soil_Order")[column].transform("mean")
    )
hdf_avg['labile_pi'] = hdf_avg['Resin_Pi'] + hdf_avg['Bicarbonate_Pi']
hdf_avg = hdf_avg.fillna(hdf_avg.mean(numeric_only=True))

hdf_avg.head()

# deleting rows with null values
hdf_drop = hdf.copy()
hdf_drop = hdf_drop.dropna().reset_index().drop('index',axis=1)
hdf_drop['labile_pi'] = hdf_drop['Resin_Pi'] + hdf_drop['Bicarbonate_Pi']

# ------------------
# Training Basic LinReg
# ------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.subheader("Basic Linear Regression Results")

# deleting rows with null values
hdf_drop = hdf.copy()
hdf_drop = hdf_drop.dropna().reset_index().drop('index',axis=1)
hdf_drop['labile_pi'] = hdf_drop['Resin_Pi'] + hdf_drop['Bicarbonate_Pi']

# Drop reference column
hdf_drop = hdf_drop.drop(columns=["Reference"])

# One-hot encode Soil_Order
hdf_drop = pd.get_dummies(hdf_drop, columns=["Soil_Order"], drop_first=True)

# Define target and predictors
target = "labile_pi"
X = hdf_drop.drop(columns=[target, 'Bicarbonate_Pi','Resin_Pi','Total_P'])
y = hdf_drop[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

st.write("R²:", r2)
st.write("RMSE:", rmse)

# --------------
# data viz
# --------------

st.markdown("**Predicted vs Actual (Linear Regression)**")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.7)
ax.set_xlabel("Actual labile_pi")
ax.set_ylabel("Predicted labile_pi")
ax.set_title("Linear Regression: Predicted vs Actual")

# 1:1 line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val])

st.pyplot(fig)

# Create coefficient DataFrame
coef_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_
})

# Sort by absolute value of coefficients
coef_df = coef_df.reindex(
    coef_df.coefficient.abs().sort_values(ascending=False).index
)

top_n = 10
top_coef_df = coef_df.head(top_n)

fig, ax = plt.subplots()
ax.barh(top_coef_df["feature"], top_coef_df["coefficient"])
ax.set_xlabel("Coefficient value")
ax.set_ylabel("Feature")
ax.set_title(f"Top {top_n} Model Coefficients")

st.pyplot(fig)

# show the full importance table
with st.expander("Show all feature importances"):
    st.dataframe(coef_df.reset_index(drop=True))



# ------------------
# Training Basic Random Forest
# ------------------
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

st.subheader("Basic Random Forest Results")

# Rebuild clean modeling dataframe (same preprocessing as LinReg)
hdf_drop = hdf.copy()
hdf_drop = hdf_drop.dropna().reset_index().drop("index", axis=1)
hdf_drop["labile_pi"] = hdf_drop["Resin_Pi"] + hdf_drop["Bicarbonate_Pi"]

# Drop reference column
hdf_drop = hdf_drop.drop(columns=["Reference"])

# One-hot encode Soil_Order
hdf_drop = pd.get_dummies(hdf_drop, columns=["Soil_Order"], drop_first=True)

# Define target and predictors
target = "labile_pi"
X = hdf_drop.drop(columns=[target, "Bicarbonate_Pi", "Resin_Pi", "Total_P"])
y = hdf_drop[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluation
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)

st.write("R²:", r2_rf)
st.write("RMSE:", rmse_rf)

# ------------------
# Plot: Predicted vs Actual
# ------------------
st.markdown("**Predicted vs Actual (Random Forest)**")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred_rf, alpha=0.7)
ax.set_xlabel("Actual labile_pi")
ax.set_ylabel("Predicted labile_pi")
ax.set_title("Random Forest: Predicted vs Actual")

# 1:1 line
min_val = min(y_test.min(), y_pred_rf.min())
max_val = max(y_test.max(), y_pred_rf.max())
ax.plot([min_val, max_val], [min_val, max_val])

st.pyplot(fig)

# ------------------
# Feature Importances
# ------------------
st.markdown("**Top Feature Importances (Random Forest)**")
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

top_n = 20
top_imp = importances.head(top_n)

fig2, ax2 = plt.subplots()
ax2.barh(top_imp.index[::-1], top_imp.values[::-1])
ax2.set_xlabel("Importance")
ax2.set_title(f"Top {top_n} Feature Importances")
st.pyplot(fig2)

# show the full importance table
with st.expander("Show all feature importances"):
    st.dataframe(importances.reset_index().rename(columns={"index": "feature", 0: "importance"}))
