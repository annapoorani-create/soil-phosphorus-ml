import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

st.title("Feature Engineering")

df = pd.read_csv(
    'HedleyP_database_20140708.xls',
    encoding='latin1',
    header=25,
)

# --- Compute labile_pi---
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
# Feature Engineering
# ------------------

st.write("Feature engineering summary: \
\
Several derived variables were created to better capture phosphorus pool structure and stoichiometric relationships. A labile inorganic P target (labile_pi) was computed as the sum of Resin_Pi and Bicarbonate_Pi. Non-labile inorganic P (Nonlabile_Pi) was calculated by summing Hydroxide_Pi, Sonic_Pi, Apatite_P, and Residue_P. Total organic P (Total_Po_calc) was derived from Bicarbonate_Po, Hydroxide_Po, and Sonic_Po. Two ratio features were added: the organic-to-inorganic P ratio (Po/Pi = Total_Po_calc / Nonlabile_Pi) and an organic C to hydroxide Pi ratio (OC:HPi = Organic_C / Hydroxide_Pi). These engineered features were included alongside one-hot–encoded soil order classes for model training.\
")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


# deleting rows with null values
hdf_drop = hdf.copy()
hdf_drop = hdf_drop.dropna().reset_index().drop('index',axis=1)
hdf_drop['labile_pi'] = hdf_drop['Resin_Pi'] + hdf_drop['Bicarbonate_Pi']



# feature engineering
# Inorganic P EXCLUDING labile pools
hdf_drop["Nonlabile_Pi"] = (
    hdf_drop["Hydroxide_Pi"]
    + hdf_drop["Sonic_Pi"]
    + hdf_drop["Apatite_P"]
    + hdf_drop["Residue_P"]
 )

# Organic P total
hdf_drop["Total_Po_calc"] = (
    hdf_drop["Bicarbonate_Po"]
    + hdf_drop["Hydroxide_Po"]
    + hdf_drop["Sonic_Po"]
)


hdf_drop["Po/Pi"] = hdf_drop["Total_Po_calc"]/hdf_drop["Nonlabile_Pi"] # RMSE drop of 4 points!
# deleting columns that aren't useful
hdf_drop = hdf_drop.drop(["Latitude"], # 0.04 point drop in RMSE
   # ["Longitude"], # dropping longitude leads to a ~1 point increase in RMSE long more important than lat
axis = 1)
# another random test (let's see if it works!)

# hdf_drop["OC:HPo+HPi"] = hdf_drop['Organic _C']/(hdf_drop['Hydroxide_Pi']+hdf_drop['Hydroxide_Po']) # RMSE drop of 0.05 points

hdf_drop["OC:HPi"] = hdf_drop['Organic _C']/(hdf_drop['Hydroxide_Pi']) # RMSE drop of 0.4 points without OC:HPo+HPi, 0.3 with

# hdf_drop["BPo+SPo:OC"] = (hdf_drop['Bicarbonate_Po'] + hdf_drop['Sonic_Po'])/hdf_drop['Organic _C'] # RMSE increase of 8 points
# hdf_drop["SPi:NlPi"] = hdf_drop['Sonic_Pi']/(hdf_drop['Nonlabile_Pi']) # RMSE increase of 1.7 points


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

# scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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

# -------------------
# Model Interpretation
# -------------------

coeffs = model.coef_
intercept = model.intercept_


coef_df = (
    pd.DataFrame({
        "feature": X.columns,
        "coefficient": model.coef_
    })
    .sort_values("coefficient", key=abs, ascending=False)
)

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

# -------------------
# Prediction Form
# -------------------
st.markdown("---")
st.header("Make a prediction")

# Soil order categories (needed because you used drop_first=True)
soil_categories = sorted(hdf["Soil_Order"].dropna().unique().tolist())
baseline_soil = soil_categories[0] if len(soil_categories) else None

soil_dummy_cols = [c for c in X.columns if c.startswith("Soil_Order_")]

# Engineered feature names you created
ENGINEERED = {"Nonlabile_Pi", "Total_Po_calc", "Po/Pi", "OC:HPi"}

# Columns the user should directly enter (everything in X except soil dummies + engineered)
input_feature_cols = [
    c for c in X.columns
    if (not c.startswith("Soil_Order_")) and (c not in ENGINEERED)
]

# Helpful defaults from training data (median is usually robust)
defaults = {c: float(X_train[c].median()) for c in input_feature_cols if c in X_train.columns}

with st.form("prediction_form"):
    st.subheader("Inputs")

    if soil_categories:
        soil_choice = st.selectbox("Soil order", soil_categories, index=0)
    else:
        soil_choice = None
        st.warning("No Soil_Order categories found in the dataset.")

    user_vals = {}
    for col in input_feature_cols:
        user_vals[col] = st.number_input(
            col,
            value=defaults.get(col, 0.0),
            format="%.6f",
        )

    submitted = st.form_submit_button("Predict labile_pi")

if submitted:
    # Build one-row DataFrame from user inputs
    row = pd.DataFrame([user_vals])

    # Recreate engineered features (only if needed for your X)
    # Nonlabile_Pi = Hydroxide_Pi + Sonic_Pi + Apatite_P + Residue_P
    if "Nonlabile_Pi" in X.columns:
        needed = ["Hydroxide_Pi", "Sonic_Pi", "Apatite_P", "Residue_P"]
        if all(c in row.columns for c in needed):
            row["Nonlabile_Pi"] = row["Hydroxide_Pi"] + row["Sonic_Pi"] + row["Apatite_P"] + row["Residue_P"]
        else:
            row["Nonlabile_Pi"] = np.nan

    # Total_Po_calc = Bicarbonate_Po + Hydroxide_Po + Sonic_Po
    if "Total_Po_calc" in X.columns:
        needed = ["Bicarbonate_Po", "Hydroxide_Po", "Sonic_Po"]
        if all(c in row.columns for c in needed):
            row["Total_Po_calc"] = row["Bicarbonate_Po"] + row["Hydroxide_Po"] + row["Sonic_Po"]
        else:
            row["Total_Po_calc"] = np.nan

    # Po/Pi = Total_Po_calc / Nonlabile_Pi
    if "Po/Pi" in X.columns:
        if "Total_Po_calc" in row.columns and "Nonlabile_Pi" in row.columns:
            row["Po/Pi"] = row["Total_Po_calc"] / row["Nonlabile_Pi"]
        else:
            row["Po/Pi"] = np.nan

    # OC:HPi = Organic _C / Hydroxide_Pi
    if "OC:HPi" in X.columns:
        if "Organic _C" in row.columns and "Hydroxide_Pi" in row.columns:
            row["OC:HPi"] = row["Organic _C"] / row["Hydroxide_Pi"]
        else:
            row["OC:HPi"] = np.nan

    # Add soil dummy columns (all 0 by default)
    for c in soil_dummy_cols:
        row[c] = 0

    # Set the chosen soil dummy to 1 (unless it's the baseline dropped category)
    if soil_choice and baseline_soil and soil_choice != baseline_soil:
        dummy_name = f"Soil_Order_{soil_choice}"
        if dummy_name in row.columns:
            row[dummy_name] = 1

    # Align columns exactly to training X
    row = row.reindex(columns=X.columns, fill_value=0)

    # Handle any NaNs created by division or missing required inputs
    # (Fallback: use training means)
    fill_means = X_train.mean(numeric_only=True)
    row = row.fillna(fill_means)

    # IMPORTANT: Your current model was fit on UN-SCALED X_train (even though you computed scaled arrays).
    # So we predict on the raw row.
    pred = model.predict(row)[0]

    st.success(f"Predicted labile_pi: {pred:.4f}")
    with st.expander("Show the exact model input row"):
        st.dataframe(row)
