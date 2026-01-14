import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

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

# --------------
# Actual stuff
# --------------

st.title("Soil Phosphorous App")
st.write("Use the sidebar to navigate between pages.")

st.subheader("Explanation of the Dataset")
st.write("""
**Dataset name:** HedleyP_database_20140708

**Dataset description:**

This dataset contains global soil phosphorus measurements derived from the Hedley phosphorus fractionation method, which separates soil phosphorus into inorganic and organic pools with different chemical stability and biological availability. The data are compiled from multiple published studies and are commonly used to study soil fertility, phosphorus cycling, and ecosystem nutrient limitation. Each row corresponds to a soil sample. Missing values are coded as -9999.

**Feature descriptions**

**Soil properties:**

Soil_Order: USDA soil order classification of the sample.
pH: Soil acidity or alkalinity.
Organic_C (%): Percentage of organic carbon in the soil.
Total_N (%): Percentage of total nitrogen in the soil.

**Inorganic phosphorus fractions (Pi):**

Resin_Pi: Immediately plant-available inorganic phosphorus.
Bicarbonate_Pi: Readily available inorganic phosphorus.
Labile_Pi: Sum of Resin_Pi and Bicarbonate_Pi, all inorganic phosphorous potentially available to plant.
Hydroxide_Pi: Inorganic phosphorus bound to iron and aluminum oxides.
Sonic_Pi: Occluded inorganic phosphorus released after sonication.
Apatite_P: Calcium-bound phosphorus, generally low availability.
Residue_P: Highly stable and resistant inorganic phosphorus.

**Organic phosphorus fractions (Po):**

Bicarbonate_Po: Labile organic phosphorus.
Hydroxide_Po: Moderately stable organic phosphorus.
Sonic_Po: Protected, stable organic phosphorus.

**Aggregate and metadata:**

Total_P: Total soil phosphorus across all fractions.
Latitude: Latitude of the sampling location in decimal degrees.
Longitude: Longitude of the sampling location in decimal degrees.
Reference: Source publication for the soil data.
""")

st.subheader("Research Question")
st.write("Can labile phosphorus, an unstable and error-prone measurement, be accurately predicted using other soil features in the dataset? \
Given that labile_pi is often inaccurate due to its ability to quickly fluctuate, this study explores whether it can be inferred from more reliable variables. After standard models such as linear regression and decision trees showed limited success, the focus shifted to whether feature engineering - particularly using ratios or products of existing features - could better capture the underlying relationships and improve predictive performance.")

st.subheader("Page Descriptions")

st.write("**Data Viz**")
st.write("This page lets users interactively explore data through heatmaps, distribution plots, and scatter plots. It also contains a preview of what the original dataset looked like.")

st.write("**Maps**")
st.write("This page maps out data so users can see a geographic spread. It was generated using raw lat/long coordinates provided in the data.")


st.write("**Baseline Models**")
st.write("This page contains some baseline ML models meant to try and predict labile phosphorous from other features in the data. They are not very good, and serve to highlight the power of feature engineering on soil datasets.")


st.write("**Feature Engineering**")
st.write("This is the culmination of the project; feature engineered models achieved huge improvements over baseline models and proved that other features of soil can be used to predict labile phosphorous.")
