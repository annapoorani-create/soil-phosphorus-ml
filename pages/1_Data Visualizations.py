import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

st.title("Data Visualizations")
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

# --- Display ---
st.subheader("Preview")
st.dataframe(df.head(20))

st.write("Rows with valid labile_pi:", int(mask.sum()))
st.write("Total rows:", len(df))


st.subheader("Missing Values Heatmap")

st.write("Pre-cleaning, the dataframe had many missing values. In the heatmap below, every row correspond to one row of data, and yellow indicates a missing value.")

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(
    hdf.isna(),
    cbar=False,
    cmap="viridis",
    ax=ax
)

ax.set_xlabel("Columns")
ax.set_ylabel("Rows")

st.pyplot(fig)

st.subheader("Correlation Heatmaps")

st.write("While cleaning this dataset, I used two main methods of removing missing values. One was to simply delete every row that includes a missing value. The other was to replace with averages. The following is a correlation heatmap for each of the types of dataframes. A redder color indicates a more positive correlation, and a bluer color a more negative correlation. If the color is not strongly red or blue, that indicates a weak correlation between the variables in question.")

# Compute correlations
corra = hdf_avg.drop(['Soil_Order', 'Reference'], axis=1).corr()
corrb = hdf_drop.drop(['Soil_Order', 'Reference'], axis=1).corr()

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

sns.heatmap(
    corra,
    annot=False,
    cmap="coolwarm",
    fmt=".2f",
    ax=axes[0]
)
axes[0].set_title("Average")

sns.heatmap(
    corrb,
    annot=False,
    cmap="coolwarm",
    fmt=".2f",
    ax=axes[1]
)
axes[1].set_title("Dropped")

# Overall title
fig.suptitle("Correlation Heatmaps", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.95])

st.pyplot(fig) 

st.subheader("Labile_Pi KDEs")
st.write("For the remainder of the visualizations, select how you want the missing values handled using the option on the sidebar. This are density distributions of labile inorganic phosphorous for each of the soil orders - you can select which soil orders you want to see them from.")

choice = st.sidebar.radio(
    "Pick one method of handling null values in visualizations:",
    ["Averaged", "Dropped"],
    index=1,  # default selection
)

if choice == "Dropped":
    mydf = hdf_drop
if choice == "Averaged":
    mydf = hdf_avg
    

st.title("KDE of labile_pi by Soil Order")

valid_groups = (
    mydf.dropna(subset=["labile_pi"])
       .groupby("Soil_Order")["labile_pi"]
       .size()
)

plottable_soils = valid_groups[valid_groups >= 2].index.tolist()

soil_choice = st.multiselect(
    "Choose which soil orders you want to display",
    plottable_soils
)

filtered_df = mydf[mydf["Soil_Order"].isin(soil_choice)]

fig, ax = plt.subplots(figsize=(10, 6))

for soil_order, sub in filtered_df.groupby("Soil_Order"):
    sns.kdeplot(
        data=sub,
        x="labile_pi",
        label=soil_order,
        linewidth=2,
        ax=ax
    )

ax.set_xlabel("labile_pi")
ax.set_ylabel("Density")
ax.legend(title="Soil_Order")

st.pyplot(fig)

st.write(
    f"Soil order with highest mean labile_pi: "
    f"{mydf.groupby('Soil_Order')['labile_pi'].mean().idxmax()}"
)

st.subheader("Linear Regression Plots")
st.write("Select the features you would like to see plotted against each other.")


# Get only numeric columns
numeric_columns = mydf.select_dtypes(include="number").columns.tolist()

# Safety check
if len(numeric_columns) < 2:
    st.error("Not enough numeric columns to plot.")
else:
    option1 = st.selectbox(
        "Select X-axis variable",
        options=numeric_columns,
        index=0
    )

    option2 = st.selectbox(
        "Select Y-axis variable",
        options=[c for c in numeric_columns if c != option1]
    )

    st.title(f"{option1} vs {option2}")

    fig, ax = plt.subplots()

    sns.scatterplot(
        data=hdf_drop,
        x=option1,
        y=option2,
        hue="Soil_Order",
        ax=ax
    )

    sns.regplot(
        data=hdf_drop,
        x=option1,
        y=option2,
        scatter=False,
        ci=None,
        ax=ax
    )

    ax.set_xlabel(option1)
    ax.set_ylabel(option2)
    ax.set_title(f"{option1} vs {option2}")

    st.pyplot(fig)
