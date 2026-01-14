# ----------------------------
# Boiler Plate
# ----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
import geopandas as gpd  # <-- you need this for GeoDataFrame

st.title("Maps")

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(
    "HedleyP_database_20140708.xls",  # (your filename)
    encoding="latin1",
    header=25,
)

# --- Compute labile_pi (same logic as your notebook/script) ---
df["labile_pi"] = -9999
mask = (df["Resin_Pi"] != -9999) & (df["Bicarbonate_Pi"] != -9999)
df.loc[mask, "labile_pi"] = df.loc[mask, "Resin_Pi"] + df.loc[mask, "Bicarbonate_Pi"]

# ----------------------------
# Build null-handling variants
# ----------------------------
hdf = df.copy().replace(-9999, np.nan)
hdf.columns = hdf.columns.astype(str).str.strip()

# fill NaNs with averages by Soil_Order (numeric columns only)
hdf_avg = hdf.copy()

for col in hdf_avg.columns:
    if hdf_avg[col].isna().any() and pd.api.types.is_numeric_dtype(hdf_avg[col]):
        if "Soil_Order" in hdf_avg.columns:
            hdf_avg[col] = hdf_avg[col].fillna(hdf_avg.groupby("Soil_Order")[col].transform("mean"))

# recompute labile_pi where possible
if {"Resin_Pi", "Bicarbonate_Pi"}.issubset(hdf_avg.columns):
    hdf_avg["labile_pi"] = hdf_avg["Resin_Pi"] + hdf_avg["Bicarbonate_Pi"]

# fill remaining numeric NaNs with overall numeric means
hdf_avg = hdf_avg.fillna(hdf_avg.mean(numeric_only=True))

# drop rows with any NaNs
hdf_drop = hdf.copy().dropna().reset_index(drop=True)
if {"Resin_Pi", "Bicarbonate_Pi"}.issubset(hdf_drop.columns):
    hdf_drop["labile_pi"] = hdf_drop["Resin_Pi"] + hdf_drop["Bicarbonate_Pi"]

# ----------------------------
# Local helpers (defined HERE so no NameError)
# ----------------------------
NULL_STRATEGIES = [
    "Nulls -> Soil_Order means (hdf_avg)",
    "Drop rows with any nulls (hdf_drop)",
]


def _get_df_by_null_strategy(df_raw, hdf, hdf_avg, hdf_drop, strategy: str) -> pd.DataFrame:

    if strategy == "Nulls -> Soil_Order means (hdf_avg)":
        return hdf_avg
    if strategy == "Drop rows with any nulls (hdf_drop)":
        return hdf_drop
    return hdf_avg


def _make_gdf(df_in: pd.DataFrame) -> gpd.GeoDataFrame | None:
    if not {"Longitude", "Latitude"}.issubset(df_in.columns):
        return None
    try:
        return gpd.GeoDataFrame(
            df_in.copy(),
            geometry=gpd.points_from_xy(df_in["Longitude"], df_in["Latitude"]),
            crs="EPSG:4326",
        )
    except Exception:
        return None


# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.subheader("Map controls")
    map_null_strategy = st.selectbox(
        "Null handling for map data",
        NULL_STRATEGIES,
        index=0,
    )

# ----------------------------
# Prepare data
# ----------------------------
df_map = _get_df_by_null_strategy(df, hdf, hdf_avg, hdf_drop, map_null_strategy)

gdf = _make_gdf(df_map)
if gdf is None:
    st.error("Map requires `Latitude` and `Longitude` columns.")
    st.stop()

# ----------------------------
# Load basemap (optional)
# ----------------------------
try:
    import geodatasets
    world = gpd.read_file(geodatasets.get_path("naturalearth.land"))
except Exception:
    world = None

# ----------------------------
# Summary metrics
# ----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Plotted points", len(gdf))
c2.metric("Soil_Orders", int(gdf["Soil_Order"].nunique()) if "Soil_Order" in gdf.columns else "—")

if "labile_pi" in gdf.columns:
    vals = gdf["labile_pi"].replace(-9999, np.nan).dropna()
    c3.metric("Median labile_pi", f"{vals.median():.2f}" if len(vals) else "—")
    c4.metric("Max labile_pi", f"{vals.max():.2f}" if len(vals) else "—")
else:
    c3.metric("Median labile_pi", "—")
    c4.metric("Max labile_pi", "—")

st.divider()

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(
    [
        "Sample locations",
        "Soil order + sized by Labile Pi",
        "Preview data",
    ]
)

# ----------------------------
# Tab 1: Sample locations
# ----------------------------
with tab1:
    st.markdown("### Sample Locations")

    fig1, ax1 = plt.subplots(figsize=(12, 7))
    if world is not None:
        world.plot(ax=ax1, color="lightgrey", edgecolor="white", linewidth=0.5)

    gdf.plot(ax=ax1, markersize=20)
    ax1.set_axis_off()
    st.pyplot(fig1, clear_figure=True)

# ----------------------------
# Tab 2: Soil order + Labile Pi sizing
# ----------------------------
with tab2:
    st.markdown("### Soil Order by Location - sized by Labile Pi")

    fig2, ax2 = plt.subplots(figsize=(12, 7))
    if world is not None:
        world.plot(ax=ax2, color="lightgrey", edgecolor="white", linewidth=0.5)

    gdf2 = gdf.copy()
    if "labile_pi" not in gdf2.columns:
        gdf2["labile_pi"] = np.nan

    sizes = gdf2["labile_pi"].replace(-9999, np.nan)
    sizes = sizes.fillna(sizes.median() if np.isfinite(sizes).any() else 10.0)
    sizes = np.clip(sizes, a_min=1.0, a_max=None)

    gdf2.plot(
        ax=ax2,
        column="Soil_Order" if "Soil_Order" in gdf2.columns else None,
        legend=True if "Soil_Order" in gdf2.columns else False,
        markersize=sizes,
    )

    ax2.set_axis_off()
    st.pyplot(fig2, clear_figure=True)

# ----------------------------
# Tab 3: Data preview
# ----------------------------
with tab3:
    st.markdown("### Map data preview")

    preview_cols = ["Latitude", "Longitude"]
    if "Soil_Order" in df_map.columns:
        preview_cols.append("Soil_Order")
    if "labile_pi" in df_map.columns:
        preview_cols.append("labile_pi")

    st.dataframe(df_map[preview_cols].head(50), use_container_width=True)
