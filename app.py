"""
==============================================================================
 LULC & Landslide Causation Dashboard
 Interactive spatial dashboard (Streamlit + Folium) to analyze land use
 change and landslide causation - inspired by ESA WorldCover viewer.
==============================================================================
 Author : Generated for Kandy Landslide LULC Project
 Input  : LULC GeoTIFF raster(s) (classified, values 1-8)
          Landslide site locations (Shapefile .shp or KML .kml)
==============================================================================
"""

import os
import io
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.sample import sample_gen
from shapely.geometry import Point
import folium
from folium.plugins import DualMap, MarkerCluster, MiniMap, Fullscreen
from streamlit_folium import st_folium
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Enable KML reading in geopandas/fiona
import fiona
fiona.drvsupport.supported_drivers['KML'] = 'rw'
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'


# ==============================================================================
# CONFIG - Must match the class codes used in the GEE classification script
# ==============================================================================
CLASS_INFO = {
    1: {"name": "Dense Forest",              "color": "#006400"},
    2: {"name": "Tea / Plantation",           "color": "#7CFC00"},
    3: {"name": "Built-up / Urban",           "color": "#FF0000"},
    4: {"name": "Bare Land / Exposed Soil",   "color": "#D2B48C"},
    5: {"name": "Water Bodies",               "color": "#0000FF"},
    6: {"name": "Grassland / Scrubland",      "color": "#ADFF2F"},
    7: {"name": "Paddy / Cropland",           "color": "#FFD700"},
    8: {"name": "Road / Rock / Barren",       "color": "#808080"},
}
NODATA_VALUE = 0  # pixels with no class / masked out

st.set_page_config(
    page_title="LULC & Landslide Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# STYLING - clean, modern, professional
# ==============================================================================
st.markdown("""
<style>
    .main { background-color: #f7f9fb; }
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; color: #1f4e3d; }
    h1, h2, h3 { color: #123524; }
    .stat-card {
        background: white; border-radius: 12px; padding: 1rem 1.2rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 0.8rem;
    }
    section[data-testid="stSidebar"] { background-color: #10261c; }
    section[data-testid="stSidebar"] * { color: #eef5f0 !important; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CACHED HELPER FUNCTIONS
# ==============================================================================

@st.cache_data(show_spinner=False)
def load_raster_bytes(file_bytes, filename):
    """Save uploaded raster bytes to a temp file and return the path."""
    suffix = os.path.splitext(filename)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file_bytes)
    tmp.close()
    return tmp.name


@st.cache_data(show_spinner=False)
def reproject_raster_to_4326(raster_path):
    """Reproject a classified raster to EPSG:4326 for web map display.
    Returns: rgba array (H,W,4 uint8), bounds [[south,west],[north,east]]
    """
    with rasterio.open(raster_path) as src:
        dst_crs = "EPSG:4326"
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        dst_array = np.zeros((height, width), dtype=np.uint8)

        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )

        # Compute lat/lon bounds
        left, bottom = transform * (0, height)
        right, top = transform * (width, 0)
        bounds = [[bottom, left], [top, right]]

        # Colorize according to CLASS_INFO
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        for code, info in CLASS_INFO.items():
            hexcol = info["color"].lstrip("#")
            r, g, b = tuple(int(hexcol[i:i+2], 16) for i in (0, 2, 4))
            mask = dst_array == code
            rgba[mask, 0] = r
            rgba[mask, 1] = g
            rgba[mask, 2] = b
            rgba[mask, 3] = 200  # opacity
        # nodata stays fully transparent (alpha channel already 0)

        return rgba, bounds


@st.cache_data(show_spinner=False)
def load_landslide_points(file_bytes, filename):
    """Load landslide points from KML or Shapefile bytes -> GeoDataFrame (EPSG:4326)."""
    suffix = os.path.splitext(filename)[1].lower()

    if suffix == ".kml":
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".kml")
        tmp.write(file_bytes)
        tmp.close()
        gdf = gpd.read_file(tmp.name, driver="KML")
    elif suffix == ".zip":
        # zipped shapefile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        tmp.write(file_bytes)
        tmp.close()
        gdf = gpd.read_file(f"zip://{tmp.name}")
    else:
        raise ValueError("Upload a .kml file or a zipped shapefile (.zip)")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    # Keep only point geometries
    gdf = gdf[gdf.geometry.geom_type == "Point"].reset_index(drop=True)
    return gdf


def sample_raster_at_points(raster_path, points_gdf):
    """Sample the raw (native CRS) raster value at each landslide point."""
    with rasterio.open(raster_path) as src:
        pts_native = points_gdf.to_crs(src.crs)
        coords = [(geom.x, geom.y) for geom in pts_native.geometry]
        values = [v[0] for v in src.sample(coords)]
    return values


def build_class_dataframe(values):
    df = pd.DataFrame({"class_code": values})
    df["class_code"] = df["class_code"].astype(int)
    df = df[df["class_code"].isin(CLASS_INFO.keys())]
    df["Land Use"] = df["class_code"].map(lambda c: CLASS_INFO[c]["name"])
    summary = (
        df.groupby("Land Use")
        .size()
        .reset_index(name="Landslide Count")
        .sort_values("Landslide Count", ascending=False)
    )
    total = summary["Landslide Count"].sum()
    summary["Percentage"] = (summary["Landslide Count"] / total * 100).round(1)
    summary["color"] = summary["Land Use"].map(
        {v["name"]: v["color"] for v in CLASS_INFO.values()}
    )
    return summary, total


# ==============================================================================
# SIDEBAR - Data inputs
# ==============================================================================
st.sidebar.title("🛰️ Dashboard Controls")
st.sidebar.markdown("Upload your classified LULC raster(s) and landslide site "
                     "locations to begin the analysis.")

st.sidebar.subheader("1. LULC Raster(s)")
lulc_files = st.sidebar.file_uploader(
    "Upload one or more classified GeoTIFFs (class values 1-8)",
    type=["tif", "tiff"],
    accept_multiple_files=True,
)

st.sidebar.subheader("2. Landslide Site Locations")
landslide_file = st.sidebar.file_uploader(
    "Upload landslide points (.kml or zipped shapefile .zip)",
    type=["kml", "zip"],
)

st.sidebar.subheader("3. Map Options")
show_dual_map = st.sidebar.checkbox("Side-by-side comparison view", value=True)
overlay_opacity = st.sidebar.slider("LULC layer opacity", 0.2, 1.0, 0.75, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Class Legend**"
)
for code, info in CLASS_INFO.items():
    st.sidebar.markdown(
        f"<span style='display:inline-block;width:12px;height:12px;"
        f"background:{info['color']};margin-right:6px;border-radius:2px;'></span>"
        f"{info['name']}",
        unsafe_allow_html=True,
    )


# ==============================================================================
# MAIN HEADER
# ==============================================================================
st.title("🛰️ LULC & Landslide Causation Dashboard")
st.caption(
    "Analyze land use / land cover change over time and identify which land "
    "use types are most associated with landslide occurrence."
)

if not lulc_files:
    st.info("👈 Upload at least one classified LULC GeoTIFF from the sidebar to get started.")
    st.stop()


# ==============================================================================
# PERIOD SELECTION
# ==============================================================================
period_labels = [f.name for f in lulc_files]
selected_label = st.selectbox("Select LULC period to display on the map", period_labels)
selected_file = lulc_files[period_labels.index(selected_label)]

raster_path = load_raster_bytes(selected_file.getvalue(), selected_file.name)
rgba, bounds = reproject_raster_to_4326(raster_path)

center_lat = (bounds[0][0] + bounds[1][0]) / 2
center_lon = (bounds[0][1] + bounds[1][1]) / 2


# ==============================================================================
# LOAD LANDSLIDE POINTS (once, reused across all periods)
# ==============================================================================
landslide_gdf = None
if landslide_file is not None:
    landslide_gdf = load_landslide_points(landslide_file.getvalue(), landslide_file.name)


# ==============================================================================
# MAP SECTION
# ==============================================================================
st.subheader(f"🗺️ Interactive Map — {selected_label}")

if show_dual_map:
    m = DualMap(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")
    target_maps = [m.m1, m.m2]
    left_title, right_title = "LULC Classification Only", "LULC + Landslide Sites"
else:
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")
    target_maps = [m]

for i, tm in enumerate(target_maps):
    folium.raster_layers.ImageOverlay(
        image=rgba,
        bounds=bounds,
        opacity=overlay_opacity,
        name=f"LULC - {selected_label}",
    ).add_to(tm)
    folium.TileLayer("Esri.WorldImagery", name="Satellite Basemap").add_to(tm)

    # Only add landslide markers on the second map (right side) or single map
    add_markers = (not show_dual_map) or (i == 1)
    if add_markers and landslide_gdf is not None:
        cluster = MarkerCluster(name="Landslide Sites").add_to(tm)
        for _, row in landslide_gdf.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                color="#8B0000",
                fill=True,
                fill_color="#FF3333",
                fill_opacity=0.9,
                popup=folium.Popup(f"Landslide site<br>Lat: {row.geometry.y:.5f}<br>"
                                    f"Lon: {row.geometry.x:.5f}", max_width=200),
            ).add_to(cluster)

    folium.LayerControl(collapsed=False).add_to(tm)
    Fullscreen().add_to(tm)

if show_dual_map:
    col_l, col_r = st.columns(2)
    col_l.markdown(f"**{left_title}**")
    col_r.markdown(f"**{right_title}**")

st_folium(m, width=None, height=560, returned_objects=[])


# ==============================================================================
# STATISTICS PANEL
# ==============================================================================
st.markdown("---")
st.subheader("📊 Landslide Causation Analysis")

if landslide_gdf is None:
    st.warning("Upload landslide site locations from the sidebar to see causation statistics.")
else:
    values = sample_raster_at_points(raster_path, landslide_gdf)
    summary, total = build_class_dataframe(values)

    if summary.empty:
        st.error("No landslide points fell within the raster extent / valid classes. "
                  "Check that your points and raster cover the same area.")
    else:
        top_row = summary.iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Landslide Sites Analyzed", int(total))
        c2.metric("Leading Land Use Cause", top_row["Land Use"])
        c3.metric("Share on Leading Land Use", f"{top_row['Percentage']}%")

        col_chart, col_table = st.columns([1.3, 1])

        with col_chart:
            fig = px.bar(
                summary.sort_values("Percentage"),
                x="Percentage", y="Land Use", orientation="h",
                text="Percentage",
                color="Land Use",
                color_discrete_map={row["Land Use"]: row["color"] for _, row in summary.iterrows()},
            )
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(
                showlegend=False, height=380,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="% of Landslides", yaxis_title="",
                plot_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            st.markdown("**Breakdown Table**")
            st.dataframe(
                summary[["Land Use", "Landslide Count", "Percentage"]]
                .rename(columns={"Percentage": "Percentage (%)"}),
                hide_index=True,
                use_container_width=True,
            )

        # ----------------------------------------------------------------
        # Multi-period comparison (if more than one LULC raster uploaded)
        # ----------------------------------------------------------------
        if len(lulc_files) > 1:
            st.markdown("---")
            st.subheader("📈 Landslide Land Use — Comparison Across All Uploaded Periods")
            st.caption("Shows what land use was recorded at the landslide points in "
                       "EACH uploaded period (e.g. compare pre- vs post-landslide land use).")

            all_period_rows = []
            for f in lulc_files:
                p_path = load_raster_bytes(f.getvalue(), f.name)
                p_values = sample_raster_at_points(p_path, landslide_gdf)
                p_summary, p_total = build_class_dataframe(p_values)
                p_summary["Period"] = f.name
                all_period_rows.append(p_summary)

            comp_df = pd.concat(all_period_rows, ignore_index=True)

            fig2 = px.bar(
                comp_df, x="Period", y="Percentage", color="Land Use",
                barmode="stack",
                color_discrete_map={v["name"]: v["color"] for v in CLASS_INFO.values()},
                text="Percentage",
            )
            fig2.update_layout(
                height=420, plot_bgcolor="white",
                yaxis_title="% of Landslide Sites", xaxis_title="",
                legend_title="Land Use",
            )
            st.plotly_chart(fig2, use_container_width=True)


st.markdown("---")
st.caption("Dashboard built with Streamlit, Folium & Rasterio · "
           "Class scheme matches the 8-class GEE Random Forest LULC classification.")