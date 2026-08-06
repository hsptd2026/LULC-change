"""
==============================================================================
 Research-Grade LULC & Landslide Causation Dashboard
 Guaranteed Map Render Edition (Fixed Height & Streamlit-Folium Sync)
==============================================================================
"""

import os
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
import folium
from folium.plugins import DualMap, Fullscreen, MeasureControl, MousePosition
from streamlit_folium import st_folium
import plotly.express as px
import fiona

# Enable KML drivers in fiona
fiona.drvsupport.supported_drivers['KML'] = 'rw'
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'

# ==============================================================================
# CLASS CONFIGURATION (GEE Scheme)
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

st.set_page_config(
    page_title="Research-Grade LULC Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS to force iframe map to display properly
st.markdown("""
<style>
    .main { background-color: #0b0f19; color: #ffffff; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    h1, h2, h3 { color: #00e676; }
    section[data-testid="stSidebar"] { background-color: #121824; }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; color: #00e676; }
    iframe { width: 100% !important; min-height: 600px !important; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

@st.cache_data(show_spinner=False)
def save_uploaded_bytes(file_bytes, filename):
    suffix = os.path.splitext(filename)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file_bytes)
    tmp.close()
    return tmp.name


@st.cache_data(show_spinner=False)
def load_landslide_vector(file_bytes, filename):
    suffix = os.path.splitext(filename)[1].lower()
    tmp_dir = tempfile.mkdtemp()

    if suffix == ".kml":
        tmp_path = os.path.join(tmp_dir, "data.kml")
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)
        gdf = gpd.read_file(tmp_path, driver="KML")
    elif suffix == ".zip":
        tmp_path = os.path.join(tmp_dir, "data.zip")
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)
        gdf = gpd.read_file(f"zip://{tmp_path}")
    else:
        raise ValueError("Upload a .kml file or a zipped shapefile (.zip)")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    return gdf


@st.cache_data(show_spinner=False)
def reproject_raster_to_rgba(raster_path, opacity_val=0.85):
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

        left, bottom = transform * (0, height)
        right, top = transform * (width, 0)
        bounds = [[bottom, left], [top, right]]

        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        alpha_byte = int(opacity_val * 255)

        for code, info in CLASS_INFO.items():
            hexcol = info["color"].lstrip("#")
            r, g, b = tuple(int(hexcol[i:i+2], 16) for i in (0, 2, 4))
            mask = dst_array == code
            rgba[mask, 0] = r
            rgba[mask, 1] = g
            rgba[mask, 2] = b
            rgba[mask, 3] = alpha_byte

        return rgba, bounds


def sample_point_across_rasters(lat, lon, raster_dict):
    pt_gdf = gpd.GeoDataFrame(geometry=[gpd.points_from_xy([lon], [lat])[0]], crs="EPSG:4326")
    results = {}

    for year_label, path in raster_dict.items():
        with rasterio.open(path) as src:
            pt_native = pt_gdf.to_crs(src.crs)
            coord = [(pt_native.geometry.iloc[0].x, pt_native.geometry.iloc[0].y)]
            val = list(src.sample(coord))[0][0]
            val = int(val)
            class_name = CLASS_INFO.get(val, {}).get("name", "Unknown / NoData")
            results[year_label] = {"code": val, "class": class_name}

    return results


def extract_masked_statistics(raster_path, geometry_gdf):
    with rasterio.open(raster_path) as src:
        geom_native = geometry_gdf.to_crs(src.crs)
        shapes = [g for g in geom_native.geometry if g.is_valid]

        try:
            out_image, _ = mask(src, shapes, crop=True, nodata=0)
            pixels = out_image[0].flatten()
            valid_px = pixels[np.isin(pixels, list(CLASS_INFO.keys()))]
            return valid_px
        except Exception:
            return np.array([])


# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.title("🛰️ GIS Control Panel")

st.sidebar.subheader("1. LULC Multi-Year Rasters")
lulc_files = st.sidebar.file_uploader(
    "Upload GeoTIFFs (e.g. 2005.tif, 2010.tif, 2025.tif)",
    type=["tif", "tiff"],
    accept_multiple_files=True,
)

st.sidebar.subheader("2. Landslide Layer")
landslide_file = st.sidebar.file_uploader(
    "Upload Vector File (.zip shapefile or .kml)",
    type=["zip", "kml"],
)

buffer_dist = st.sidebar.slider("Landslide Buffer Distance (Meters)", 0, 500, 100, 25)
overlay_opacity = st.sidebar.slider("LULC Layer Opacity", 0.2, 1.0, 0.85, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("**LULC Class Legend**")
for code, info in CLASS_INFO.items():
    st.sidebar.markdown(
        f"<span style='display:inline-block;width:12px;height:12px;"
        f"background:{info['color']};margin-right:8px;border-radius:2px;'></span>"
        f"**{info['name']}**",
        unsafe_allow_html=True,
    )


# ==============================================================================
# MAIN PAGE PROCESSING
# ==============================================================================
st.title("🛰️ Research-Grade LULC & Landslide Web GIS Dashboard")

if not lulc_files:
    st.info("👈 Please upload your classified LULC GeoTIFFs in the sidebar to load the map.")
    st.stop()

raster_dict = {}
for f in lulc_files:
    path = save_uploaded_bytes(f.getvalue(), f.name)
    raster_dict[f.name] = path

sorted_periods = sorted(list(raster_dict.keys()))

landslide_gdf = None
buffered_gdf = None

if landslide_file is not None:
    try:
        landslide_gdf = load_landslide_vector(landslide_file.getvalue(), landslide_file.name)
        st.sidebar.success(f"Loaded {len(landslide_gdf)} Landslide Features")

        if buffer_dist > 0:
            projected = landslide_gdf.to_crs("EPSG:3857")
            projected_buffer = projected.buffer(buffer_dist)
            buffered_gdf = gpd.GeoDataFrame(geometry=projected_buffer, crs="EPSG:3857").to_crs("EPSG:4326")
        else:
            buffered_gdf = landslide_gdf.copy()

    except Exception as e:
        st.sidebar.error(f"Vector Layer Error: {e}")


# ==============================================================================
# SYNCHRONIZED DUAL-MAP VIEW
# ==============================================================================
st.subheader("🗺️ Synchronized Dual-Map Comparison View")

c_left, c_right = st.columns(2)
with c_left:
    left_year = st.selectbox("Left Map Period (T1)", sorted_periods, index=0)
with c_right:
    right_idx = len(sorted_periods) - 1 if len(sorted_periods) > 1 else 0
    right_year = st.selectbox("Right Map Period (T2)", sorted_periods, index=right_idx)

rgba_left, bounds_left = reproject_raster_to_rgba(raster_dict[left_year], overlay_opacity)
rgba_right, bounds_right = reproject_raster_to_rgba(raster_dict[right_year], overlay_opacity)

center_lat = (bounds_left[0][0] + bounds_left[1][0]) / 2
center_lon = (bounds_left[0][1] + bounds_left[1][1]) / 2

# DualMap Initialization
m = DualMap(location=[center_lat, center_lon], zoom_start=13, tiles=None)

# Add Satellite Basemaps
folium.TileLayer("Esri.WorldImagery", name="Satellite").add_to(m.m1)
folium.TileLayer("Esri.WorldImagery", name="Satellite").add_to(m.m2)

# Add LULC Overlays
folium.raster_layers.ImageOverlay(
    image=rgba_left, bounds=bounds_left, opacity=overlay_opacity, name=f"LULC {left_year}"
).add_to(m.m1)

folium.raster_layers.ImageOverlay(
    image=rgba_right, bounds=bounds_right, opacity=overlay_opacity, name=f"LULC {right_year}"
).add_to(m.m2)

# Add Landslide Layer to both panes
if buffered_gdf is not None:
    for map_pane in [m.m1, m.m2]:
        folium.GeoJson(
            buffered_gdf,
            name=f"Landslide Boundary / Buffer ({buffer_dist}m)",
            style_function=lambda x: {
                'fillColor': '#ff0000',
                'color': '#8b0000',
                'weight': 2.5,
                'fillOpacity': 0.4
            }
        ).add_to(map_pane)

Fullscreen().add_to(m.m1)
MeasureControl(position='bottomleft').add_to(m.m1)
MousePosition().add_to(m.m1)

folium.LayerControl(collapsed=False).add_to(m.m1)
folium.LayerControl(collapsed=False).add_to(m.m2)

# MANDATORY GUARANTEED RENDER CALL FOR STREAMLIT
map_output = st_folium(
    m,
    key="lulc_dual_map",
    height=600,
    use_container_width=True,
    returned_objects=["last_clicked"]
)


# ==============================================================================
# CLICK ANYWHERE POINT INSPECTOR & TIMELINE
# ==============================================================================
st.markdown("---")
st.subheader("📍 Point Inspection & Temporal LULC Timeline")

if map_output and map_output.get("last_clicked"):
    click_lat = map_output["last_clicked"]["lat"]
    click_lon = map_output["last_clicked"]["lng"]

    timeline_data = sample_point_across_rasters(click_lat, click_lon, raster_dict)

    st.write(f"**Inspected Coordinate:** Lat `{click_lat:.5f}`, Lon `{click_lon:.5f}`")

    timeline_rows = []
    for yr_name, info in timeline_data.items():
        timeline_rows.append({
            "LULC Period": yr_name,
            "Class Code": info["code"],
            "Land Use": info["class"]
        })

    df_timeline = pd.DataFrame(timeline_rows)

    col_t1, col_t2 = st.columns([1, 1.5])
    with col_t1:
        st.dataframe(df_timeline, hide_index=True, use_container_width=True)

    with col_t2:
        fig_time = px.line(
            df_timeline, x="LULC Period", y="Land Use", markers=True,
            title="LULC Trajectory at Clicked Point",
        )
        fig_time.update_layout(height=280, plot_bgcolor="#121824", paper_bgcolor="#121824", font_color="#ffffff")
        st.plotly_chart(fig_time, use_container_width=True)
else:
    st.info("👆 Map-la edhavadhu oru spot-a click pannunga, andha spot-oda multi-year land use change timeline inga live-a kaattum.")


# ==============================================================================
# LANDSLIDE BUFFER AREA ANALYSIS
# ==============================================================================
st.markdown("---")
st.subheader("📊 Landslide Area LULC Causation Analytics")

if buffered_gdf is None:
    st.warning("Upload a Landslide Shapefile/KML in the sidebar to run buffer extraction analysis.")
else:
    target_stat_year = st.selectbox("Select Year for Landslide Zone Analysis", sorted_periods)
    pixels = extract_masked_statistics(raster_dict[target_stat_year], buffered_gdf)

    if len(pixels) == 0:
        st.error("No valid pixels found within the landslide zones. Verify spatial overlap / CRS.")
    else:
        df_px = pd.DataFrame({"class_code": pixels})
        df_px["Land Use"] = df_px["class_code"].map(lambda c: CLASS_INFO[c]["name"])

        summary = df_px.groupby("Land Use").size().reset_index(name="Pixel Count")
        total_px = summary["Pixel Count"].sum()
        summary["Percentage (%)"] = ((summary["Pixel Count"] / total_px) * 100).round(2)
        summary["Color"] = summary["Land Use"].map({v["name"]: v["color"] for v in CLASS_INFO.values()})
        summary = summary.sort_values("Percentage (%)", ascending=False)

        top_cause = summary.iloc[0]

        m1, m2, m3 = st.columns(3)
        m1.metric("Analyzed Zone Pixels", f"{total_px:,}")
        m2.metric("Primary Causative Cover", top_cause["Land Use"])
        m3.metric("Dominant Cover Share", f"{top_cause['Percentage (%)']}%")

        c_chart, c_table = st.columns([1.4, 1])

        with c_chart:
            fig_bar = px.bar(
                summary, x="Percentage (%)", y="Land Use", orientation="h",
                text="Percentage (%)", color="Land Use",
                color_discrete_map={r["Land Use"]: r["Color"] for _, r in summary.iterrows()},
                title=f"LULC Breakdown within Landslide Zone ({target_stat_year})"
            )
            fig_bar.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_bar.update_layout(showlegend=False, height=360, plot_bgcolor="#121824", paper_bgcolor="#121824", font_color="#ffffff")
            st.plotly_chart(fig_bar, use_container_width=True)

        with c_table:
            st.dataframe(summary[["Land Use", "Pixel Count", "Percentage (%)"]], hide_index=True, use_container_width=True)
