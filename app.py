import os
import json
import tempfile
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import geometry_mask
from shapely import wkt as shapely_wkt
from shapely.ops import unary_union
from PIL import Image
import streamlit.components.v1 as components
import plotly.express as px

# ==============================================================================
# CLASS CONFIGURATION
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
    page_title="ESA WorldCover LULC Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0d1117; color: #e6edf3; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    h1, h2, h3 { color: #2ea043; }
    section[data-testid="stSidebar"] { background-color: #161b22; }
    section[data-testid="stSidebar"] * { color: #f0f6fc !important; }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; color: #3fb950; }
    iframe { border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
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
def load_boundary_vector(file_bytes, filename):
    """Load the boundary/landslide vector (points OR polygons) as EPSG:4326."""
    suffix = os.path.splitext(filename)[1].lower()
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, filename)
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    if suffix == ".kml":
        gdf = gpd.read_file(tmp_path, engine="pyogrio")
    elif suffix == ".zip":
        gdf = gpd.read_file(f"zip://{tmp_path}", engine="pyogrio")
    elif suffix in [".geojson", ".json"]:
        gdf = gpd.read_file(tmp_path, engine="pyogrio")
    else:
        raise ValueError("Upload a .kml, .geojson, or zipped shapefile (.zip)")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    return gdf


def feature_label(row, idx):
    """Best-effort human readable label for a boundary feature."""
    for col in ["name", "Name", "NAME", "id", "ID", "Id", "label", "Label", "title", "Title"]:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col])
    return f"Boundary Feature #{idx + 1}"


@st.cache_data(show_spinner=False)
def reproject_raster_to_rgba(raster_path, opacity_val=0.85, mask_wkt=None):
    """Reproject a classified raster to EPSG:4326 RGBA.

    If mask_wkt is provided (a single WKT polygon/multipolygon string), every
    pixel OUTSIDE that geometry is forced fully transparent - this is what
    makes the LULC colours strictly confined to the uploaded boundary /
    buffer zone instead of spilling across the raster's full extent.
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

        left, bottom = transform * (0, height)
        right, top = transform * (width, 0)
        bounds = [[bottom, left], [top, right]]

        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        alpha_byte = int(opacity_val * 255)

        for code, info in CLASS_INFO.items():
            hexcol = info["color"].lstrip("#")
            r, g, b = tuple(int(hexcol[i:i+2], 16) for i in (0, 2, 4))
            class_mask = (dst_array == code)
            rgba[class_mask, 0] = r
            rgba[class_mask, 1] = g
            rgba[class_mask, 2] = b
            rgba[class_mask, 3] = alpha_byte

        # --- Strictly confine colour to the boundary / buffer geometry ---
        if mask_wkt:
            geom = shapely_wkt.loads(mask_wkt)
            # geometry_mask default (invert=False) => True OUTSIDE the shapes.
            outside = geometry_mask([geom], out_shape=(height, width), transform=transform, invert=False)
            rgba[outside, 3] = 0  # force transparent outside the boundary

        return rgba, bounds


def extract_masked_statistics(raster_path, geom_wkt):
    """Sample class pixels strictly inside the given WKT geometry (native CRS reprojected)."""
    if not geom_wkt:
        return np.array([])
    with rasterio.open(raster_path) as src:
        geom_4326 = shapely_wkt.loads(geom_wkt)
        geom_gdf = gpd.GeoDataFrame(geometry=[geom_4326], crs="EPSG:4326")
        geom_native = geom_gdf.to_crs(src.crs)
        shapes = [g.__geo_interface__ for g in geom_native.geometry if g.is_valid]
        if not shapes:
            return np.array([])
        try:
            from rasterio.mask import mask as rio_mask
            out_image, _ = rio_mask(src, shapes, crop=True, nodata=0)
            pixels = out_image[0].flatten()
            valid_px = pixels[np.isin(pixels, list(CLASS_INFO.keys()))]
            return valid_px
        except Exception:
            return np.array([])


def gdf_to_geojson_dict(gdf):
    if gdf is None or len(gdf) == 0:
        return None
    return json.loads(gdf.to_json())


def rgba_array_to_data_uri(rgba_array):
    img = Image.fromarray(rgba_array, mode="RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG")
    import base64
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.title("🛰️ Dashboard Controls")

st.sidebar.subheader("1. Boundary / Landslide Outline")
st.sidebar.caption("Upload this FIRST — it stays glowing on top of the map and defines the exact zone the LULC rasters are cropped to.")
landslide_file = st.sidebar.file_uploader(
    "Upload Boundary (.zip shapefile / .kml / .geojson)",
    type=["zip", "kml", "geojson", "json"],
)

st.sidebar.subheader("2. Multi-Year LULC Rasters")
lulc_files = st.sidebar.file_uploader(
    "Upload GeoTIFFs (e.g., 2005.tif, 2025.tif)",
    type=["tif", "tiff"],
    accept_multiple_files=True,
)

buffer_dist = st.sidebar.slider("Boundary Buffer (Meters)", 0, 500, 50, 25)
overlay_opacity = st.sidebar.slider("LULC Layer Opacity", 0.2, 1.0, 0.85, 0.05)
restrict_colors_to_boundary = st.sidebar.checkbox(
    "Restrict LULC colours to boundary only",
    value=False,
    help="OFF (default): full classified LULC colour shows everywhere, like ESA WorldCover, "
         "with your boundary glowing on top as a highlight — zoom in to inspect a site closely. "
         "ON: LULC colour is masked to strictly inside the boundary/buffer, everything else "
         "left transparent. Statistics below always use the boundary-restricted pixels either way.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Class Legend**")
for code, info in CLASS_INFO.items():
    st.sidebar.markdown(
        f"<span style='display:inline-block;width:12px;height:12px;"
        f"background:{info['color']};margin-right:8px;border-radius:2px;'></span>"
        f"**{info['name']}**",
        unsafe_allow_html=True,
    )

# ==============================================================================
# MAIN PAGE
# ==============================================================================
st.title("🛰️ ESA WorldCover Style LULC Swipe & Boundary Dashboard")

if landslide_file is None:
    st.info("👈 Start by uploading your boundary / landslide outline (.zip, .kml, or .geojson) in the sidebar. "
            "It will glow on top of the map, and the LULC rasters you upload next will be cropped strictly to it.")
    st.stop()

if not lulc_files:
    st.info("👈 Boundary loaded. Now upload one or more LULC GeoTIFFs from the sidebar to overlay behind it.")

# --- Load boundary ---
landslide_gdf = None
buffered_gdf = None
try:
    landslide_gdf = load_boundary_vector(landslide_file.getvalue(), landslide_file.name)
    st.sidebar.success(f"Loaded {len(landslide_gdf)} boundary feature(s)")

    if buffer_dist > 0:
        projected = landslide_gdf.to_crs("EPSG:3857")
        projected_buffer = projected.buffer(buffer_dist)
        buffered_gdf = gpd.GeoDataFrame(
            landslide_gdf.drop(columns="geometry").reset_index(drop=True),
            geometry=projected_buffer.reset_index(drop=True),
            crs="EPSG:3857",
        ).to_crs("EPSG:4326")
    else:
        # Points with 0 buffer have no area - always apply a tiny minimum
        # buffer so point-based boundaries still produce a visible/maskable zone.
        geom_types = landslide_gdf.geometry.geom_type.unique()
        if any(t in ["Point", "MultiPoint"] for t in geom_types):
            projected = landslide_gdf.to_crs("EPSG:3857")
            projected_buffer = projected.buffer(25)
            buffered_gdf = gpd.GeoDataFrame(
                landslide_gdf.drop(columns="geometry").reset_index(drop=True),
                geometry=projected_buffer.reset_index(drop=True),
                crs="EPSG:3857",
            ).to_crs("EPSG:4326")
        else:
            buffered_gdf = landslide_gdf.copy()
except Exception as e:
    st.sidebar.error(f"Boundary load error: {e}")
    st.stop()

# --- Feature selector: analyze ALL boundaries together, or click into ONE ---
feature_options = ["🌐 All Boundaries (Combined)"] + [
    feature_label(row, i) for i, row in landslide_gdf.iterrows()
]
selected_feature = st.selectbox(
    "🎯 Select a boundary to inspect (choose one landslide site for a focused comparison, "
    "or keep 'All Boundaries' to analyze everything together)",
    feature_options,
)

if selected_feature == feature_options[0]:
    selection_gdf = buffered_gdf
else:
    sel_idx = feature_options.index(selected_feature) - 1
    selection_gdf = buffered_gdf.iloc[[sel_idx]]

# Single merged geometry (WKT) used for raster masking + stats - this is the
# "hashable" cache key that lets reproject_raster_to_rgba stay cached per selection.
selection_geom = unary_union(selection_gdf.geometry.values)
selection_wkt = selection_geom.wkt

if not lulc_files:
    st.stop()

raster_dict = {}
for f in lulc_files:
    path = save_uploaded_bytes(f.getvalue(), f.name)
    raster_dict[f.name] = path
sorted_periods = sorted(list(raster_dict.keys()))

# ==============================================================================
# PERIOD SELECTION
# ==============================================================================
col_l, col_r = st.columns(2)
with col_l:
    left_period = st.selectbox("Left Map Period (T1)", sorted_periods, index=0)
with col_r:
    right_idx = len(sorted_periods) - 1 if len(sorted_periods) > 1 else 0
    right_period = st.selectbox("Right Map Period (T2)", sorted_periods, index=right_idx)

# Both rasters are reprojected AND masked to the selected boundary/buffer -
# this is what stops LULC colour from spilling outside your study area and
# makes the swipe comparison strictly about the zone you care about.
display_mask_wkt = selection_wkt if restrict_colors_to_boundary else None
rgba_left, bounds_left = reproject_raster_to_rgba(raster_dict[left_period], overlay_opacity, display_mask_wkt)
rgba_right, bounds_right = reproject_raster_to_rgba(raster_dict[right_period], overlay_opacity, display_mask_wkt)

sel_bounds = selection_gdf.total_bounds  # [minx, miny, maxx, maxy]
center_lat = (sel_bounds[1] + sel_bounds[3]) / 2
center_lon = (sel_bounds[0] + sel_bounds[2]) / 2
fit_bounds_latlon = [[sel_bounds[1], sel_bounds[0]], [sel_bounds[3], sel_bounds[2]]]

uri_left = rgba_array_to_data_uri(rgba_left)
uri_right = rgba_array_to_data_uri(rgba_right)

boundary_geojson = gdf_to_geojson_dict(selection_gdf)
boundary_geojson_str = json.dumps(boundary_geojson) if boundary_geojson else "null"

MAP_HEIGHT_PX = 640

# ==============================================================================
# MAP: vanilla-JS Leaflet swipe (no risky third-party plugin dependency).
# Layer order (bottom -> top):
#   1. Satellite basemap
#   2. LULC T1 (masked to boundary) - always fully visible underneath
#   3. LULC T2 (masked to boundary) - clipped by the swipe divider, on top of T1
#   4. Boundary glow (outer wide translucent stroke + inner dashed border) -
#      added LAST so it always sits above both LULC layers, on BOTH sides of
#      the swipe, completely unaffected by the divider.
# ==============================================================================
_MAP_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<style>
  html, body { margin:0; padding:0; width:100%; height:__HEIGHT_PX__px; background:#0b0f19; overflow:hidden; }
  #map-wrap { position:relative; width:100%; height:__HEIGHT_PX__px; background:#0b0f19; }
  #map { position:absolute; top:0; left:0; right:0; bottom:0; }
  #err-box {
    display:none; position:absolute; top:0; left:0; right:0; z-index:2000;
    background:#3b0d0d; color:#ffb4b4; font-family:monospace; font-size:12px;
    padding:10px 14px; border-bottom:2px solid #ff4d4d; white-space:pre-wrap;
  }
  .panel-label {
    position:absolute; top:12px; z-index:900; background:rgba(11,15,25,0.88);
    color:#00e676; font-family:sans-serif; font-weight:700; font-size:13px;
    padding:5px 12px; border-radius:6px; border:1px solid #00e676;
    pointer-events:none;
  }
  #label-left { left:12px; }
  #label-right { right:12px; }
  #coord-box {
    position:absolute; bottom:12px; left:12px; z-index:900;
    background:rgba(11,15,25,0.88); color:#e2e8f0; font-family:monospace;
    font-size:12px; padding:6px 10px; border-radius:6px; border:1px solid #333;
  }
  #divider {
    position:absolute; top:0; bottom:0; width:4px; margin-left:-2px;
    background:#00e676; z-index:950; cursor:ew-resize; box-shadow:0 0 6px rgba(0,230,118,0.8);
  }
  #divider::after {
    content:"\\2194"; position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
    width:32px; height:32px; line-height:32px; text-align:center; border-radius:50%;
    background:#00e676; color:#0b0f19; font-weight:bold; font-size:16px;
  }
</style>
</head>
<body>
<div id="map-wrap">
  <div id="err-box"></div>
  <div id="map"></div>
  <div class="panel-label" id="label-left">LULC __LEFT_LABEL__</div>
  <div class="panel-label" id="label-right">LULC __RIGHT_LABEL__</div>
  <div id="divider"></div>
  <div id="coord-box">Drag the center handle to swipe between years</div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
function showErr(msg) {
  var box = document.getElementById('err-box');
  box.style.display = 'block';
  box.textContent = "Map failed to load: " + msg;
}
window.onerror = function(msg) { showErr(msg); return false; };

try {
  var map = L.map('map', { zoomControl: true }).setView([__CENTER_LAT__, __CENTER_LON__], 15);

  var esri = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Esri', maxZoom: 20
  }).addTo(map);

  var osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: 'OSM', maxZoom: 19
  });

  var leftBounds = __LEFT_BOUNDS__;
  var rightBounds = __RIGHT_BOUNDS__;

  // Layer 1 (bottom): T1 LULC, masked to boundary, always fully visible
  var lulcLeft = L.imageOverlay("__URI_LEFT__", leftBounds, { opacity: __OPACITY__ }).addTo(map);
  // Layer 2 (on top of T1): T2 LULC, masked to boundary, revealed by swipe
  var lulcRight = L.imageOverlay("__URI_RIGHT__", rightBounds, { opacity: __OPACITY__ }).addTo(map);

  var fitBounds = __FIT_BOUNDS__;
  map.fitBounds(fitBounds, { padding: [30, 30] });

  // Layer 3 (top, ALWAYS visible on both sides): glowing boundary outline
  var boundaryData = __GEOJSON__;
  if (boundaryData) {
    // Outer glow
    L.geoJSON(boundaryData, {
      style: function() {
        return { color: '#FF0055', weight: 9, opacity: 0.55, fillOpacity: 0 };
      },
      interactive: false
    }).addTo(map);
    // Sharp dashed inner border
    L.geoJSON(boundaryData, {
      style: function() {
        return { color: '#FFE600', weight: 3, opacity: 1.0, dashArray: '6, 6', fillOpacity: 0 };
      },
      onEachFeature: function(feature, layer) {
        var props = feature.properties || {};
        var keys = Object.keys(props);
        if (keys.length > 0) {
          var txt = keys.slice(0, 4).map(function(k) { return k + ": " + props[k]; }).join("<br>");
          layer.bindTooltip(txt);
        }
      }
    }).addTo(map);
  }

  L.control.layers({ "Satellite": esri, "Street Map": osm }, {}, { collapsed: true }).addTo(map);
  L.control.scale({ position: 'bottomleft' }).addTo(map);

  // ---- Vanilla swipe/reveal slider ----
  var mapWrap = document.getElementById('map-wrap');
  var divider = document.getElementById('divider');
  var dividerFraction = 0.5;

  function applyClip() {
    var rightEl = lulcRight.getElement();
    if (!rightEl) return;
    var wrapRect = mapWrap.getBoundingClientRect();
    var dividerAbsX = wrapRect.left + wrapRect.width * dividerFraction;
    var imgRect = rightEl.getBoundingClientRect();
    var clipPx = dividerAbsX - imgRect.left;
    if (clipPx < 0) clipPx = 0;
    rightEl.style.clipPath = "inset(0 0 0 " + clipPx + "px)";
    rightEl.style.webkitClipPath = "inset(0 0 0 " + clipPx + "px)";
    divider.style.left = (wrapRect.width * dividerFraction) + "px";
  }

  lulcLeft.on('load', applyClip);
  lulcRight.on('load', applyClip);
  map.on('move zoom moveend zoomend', applyClip);
  window.addEventListener('resize', applyClip);
  setTimeout(applyClip, 300);

  var isDragging = false;
  function setDividerFromClientX(clientX) {
    var wrapRect = mapWrap.getBoundingClientRect();
    var frac = (clientX - wrapRect.left) / wrapRect.width;
    dividerFraction = Math.min(1, Math.max(0, frac));
    applyClip();
  }
  divider.addEventListener('mousedown', function(e) { isDragging = true; e.preventDefault(); });
  window.addEventListener('mousemove', function(e) { if (isDragging) setDividerFromClientX(e.clientX); });
  window.addEventListener('mouseup', function() { isDragging = false; });
  divider.addEventListener('touchstart', function(e) { isDragging = true; }, { passive: true });
  window.addEventListener('touchmove', function(e) {
    if (isDragging && e.touches.length) setDividerFromClientX(e.touches[0].clientX);
  }, { passive: true });
  window.addEventListener('touchend', function() { isDragging = false; });

  var coordBox = document.getElementById('coord-box');
  map.on('click', function(e) {
    var lat = e.latlng.lat.toFixed(6);
    var lng = e.latlng.lng.toFixed(6);
    coordBox.innerHTML = "Clicked: Lat " + lat + ", Lon " + lng;
  });

} catch (err) {
  showErr(err.message || String(err));
}
</script>
</body>
</html>
"""

map_html = (
    _MAP_TEMPLATE
    .replace("__HEIGHT_PX__", str(MAP_HEIGHT_PX))
    .replace("__LEFT_LABEL__", left_period)
    .replace("__RIGHT_LABEL__", right_period)
    .replace("__CENTER_LAT__", str(center_lat))
    .replace("__CENTER_LON__", str(center_lon))
    .replace("__LEFT_BOUNDS__", json.dumps(bounds_left))
    .replace("__RIGHT_BOUNDS__", json.dumps(bounds_right))
    .replace("__FIT_BOUNDS__", json.dumps(fit_bounds_latlon))
    .replace("__URI_LEFT__", uri_left)
    .replace("__URI_RIGHT__", uri_right)
    .replace("__OPACITY__", str(overlay_opacity))
    .replace("__GEOJSON__", boundary_geojson_str)
)

try:
    components.html(map_html, height=MAP_HEIGHT_PX + 10, scrolling=False)
except Exception as e:
    st.error(f"Map component failed to render: {e}")

_mask_note = (
    "LULC colour is restricted to strictly inside the glowing boundary — everything else is transparent."
    if restrict_colors_to_boundary else
    "LULC colour covers the full classified area (like ESA WorldCover) — the glowing outline just highlights your boundary; toggle "
    "'Restrict LULC colours to boundary only' in the sidebar to crop colour strictly to it."
)
st.caption(
    "🟢 Drag the center handle to sweep between **" + left_period + "** and **" + right_period +
    "**. The pink/yellow glowing outline is your boundary — it never moves or changes as you swipe. " + _mask_note
)

# ==============================================================================
# ANALYSIS SECTION (scoped to the selected boundary feature)
# ==============================================================================
st.markdown("---")
st.subheader(f"📊 LULC Change & Analysis — {selected_feature}")

time_stats = []
for yr in sorted_periods:
    pixels = extract_masked_statistics(raster_dict[yr], selection_wkt)
    if len(pixels) > 0:
        df_p = pd.DataFrame({"class_code": pixels})
        df_p["Land Use"] = df_p["class_code"].map(lambda c: CLASS_INFO[c]["name"])
        grp = df_p.groupby("Land Use").size().reset_index(name="Pixel Count")
        tot = grp["Pixel Count"].sum()
        grp["Percentage (%)"] = ((grp["Pixel Count"] / tot) * 100).round(2)
        grp["Period"] = yr
        time_stats.append(grp)

if not time_stats:
    st.warning("No valid LULC pixels found inside this boundary for any uploaded period. "
               "Check that your raster(s) and boundary cover the same geographic area.")
else:
    all_df = pd.concat(time_stats, ignore_index=True)

    left_stats = all_df[all_df["Period"] == left_period].sort_values("Percentage (%)", ascending=False)
    right_stats = all_df[all_df["Period"] == right_period].sort_values("Percentage (%)", ascending=False)

    c1, c2, c3 = st.columns(3)
    if not left_stats.empty:
        top_left = left_stats.iloc[0]
        c1.metric(f"Dominant Land Use ({left_period})", top_left["Land Use"], f"{top_left['Percentage (%)']}%")
    if not right_stats.empty:
        top_right = right_stats.iloc[0]
        c2.metric(f"Dominant Land Use ({right_period})", top_right["Land Use"], f"{top_right['Percentage (%)']}%")
        c3.metric("Current Primary Land Use", f"{top_right['Land Use']}")

    st.markdown("### LULC Distribution Inside the Selected Boundary — T1 vs T2")
    fig = px.bar(
        all_df[all_df["Period"].isin([left_period, right_period])],
        x="Period", y="Percentage (%)", color="Land Use",
        barmode="group",
        color_discrete_map={v["name"]: v["color"] for v in CLASS_INFO.values()},
        text="Percentage (%)"
    )
    fig.update_layout(height=400, plot_bgcolor="#161b22", paper_bgcolor="#161b22", font_color="#ffffff")
    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------------------
    # MULTI-YEAR CAUSATION TIMELINE (within the selected boundary)
    # --------------------------------------------------------------------
    if len(sorted_periods) > 1:
        st.markdown("### 📈 Land Use Change Over Time Inside the Boundary")
        st.caption(
            "Percentage share of each LULC class inside the selected boundary, for every "
            "uploaded period — watch which class grows right before the landslide period."
        )
        fig_stack = px.bar(
            all_df, x="Period", y="Percentage (%)", color="Land Use",
            barmode="stack",
            color_discrete_map={v["name"]: v["color"] for v in CLASS_INFO.values()},
            title="LULC Composition Over Time"
        )
        fig_stack.update_layout(height=420, plot_bgcolor="#161b22", paper_bgcolor="#161b22", font_color="#ffffff")
        st.plotly_chart(fig_stack, use_container_width=True)

        watch_classes = ["Dense Forest", "Tea / Plantation", "Bare Land / Exposed Soil", "Built-up / Urban"]
        df_watch = all_df[all_df["Land Use"].isin(watch_classes)]
        if not df_watch.empty:
            fig_trend = px.line(
                df_watch, x="Period", y="Percentage (%)", color="Land Use", markers=True,
                color_discrete_map={v["name"]: v["color"] for v in CLASS_INFO.values()},
                title="Vegetation Loss vs. Bare/Built-up Exposure Trend"
            )
            fig_trend.update_layout(height=340, plot_bgcolor="#161b22", paper_bgcolor="#161b22", font_color="#ffffff")
            st.plotly_chart(fig_trend, use_container_width=True)

        with st.expander("📋 Full year-by-year breakdown table"):
            pivot = all_df.pivot(index="Land Use", columns="Period", values="Percentage (%)").fillna(0)
            st.dataframe(pivot, use_container_width=True)

    with st.expander(f"📋 Detailed table — {left_period} vs {right_period}"):
        st.dataframe(
            all_df[all_df["Period"].isin([left_period, right_period])][["Period", "Land Use", "Pixel Count", "Percentage (%)"]],
            hide_index=True, use_container_width=True,
        )
