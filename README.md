# LULC & Landslide Causation Dashboard

An interactive Streamlit + Folium web dashboard (ESA WorldCover-style) to
visualize classified Land Use / Land Cover (LULC) rasters, overlay landslide
site locations, and automatically compute which land use types are most
associated with landslide occurrence.

---

## 1. What You Need Before Running

| Data | Format | Notes |
|---|---|---|
| Classified LULC raster(s) | `.tif` / `.tiff` | Pixel values must be integers 1–8 matching the class scheme below. Export these directly from your GEE Random Forest script (`Export.image.toDrive`), then download from Google Drive. |
| Landslide site locations | `.kml` **or** zipped Shapefile `.zip` (containing `.shp`, `.shx`, `.dbf`, `.prj`) | Point geometries only. |

### Class scheme (must match your GEE script exactly)
```
1 = Dense Forest
2 = Tea / Plantation
3 = Built-up / Urban
4 = Bare Land / Exposed Soil
5 = Water Bodies
6 = Grassland / Scrubland
7 = Paddy / Cropland
8 = Road / Rock / Barren
```

---

## 2. Installation (Local Machine)

### Step 1 — Install Python
Python 3.9–3.11 recommended. Check with:
```bash
python3 --version
```

### Step 2 — Create a virtual environment (recommended)
```bash
python3 -m venv lulc_env
source lulc_env/bin/activate        # Windows: lulc_env\Scripts\activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

> **Note on `rasterio` / `fiona` (GDAL dependency):**
> - **Windows/Linux (recommended):** `pip install rasterio fiona` usually works out of the box via prebuilt wheels.
> - **Mac (Apple Silicon) or if pip install fails:** use conda instead —
>   ```bash
>   conda create -n lulc_env python=3.10
>   conda activate lulc_env
>   conda install -c conda-forge rasterio geopandas fiona streamlit folium
>   pip install streamlit-folium plotly
>   ```

---

## 3. Run the Dashboard

From inside the project folder:
```bash
streamlit run app.py
```

This opens automatically in your browser at:
```
http://localhost:8501
```

---

## 4. Using the Dashboard

1. **Sidebar → Section 1:** Upload one or more classified LULC `.tif` files
   (e.g. `2005_2010.tif`, `2013_2015.tif`, `2018_2023.tif`,
   `2025_Pre_Landslide.tif`, `2025_2026_Post_Landslide.tif`).
2. **Sidebar → Section 2:** Upload your landslide site locations
   (`.kml` or zipped shapefile).
3. **Select LULC period** from the dropdown above the map to switch which
   raster is displayed.
4. **Side-by-side toggle:** left map = LULC only, right map = LULC +
   landslide site markers (synced pan/zoom).
5. **Statistics panel** (bottom of page) automatically shows:
   - Total landslide sites analyzed
   - The leading land use type at landslide locations
   - A bar chart + table breakdown by land use (%)
   - If you uploaded **multiple periods**, a stacked comparison chart shows
     how the land use at landslide points changed across time (e.g. Forest
     → Bare Land right before a landslide event is a strong causation signal).

---

## 5. Preparing a Zipped Shapefile (if using .shp)

Shapefiles are made of multiple files. Zip all of them together before
uploading:
```bash
zip landslide_points.zip landslide_points.shp landslide_points.shx \
    landslide_points.dbf landslide_points.prj
```

---

## 6. Troubleshooting

| Issue | Fix |
|---|---|
| `rasterio`/`fiona` install fails | Use the conda install method above (Step 3 note) |
| Map shows blank / no color | Check raster values are 1–8 integers, not float or 0-only |
| "No landslide points fell within raster extent" | Your raster and points must be in overlapping geographic areas — reproject in QGIS if needed |
| KML upload fails | Make sure it's a plain `.kml` (not `.kmz` — unzip a `.kmz` to get the `.kml` inside first) |
| App is slow with a very large raster | Downsample/resample the raster in QGIS (e.g. to 30m) before uploading for smoother web display |

---

## 7. Folder Structure
```
lulc_dashboard/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md            # This file
```