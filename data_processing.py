import pandas as pd
import numpy as np
import re
import geopandas as gpd
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

def clean_mobility_data(df_raw):
    """Complete data wrangling pipeline with improved issue normalisation."""
    df = df_raw.copy()
    
    # ---- 1. Parse dates ----
    df['reported_at'] = pd.to_datetime(df['reported_at'], errors='coerce')
    df['resolved_at'] = pd.to_datetime(df['resolved_at'], errors='coerce')
    if df['reported_at'].dt.tz is not None:
        df['reported_at'] = df['reported_at'].dt.tz_localize(None)
    if df['resolved_at'].dt.tz is not None:
        df['resolved_at'] = df['resolved_at'].dt.tz_localize(None)
    
    # ---- 2. Deduplicate ----
    if 'report_id' in df.columns:
        df = df.sort_values('reported_at', na_position='last')
        df = df.drop_duplicates(subset=['report_id'], keep='last')
    
    # ---- 3. Issue type normalisation (aggressive) ----
    # Map common variants to clean names
    norm_map = {
        'pothole': 'Pothole',
        'broken traffic light': 'Traffic Light',
        'traffic light': 'Traffic Light',
        'missing sign': 'Missing Sign',
        'unsafe crossing': 'Unsafe Crossing',
        'bus stop damage': 'Bus Stop Damage',
        'busstop': 'Bus Stop Damage',
        'damaged bus': 'Bus Stop Damage',
        'blocked lane': 'Blocked Lane',
        'lane blocked': 'Blocked Lane',      # <-- add this
        'sidewalk obstruction': 'Sidewalk Obstruction',
        'sidewalk': 'Sidewalk Obstruction',
    }
    def normalize_issue(txt):
        if not isinstance(txt, str):
            return 'Other'
        txt_lower = txt.lower().strip()
        # remove punctuation and extra spaces
        txt_clean = re.sub(r'[^\w\s]', '', txt_lower)
        for key, val in norm_map.items():
            if key in txt_clean:
                return val
        # if nothing matches, capitalise first letter of each word
        return ' '.join([w.capitalize() for w in txt_clean.split()])
    df['issue_type'] = df['issue_type'].apply(normalize_issue)
    
    # ---- 4. Severity to numeric ----
    severity_map = {'low': 1, 'medium': 3, 'high': 5}
    def parse_severity(val):
        if pd.isna(val):
            return 3
        if isinstance(val, (int, float)):
            return int(max(1, min(5, val)))
        if isinstance(val, str):
            return severity_map.get(val.lower().strip(), 3)
        return 3
    df['severity'] = df['severity'].apply(parse_severity)
    
    # ---- 5. Cost ----
    def clean_cost(val):
        if pd.isna(val):
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        cleaned = re.sub(r'[^\d\.\-]', '', str(val))
        try:
            return float(cleaned)
        except:
            return 0.0
    df['estimated_impact_cost'] = df['estimated_impact_cost'].apply(clean_cost)
    
    # ---- 6. Coordinates ----
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    valid = df['lat'].between(-90, 90) & df['lon'].between(-180, 180)
    df = df[valid].copy()
    
    # ---- 7. Feature engineering ----
    today = pd.Timestamp.now().tz_localize(None).normalize()
    df['resolution_days'] = (df['resolved_at'] - df['reported_at']).dt.days
    df['is_unresolved'] = df['resolved_at'].isna()
    unresolved_mask = df['is_unresolved']
    df.loc[unresolved_mask, 'resolution_days'] = (today - df.loc[unresolved_mask, 'reported_at']).dt.days
    df['resolution_days'] = df['resolution_days'].clip(lower=0)
    
    df['report_month'] = df['reported_at'].dt.to_period('M').astype(str)
    df['report_week'] = df['reported_at'].dt.isocalendar().week.astype(str) + '-' + df['reported_at'].dt.year.astype(str)
    
    def severity_band(s):
        if s <= 2: return 'Low'
        elif s == 3: return 'Medium'
        else: return 'High'
    df['severity_band'] = df['severity'].apply(severity_band)
    
    return df

def spatial_join_reports(df_reports, geojson_path):
    """Spatial join with robust district column detection."""
    try:
        gdf_districts = gpd.read_file(geojson_path)
    except:
        df_reports['district'] = 'Unknown'
        return df_reports
    
    if gdf_districts.crs is None:
        gdf_districts.set_crs(epsg=4326, inplace=True)
    else:
        gdf_districts = gdf_districts.to_crs(epsg=4326)
    
    geometry = [Point(xy) for xy in zip(df_reports['lon'], df_reports['lat'])]
    gdf_reports = gpd.GeoDataFrame(df_reports, geometry=geometry, crs='epsg:4326')
    joined = gpd.sjoin(gdf_reports, gdf_districts, how='left', predicate='within')
    
    # Prefer zone_id, then zone_name, then any string column
    district_col = None
    for col in ['zone_id', 'zone_name', 'district', 'name', 'District']:
        if col in joined.columns:
            district_col = col
            break
    if district_col is None:
        # fallback: first object column excluding geometry
        for col in joined.columns:
            if col not in ['index_right', 'geometry'] and joined[col].dtype == 'object':
                district_col = col
                break
    if district_col is None:
        joined['district'] = 'Unknown'
    else:
        joined.rename(columns={district_col: 'district'}, inplace=True)
    joined['district'] = joined['district'].fillna('Unknown')
    
    drop_cols = ['geometry', 'index_right'] if 'index_right' in joined.columns else ['geometry']
    return joined.drop(columns=drop_cols, errors='ignore')