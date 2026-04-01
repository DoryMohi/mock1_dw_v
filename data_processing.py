"""
Data Processing Module for City Transport Efficiency Dashboard
"""

# ===== FIONA COMPATIBILITY PATCH =====
import warnings
warnings.filterwarnings('ignore')

# Comprehensive fiona patch for version compatibility
try:
    import fiona
    import fiona._path
    
    # Check if the required attributes exist
    if not hasattr(fiona, 'path'):
        # Create a mock path module if needed
        class MockPath:
            @staticmethod
            def ParsedPath(*args, **kwargs):
                from fiona._path import ParsedPath as _ParsedPath
                return _ParsedPath(*args, **kwargs)
        
        fiona.path = MockPath()
        print("✅ Patched fiona.path for compatibility")
    
    # Additional patch for ParsedPath if missing
    if not hasattr(fiona._path, 'ParsedPath'):
        try:
            # Try to import from alternative location
            from fiona.path import ParsedPath
            fiona._path.ParsedPath = ParsedPath
        except ImportError:
            # Create a fallback
            class FallbackParsedPath:
                def __init__(self, *args, **kwargs):
                    pass
            fiona._path.ParsedPath = FallbackParsedPath
            print("✅ Created fallback ParsedPath")
            
except ImportError as e:
    print(f"⚠️ Fiona import note: {e}")
except Exception as e:
    print(f"⚠️ Fiona patch note: {e}")

# ===== NORMAL IMPORTS =====
import os
import pandas as pd
import numpy as np
import re
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataQualityMonitor:
    """Monitor and report data quality metrics"""
    
    def __init__(self):
        self.quality_metrics = {}
        
    def calculate_quality_score(self, df):
        scores = []
        completeness = 1 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
        scores.append(completeness * 100)
        
        if 'report_id' in df.columns:
            uniqueness = len(df['report_id'].unique()) / len(df)
            scores.append(uniqueness * 100)
        
        if 'reported_at' in df.columns:
            valid_dates = df['reported_at'].notna().sum() / len(df)
            scores.append(valid_dates * 100)
        
        if 'lat' in df.columns and 'lon' in df.columns:
            valid_coords = ((df['lat'].between(-90, 90)) & (df['lon'].between(-180, 180))).sum() / len(df)
            scores.append(valid_coords * 100)
        
        self.quality_metrics['overall_score'] = np.mean(scores)
        self.quality_metrics['completeness'] = completeness * 100
        self.quality_metrics['uniqueness'] = scores[1] if len(scores) > 1 else 100
        self.quality_metrics['validity'] = np.mean(scores[2:]) if len(scores) > 2 else 100
        
        return self.quality_metrics
    
    def get_quality_report(self):
        return pd.DataFrame([self.quality_metrics])


class AdvancedDataProcessor:
    """Enhanced data processing with advanced features"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def clean_mobility_data(self, df_raw):
        """Complete data wrangling pipeline"""
        df = df_raw.copy()
        
        # Parse dates
        date_columns = ['reported_at', 'resolved_at', 'created_at', 'updated_at']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if hasattr(df[col], 'dt') and df[col].dt.tz is not None:
                    df[col] = df[col].dt.tz_localize(None)
        
        # Deduplicate
        if 'report_id' in df.columns:
            df = df.sort_values('reported_at', na_position='last')
            df = df.drop_duplicates(subset=['report_id'], keep='last')
        
        # Issue type normalization
        norm_map = {
            'pothole': 'Pothole', 'potholes': 'Pothole', 'crack': 'Pothole',
            'broken traffic light': 'Traffic Light', 'traffic light': 'Traffic Light', 'signal': 'Traffic Light',
            'missing sign': 'Missing Sign', 'sign': 'Missing Sign',
            'unsafe crossing': 'Unsafe Crossing', 'crosswalk': 'Unsafe Crossing',
            'bus stop damage': 'Bus Stop Damage', 'busstop': 'Bus Stop Damage',
            'blocked lane': 'Blocked Lane', 'lane blocked': 'Blocked Lane', 'obstruction': 'Blocked Lane',
            'sidewalk obstruction': 'Sidewalk Obstruction', 'sidewalk': 'Sidewalk Obstruction',
            'debris': 'Road Debris', 'litter': 'Road Debris',
            'flooding': 'Flooding/Drainage', 'drain': 'Flooding/Drainage'
        }
        
        def normalize_issue(txt):
            if not isinstance(txt, str):
                return 'Other'
            txt_lower = txt.lower().strip()
            txt_clean = re.sub(r'[^\w\s]', '', txt_lower)
            
            for key, val in norm_map.items():
                if key in txt_clean:
                    return val
            
            words = set(txt_clean.split())
            for key, val in norm_map.items():
                if set(key.split()).intersection(words):
                    return val
            
            return ' '.join([w.capitalize() for w in txt_clean.split()])
        
        df['issue_type'] = df['issue_type'].apply(normalize_issue)
        
        # Severity mapping
        severity_map = {'low': 1, 'minor': 1, 'medium': 3, 'moderate': 3, 'high': 5, 'severe': 5, 'critical': 5}
        
        def parse_severity(val):
            if pd.isna(val):
                return 3
            if isinstance(val, (int, float)):
                return int(max(1, min(5, val)))
            if isinstance(val, str):
                return severity_map.get(val.lower().strip(), 3)
            return 3
        
        df['severity'] = df['severity'].apply(parse_severity)
        
        # Cost cleaning
        def clean_cost(val):
            if pd.isna(val):
                return 0.0
            if isinstance(val, (int, float)):
                return float(max(0, val))
            cleaned = re.sub(r'[^\d\.\-]', '', str(val))
            try:
                return max(0, float(cleaned))
            except:
                return 0.0
        
        df['estimated_impact_cost'] = df['estimated_impact_cost'].apply(clean_cost)
        
        if len(df) > 0:
            cost_99th = df['estimated_impact_cost'].quantile(0.99)
            df['estimated_impact_cost'] = df['estimated_impact_cost'].clip(upper=cost_99th)
        
        # Coordinate validation
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
        valid = df['lat'].between(-90, 90) & df['lon'].between(-180, 180)
        df = df[valid].copy()
        
        # Feature engineering
        if 'reported_at' in df.columns and not df['reported_at'].isnull().all():
            df['reported_hour'] = df['reported_at'].dt.hour
            df['reported_day_of_week'] = df['reported_at'].dt.dayofweek
            df['reported_day_name'] = df['reported_at'].dt.day_name()
            df['reported_month'] = df['reported_at'].dt.month
            df['reported_year'] = df['reported_at'].dt.year
            df['reported_week'] = df['reported_at'].dt.isocalendar().week
            df['is_weekend'] = df['reported_day_of_week'].isin([5, 6]).astype(int)
            
            def get_time_of_day(hour):
                if 6 <= hour < 12: return 'Morning'
                elif 12 <= hour < 18: return 'Afternoon'
                elif 18 <= hour < 24: return 'Evening'
                else: return 'Night'
            
            df['time_of_day'] = df['reported_hour'].apply(get_time_of_day)
        
        # Resolution metrics
        if 'resolved_at' in df.columns and 'reported_at' in df.columns:
            today = pd.Timestamp.now().tz_localize(None).normalize()
            df['resolution_days'] = (df['resolved_at'] - df['reported_at']).dt.days
            df['is_unresolved'] = df['resolved_at'].isna()
            unresolved_mask = df['is_unresolved']
            df.loc[unresolved_mask, 'resolution_days'] = (today - df.loc[unresolved_mask, 'reported_at']).dt.days
            df['resolution_days'] = df['resolution_days'].clip(lower=0)
            df['resolution_efficiency'] = 100 / (df['resolution_days'] + 1)
        
        # Severity bands
        def severity_band(s):
            if s <= 2: return 'Low'
            elif s == 3: return 'Medium'
            else: return 'High'
        
        df['severity_band'] = df['severity'].apply(severity_band)
        
        # Urgency score
        severity_weights = {'Low': 1, 'Medium': 2, 'High': 3}
        if 'severity_band' in df.columns and 'is_unresolved' in df.columns:
            df['urgency_score'] = df['severity_band'].map(severity_weights) * (1 + df['is_unresolved'].astype(int))
        
        # Cost per day
        if 'estimated_impact_cost' in df.columns and 'resolution_days' in df.columns:
            df['cost_per_day'] = df['estimated_impact_cost'] / (df['resolution_days'] + 1)
        
        # Seasonal features
        if 'reported_month' in df.columns:
            df['season'] = pd.cut(df['reported_month'], bins=[0, 3, 6, 9, 12],
                                  labels=['Winter', 'Spring', 'Summer', 'Fall'], include_lowest=True)
        
        # Additional derived metrics
        if 'reported_at' in df.columns:
            df['report_date'] = df['reported_at'].dt.date
            df['report_week_str'] = df['reported_at'].dt.strftime('%Y-W%V')
            df['report_month_str'] = df['reported_at'].dt.strftime('%Y-%m')
        
        return df
    
    def spatial_join_reports(self, df_reports, geojson_path):
        """Spatial join with robust district detection"""
        try:
            if not os.path.exists(geojson_path):
                df_reports['district'] = 'Unknown'
                df_reports['district_id'] = 'UNK'
                return df_reports
            
            gdf_districts = gpd.read_file(geojson_path)
            
            if gdf_districts.crs is None:
                gdf_districts = gdf_districts.set_crs(epsg=4326)
            else:
                gdf_districts = gdf_districts.to_crs(epsg=4326)
            
            geometry = [Point(xy) for xy in zip(df_reports['lon'], df_reports['lat'])]
            gdf_reports = gpd.GeoDataFrame(df_reports, geometry=geometry, crs='epsg:4326')
            joined = gpd.sjoin(gdf_reports, gdf_districts, how='left', predicate='within')
            
            # Find district column
            district_col = None
            possible_names = ['zone_id', 'zone_name', 'district', 'name', 'District', 
                            'district_name', 'area_name', 'neighborhood']
            
            for col in possible_names:
                if col in joined.columns:
                    district_col = col
                    break
            
            if district_col is None:
                for col in joined.columns:
                    if col not in ['index_right', 'geometry'] and joined[col].dtype == 'object':
                        district_col = col
                        break
            
            if district_col is not None:
                joined['district'] = joined[district_col].fillna('Unknown')
            else:
                joined['district'] = 'Unknown'
            
            district_ids = {name: f"D{idx:03d}" for idx, name in enumerate(joined['district'].unique())}
            joined['district_id'] = joined['district'].map(district_ids)
            
            drop_cols = []
            if 'geometry' in joined.columns:
                drop_cols.append('geometry')
            if 'index_right' in joined.columns:
                drop_cols.append('index_right')
            
            result = joined.drop(columns=drop_cols, errors='ignore')
            
            if 'district' not in result.columns:
                result['district'] = 'Unknown'
            
            return result
            
        except Exception as e:
            print(f"Error in spatial join: {e}")
            df_reports['district'] = 'Unknown'
            df_reports['district_id'] = 'UNK'
            return df_reports
    
    def prepare_ml_features(self, df):
        """Prepare features for machine learning"""
        ml_df = df.copy()
        
        feature_columns = ['severity', 'reported_hour', 'reported_day_of_week', 
                          'is_weekend', 'resolution_days', 'estimated_impact_cost',
                          'urgency_score', 'cost_per_day']
        
        available_features = [col for col in feature_columns if col in ml_df.columns]
        
        categorical_cols = ['severity_band', 'time_of_day', 'season']
        for col in categorical_cols:
            if col in ml_df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    ml_df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(ml_df[col].astype(str))
                else:
                    ml_df[f'{col}_encoded'] = self.label_encoders[col].transform(ml_df[col].astype(str))
                available_features.append(f'{col}_encoded')
        
        X = ml_df[available_features].fillna(0)
        return X, available_features
    
    def detect_anomalies(self, df, contamination=0.1):
        return pd.Series([False] * len(df), index=df.index)


# Backward compatibility functions
def clean_mobility_data(df_raw):
    processor = AdvancedDataProcessor()
    return processor.clean_mobility_data(df_raw)

def spatial_join_reports(df_reports, geojson_path):
    processor = AdvancedDataProcessor()
    return processor.spatial_join_reports(df_reports, geojson_path)

def get_data_quality_report(df):
    monitor = DataQualityMonitor()
    metrics = monitor.calculate_quality_score(df)
    return monitor.get_quality_report()

def validate_data_integrity(df):
    issues = []
    critical_fields = ['report_id', 'reported_at', 'lat', 'lon']
    for field in critical_fields:
        if field in df.columns and df[field].isnull().any():
            issues.append(f"Missing values in {field}: {df[field].isnull().sum()}")
    
    if 'resolution_days' in df.columns:
        negative_res = (df['resolution_days'] < 0).sum()
        if negative_res > 0:
            issues.append(f"Negative resolution days found: {negative_res}")
    
    if 'reported_at' in df.columns and 'resolved_at' in df.columns:
        invalid_dates = (df['resolved_at'] < df['reported_at']).sum()
        if invalid_dates > 0:
            issues.append(f"Reports resolved before reported: {invalid_dates}")
    
    return issues