import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def clean_traffic_data(input_path='dataset.csv', output_path='output/cleaned_dataset.csv'):
    print(f"[*] Reading raw data from: {input_path}...")
    
    # Step 1: Read CSV file
    try:
        df = pd.read_csv(
            input_path, 
            delimiter=',',         
            on_bad_lines='skip',   
            encoding='utf-8-sig',  
            low_memory=False       
        )
    except Exception as e:
        print(f"[!] Critical error reading file: {e}")
        return None
    
    # Step 2: Validate required columns
    required_cols = ['Injury Severity', 'Crash Date/Time']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"[!] ERROR: Columns not found: {missing_cols}.")
        return None

    df_clean = df.copy()

    # Step 3: Drop unnecessary columns
    cols_to_drop = ['Non-Motorist Substance Abuse', 'Related Non-Motorist', 'Municipality', 'Off-Road Description', 'Circumstance']
    df_clean = df_clean.drop(columns=cols_to_drop, errors='ignore')

    # Step 4: Clean and filter Injury Severity
    df_clean['Injury Severity'] = df_clean['Injury Severity'].astype(str).str.upper().str.strip()
    valid_severities = ['NO APPARENT INJURY', 'POSSIBLE INJURY', 'SUSPECTED MINOR INJURY', 'SUSPECTED SERIOUS INJURY', 'FATAL INJURY']
    df_clean = df_clean[df_clean['Injury Severity'].isin(valid_severities)]

    # Step 5: Map severity to numeric class
    severity_map = {
        'NO APPARENT INJURY': 1, 
        'POSSIBLE INJURY': 2, 
        'SUSPECTED MINOR INJURY': 2,
        'SUSPECTED SERIOUS INJURY': 3, 
        'FATAL INJURY': 3 
    }
    df_clean['Severity_Class'] = df_clean['Injury Severity'].map(severity_map)

    print("- Cleaning and casting core features...")
    # Step 6: Standardize core categorical columns
    core_cols = ['Driver Substance Abuse', 'Driver Distracted By', 'Weather', 'Surface Condition']
    for col in core_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.upper().str.strip()

    # Step 7: Encode impairment flag
    if 'Driver Substance Abuse' in df_clean.columns:
        safe_substance = ['NONE', 'UNKNOWN', 'NAN', '']
        df_clean['Is_Impaired'] = np.where(df_clean['Driver Substance Abuse'].isin(safe_substance), 0, 1)
        df_clean.drop(columns=['Driver Substance Abuse'], inplace=True)

    # Step 8: Encode distraction flag
    if 'Driver Distracted By' in df_clean.columns:
        safe_distract = ['NOT DISTRACTED', 'UNKNOWN', 'NAN', '']
        df_clean['Is_Distracted'] = np.where(df_clean['Driver Distracted By'].isin(safe_distract), 0, 1)
        df_clean.drop(columns=['Driver Distracted By'], inplace=True)

    # Step 9: Group weather conditions
    if 'Weather' in df_clean.columns:
        weather_conds = [
            df_clean['Weather'].isin(['CLEAR', 'CLOUDY', 'UNKNOWN', 'NAN', '']),  
            df_clean['Weather'].isin(['RAINING', 'RAIN'])           
        ]
        df_clean['Weather_Group'] = np.select(weather_conds, ['CLEAR', 'RAIN'], default='EXTREME')
        df_clean.drop(columns=['Weather'], inplace=True)

    # Step 10: Group surface conditions
    if 'Surface Condition' in df_clean.columns:
        surface_conds = [
            df_clean['Surface Condition'].isin(['DRY', 'UNKNOWN', 'NAN', '']),    
            df_clean['Surface Condition'].isin(['WET', 'WATER(STANDING, MOVING)']) 
        ]
        df_clean['Surface_Group'] = np.select(surface_conds, ['DRY', 'WET'], default='ICE_SNOW')
        df_clean.drop(columns=['Surface Condition'], inplace=True)

    # Step 11: Fill missing values in categorical columns
    cat_cols_to_fill = ['Light', 'Traffic Control', 'Route Type', 'Collision Type', 'Vehicle Movement']
    for col in cat_cols_to_fill:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('UNKNOWN').astype(str).str.upper()

    # Step 12: Clean Speed Limit
    if 'Speed Limit' in df_clean.columns:
        df_clean['Speed Limit'] = pd.to_numeric(df_clean['Speed Limit'], errors='coerce')
        df_clean.loc[(df_clean['Speed Limit'] < 0) | (df_clean['Speed Limit'] > 150), 'Speed Limit'] = np.nan
        fallback_speed = df_clean['Speed Limit'].median() if not pd.isna(df_clean['Speed Limit'].median()) else 35
        df_clean['Speed Limit'] = df_clean['Speed Limit'].fillna(fallback_speed)

    # Step 13: Clean Vehicle Year
    if 'Vehicle Year' in df_clean.columns:
        current_year = pd.Timestamp.now().year
        df_clean['Vehicle Year'] = pd.to_numeric(df_clean['Vehicle Year'], errors='coerce')
        df_clean.loc[(df_clean['Vehicle Year'] < 1900) | (df_clean['Vehicle Year'] > current_year + 1), 'Vehicle Year'] = np.nan
        fallback_year = df_clean['Vehicle Year'].median() if not pd.isna(df_clean['Vehicle Year'].median()) else current_year
        df_clean['Vehicle Year'] = df_clean['Vehicle Year'].fillna(fallback_year)

    # Step 14: Parse datetime and extract time features
    df_clean['Crash Date/Time'] = pd.to_datetime(df_clean['Crash Date/Time'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    df_clean = df_clean.dropna(subset=['Crash Date/Time']) 

    df_clean['Hour'] = df_clean['Crash Date/Time'].dt.hour 
    df_clean['Month'] = df_clean['Crash Date/Time'].dt.month
    df_clean['Day_of_Week'] = df_clean['Crash Date/Time'].dt.day_name()
    df_clean['Year'] = df_clean['Crash Date/Time'].dt.year
    df_clean['Is_Weekend'] = df_clean['Day_of_Week'].isin(['Saturday', 'Sunday']).astype(int)

    def get_time_of_day(hour):
        if 5 <= hour < 12: return 'Morning'
        elif 12 <= hour < 18: return 'Afternoon'
        elif 18 <= hour <= 23: return 'Evening'
        else: return 'Night'
    df_clean['Time_of_Day'] = df_clean['Hour'].apply(get_time_of_day)

    # Step 15: Group vehicle body types
    if 'Vehicle Body Type' in df_clean.columns:
        def group_vehicle(vehicle_type):
            vehicle_type = str(vehicle_type).upper()
            if 'CAR' in vehicle_type or 'SEDAN' in vehicle_type: return 'Car'
            elif 'TRUCK' in vehicle_type or 'PICKUP' in vehicle_type or 'VAN' in vehicle_type: return 'Truck/Van'
            elif 'MOTORCYCLE' in vehicle_type or 'MOPED' in vehicle_type: return 'Motorcycle'
            elif 'BUS' in vehicle_type: return 'Bus'
            else: return 'Other'
        df_clean['Vehicle_Group'] = df_clean['Vehicle Body Type'].apply(group_vehicle)

    # Step 16: Clean and filter GPS coordinates
    if {'Latitude', 'Longitude'}.issubset(df_clean.columns):
        df_clean['Latitude'] = pd.to_numeric(df_clean['Latitude'], errors='coerce')
        df_clean['Longitude'] = pd.to_numeric(df_clean['Longitude'], errors='coerce')

        def restore_decimal(coord):
            if pd.isna(coord) or coord == 0:
                return np.nan
            sign = -1 if coord < 0 else 1
            val = abs(coord)
            
            while val >= 100:
                val /= 10
                
            return val * sign

        df_clean['Latitude'] = df_clean['Latitude'].apply(restore_decimal)
        df_clean['Longitude'] = df_clean['Longitude'].apply(restore_decimal)

        df_clean = df_clean.dropna(subset=['Latitude', 'Longitude'])

        LAT_MIN, LAT_MAX = 38.5, 39.8
        LON_MIN, LON_MAX = -78.0, -76.5

        df_clean = df_clean[
            df_clean['Latitude'].between(LAT_MIN, LAT_MAX) & 
            df_clean['Longitude'].between(LON_MIN, LON_MAX)
        ]    

    # Step 17: Cast categorical columns to category dtype
    cat_columns = ['Weather_Group', 'Surface_Group', 'Light', 'Traffic Control', 'Time_of_Day', 'Vehicle_Group']
    for col in cat_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype('category')

    # Step 18: Remove duplicates and save output
    df_clean = df_clean.drop_duplicates()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    
    print(f"[V] Cleaning complete. Data saved at: {output_path}")
    print(f"Dataset shape: {df_clean.shape}")
    return df_clean

if __name__ == "__main__":
    clean_traffic_data()
