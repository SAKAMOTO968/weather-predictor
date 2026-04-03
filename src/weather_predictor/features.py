import pandas as pd
import numpy as np
from pathlib import Path

def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load CSV and parse dates."""
    df = pd.read_csv(filepath, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features and targets For ML.
    
    Features (วันนี้):
    - temperature_max, temperature_min, precipitation, windspeed_max
    - temp_range: ช่วงอุณหภูมิ (max - min)
    - month: เดือน (ใช้จับ seasonality)
    
    Target (พรุ่งนี้):
    - next_temp_max: อุณหภูมิสูงสุดของวันพรุ่งนี้ (regression)
    - will_rain: จะมีฝนพรุ่งนี้ไหม 0/1 (classification)
    """
    df = df.copy()
    
    # Features engineering
    df["temp_range"] = df["temperature_max"] - df["temperature_min"]
    df["month"] = df["date"].dt.month
    
    # shift target - เอาค่าแถวถัดไปเป็น target ของแถวนี้
    df["next_temp_max"] = df["temperature_max"].shift(-1)
    df["will_rain"] = (df["precipitation"].shift(-1) > 0).astype(int)
    
    # ตัดแถวสุดท้ายออก เพราะไม่มี target
    df = df.dropna(subset=["next_temp_max",]).reset_index(drop=True)
    
    return df

FEATURE_COLS = [
    "temperature_max",
    "temperature_min",
    "precipitation",
    "windspeed_max",
    "temp_range",
    "month",
]

TARGET_REGRESSION = "next_temp_max"
TARGET_CLASSIFICATION = "will_rain"