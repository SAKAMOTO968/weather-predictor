import pandas as pd
from weather_predictor.features import build_features, FEATURE_COLS

def make_df() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=5),
        "temperature_max": [33.5, 33.7, 33.5, 32.1, 34.0],
        "temperature_min": [22.4, 22.0, 22.6, 23.5, 21.0],
        "precipitation":   [0.0,  0.0,  5.6,  0.0,  0.0],
        "windspeed_max":   [9.1,  8.3,  8.9,  11.1, 13.6],
    })
    
    
def test_build_features_drops_last_row():
    df = build_features(make_df())
    assert len(df) == 4 # 5 rows - 1 (no target for last row)
    
def test_temp_range_calculated_correctly():
    df = build_features(make_df())
    assert df["temp_range"].iloc[0] == 33.5 - 22.4
    
def test_will_rain_is_binary():
    df = build_features(make_df())
    assert set(df["will_rain"].unique()).issubset({0, 1})
    
def test_will_rain_correct_values():
    df = build_features(make_df())
    # วันที่ 3 (index 2) มีฝนตก 5.6mm ดังนั้น will_rain ของวันที่ 2 (index 1) = 1
    assert df["will_rain"].iloc[1] == 1
    assert df["will_rain"].iloc[0] == 0
    
def test_feature_columns_present():
    df = build_features(make_df())
    for col in FEATURE_COLS:
        assert col in df.columns, f"Missing column: {col}"
    