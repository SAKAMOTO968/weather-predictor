import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import joblib
from pathlib import Path
from .features import load_data, build_features, FEATURE_COLS, TARGET_REGRESSION, TARGET_CLASSIFICATION

def train(data_path: str | Path, model_dir: str | Path = "models") -> None:
    """
    Train regression and classification models.
    
    Args:
    data_path: Path to the CSV file.
    model_dir: Directory to save trained models
    """
    Path(model_dir).mkdir(exist_ok=True)
    
    # โหลดและ build features
    df_raw = load_data(data_path)
    df = build_features(df_raw)
    
    x = df[FEATURE_COLS]
    
    # 1. Regression ทำนายอุณหภูมิสูงสุดของวันพรุ่งนี้
    y_reg = df[TARGET_REGRESSION]
    x_train, x_test, y_train, y_test = train_test_split(x, y_reg, test_size=0.2, shuffle=False) # shuffle=False เพราะเป็น time series
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(x_train, y_train)
    y_pred_reg = reg_model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred_reg)
    print(f"Regression MAE: {mae:.2f}°C")
    
    # 2. Classification ทำนายว่าจะมีฝนตกวันพรุ่งนี้ไหม
    y_cls = df[TARGET_CLASSIFICATION]
    x_train, x_test, y_train, y_test = train_test_split(x, y_cls, test_size=0.2, shuffle=False)
    cls_model = RandomForestClassifier(n_estimators=100, random_state=42)
    cls_model.fit(x_train, y_train)
    y_pred_cls = cls_model.predict(x_test)
    acc = accuracy_score(y_test, y_pred_cls)
    print(f"Classification Accuracy: {acc:.1%}")
    
    # บันทึกโมเดล
    joblib.dump(reg_model, Path(model_dir) / "temp_model.pkl")
    joblib.dump(cls_model, Path(model_dir) / "rain_model.pkl")
    print(f"Models saved to {model_dir}")
    
    # แสดง feature importance
    print("\nTop features (temperature model):")
    importance = pd.Series(
        reg_model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)
    for feat, score in importance.items():
        print(f" {feat}: {score:.3f}")
    