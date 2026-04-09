import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error, accuracy_score,
    precision_score, recall_score, f1_score,
    classification_report,
)
import joblib
import numpy as np
from pathlib import Path
from .features import (
    load_data, build_features,
    FEATURE_COLS, TARGET_REGRESSION, TARGET_CLASSIFICATION,
)


def evaluate_regression(model, X, y, n_splits: int = 4) -> dict:
    """Cross-validate regression model ด้วย TimeSeriesSplit."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, model.predict(X_test))
        maes.append(mae)
        print(f"  Fold {fold}: MAE = {mae:.2f}°C")
    print(f"  Average MAE: {np.mean(maes):.2f}°C ± {np.std(maes):.2f}")
    return {"mae_mean": np.mean(maes), "mae_std": np.std(maes)}


def evaluate_classification(model, X, y, n_splits: int = 4) -> dict:
    """Cross-validate classification model พร้อม precision/recall."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    f1s, precisions, recalls = [], [], []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1  = f1_score(y_test, y_pred, zero_division=0)
        pre = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1s.append(f1)
        precisions.append(pre)
        recalls.append(rec)
        print(f"  Fold {fold}: F1={f1:.2f}  Precision={pre:.2f}  Recall={rec:.2f}")
    print(f"  Average F1: {np.mean(f1s):.2f} ± {np.std(f1s):.2f}")
    return {"f1_mean": np.mean(f1s), "precision_mean": np.mean(precisions), "recall_mean": np.mean(recalls)}


def train(data_path: str | Path, model_dir: str | Path = "models") -> None:
    Path(model_dir).mkdir(exist_ok=True)

    df_raw = load_data(data_path)
    df = build_features(df_raw)

    X = df[FEATURE_COLS]

    # ตรวจ class imbalance ก่อน
    rain_days = df[TARGET_CLASSIFICATION].sum()
    total = len(df)
    print(f"Class distribution: มีฝน {rain_days} วัน ({rain_days/total:.1%}) | ไม่มีฝน {total-rain_days} วัน ({(total-rain_days)/total:.1%})")
    print()

    # Regression
    print("Regression — ทำนายอุณหภูมิสูงสุดพรุ่งนี้:")
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    evaluate_regression(reg_model, X, df[TARGET_REGRESSION])

    # train ด้วยข้อมูลทั้งหมดก่อน save
    reg_model.fit(X, df[TARGET_REGRESSION])
    joblib.dump(reg_model, Path(model_dir) / "temp_model.pkl")

    print()

    # Classification พร้อมแก้ class imbalance ด้วย class_weight
    print("Classification — ทำนายว่าจะมีฝนไหม:")
    cls_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",  # แก้ class imbalance อัตโนมัติ
    )
    evaluate_classification(cls_model, X, df[TARGET_CLASSIFICATION])

    cls_model.fit(X, df[TARGET_CLASSIFICATION])
    joblib.dump(cls_model, Path(model_dir) / "rain_model.pkl")

    print(f"\nModels saved to {model_dir}/")

    # Feature importance
    print("\nTop features (temperature model):")
    importance = pd.Series(
        reg_model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)
    for feat, score in importance.items():
        print(f"  {feat}: {score:.3f}")