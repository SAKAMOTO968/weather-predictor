import argparse
from pathlib import Path
from .train import train
from .features import load_data, build_features, FEATURE_COLS
import joblib

def main() -> None:
    parser = argparse.ArgumentParser(description="Weather Predictor")
    subparsers = parser.add_subparsers(dest="command")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("data", help="Path to CSV file")
    train_parser.add_argument("--model-dir", default="models")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict tomorrow")
    predict_parser.add_argument("data", help="Path to CSV file")
    predict_parser.add_argument("--model-dir", default="models")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(args.data, args.model_dir)
        
    elif args.command == "predict":
        df = load_data(args.data)
        df = build_features(df)
        last = df[FEATURE_COLS].iloc[-1:] # เอาแค่แถวสุดท้าย (ข้อมูลวันนี้)
        
        reg_model = joblib.load(Path(args.model_dir) / "temp_model.pkl")
        cls_model = joblib.load(Path(args.model_dir) / "rain_model.pkl")
        
        temp_pred = reg_model.predict(last)[0]
        rain_pred = cls_model.predict(last)[0]
        rain_prob = cls_model.predict_proba(last)[0][1]
        
        print(f"\nPrediction for tomorrow")
        print(f" Temperature max: {temp_pred:.1f}°C")
        print(f" Rain: {'Yes' if rain_pred else 'No'} (probability: {rain_prob:.1%})")
        
    else:
        parser.print_help()
        
if __name__ == "__main__":
    main()