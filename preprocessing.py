import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import argparse
class DatasetPreprocessor:
    def __init__(self, output_dir="preprocessed_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_dataset(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")
        if path.endswith('.csv'):
            return pd.read_csv(path)
        elif path.endswith('.json'):
            return pd.read_json(path)
        elif path.endswith(('.xlsx', '.xls')):
            return pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported format: {path}")

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        # Handle numeric columns based on the selected strategy
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        strategies = {
            'mean': lambda x: x.fillna(x.mean(numeric_only=True)),
            'median': lambda x: x.fillna(x.median(numeric_only=True)),
            'mode': lambda x: x.fillna(x.mode().iloc[0]),
            'interpolate': lambda x: x.interpolate()
        }
        if strategy not in strategies:
            raise ValueError(f"Invalid strategy. Choose from {list(strategies.keys())}")
        df[numeric_cols] = strategies[strategy](df[numeric_cols])

        # Fill missing values in string columns with their mode
        string_cols = df.select_dtypes(include=['object', 'string']).columns
        for col in string_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'unknown'
                df[col] = df[col].fillna(mode_val)

        # Also ensure 'gender' columns are explicitly handled
        for col in df.columns:
            if 'gender' in col.lower():
                df[col] = df[col].fillna('unknown')

        return df

    def clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                df[col] = df[col].clip(lower=0).round().astype(int)
            elif pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].astype(str).str.strip().str.lower().replace({'nan': 'unknown', '': 'unknown'}).fillna('unknown')
                if 'gender' in col.lower():
                    df[col] = df[col].replace({
                        'm': 'male', 'male': 'male',
                        'f': 'female', 'female': 'female'
                    }).fillna('unknown')
        return df

    def scale_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if numerical_cols.empty:
            return df
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        return df

    def split_dataset(self, df: pd.DataFrame, test_size: float = 0.2):
        return train_test_split(df, test_size=test_size, random_state=42)

    def preprocess_and_save(self, input_path: str, output_prefix: str = "preprocessed",
                            output_dir: str = "preprocessed_data",
                            missing_strategy: str = 'mean', scaling_method: str = 'standard',
                            test_size: float = 0.2):
        try:
            print(f"📂 Loading dataset from: {input_path}")
            df = self.load_dataset(input_path)

            print("🧼 Handling missing values...")
            df = self.handle_missing_values(df, missing_strategy)

            print("✨ Cleaning and validating columns...")
            df = self.clean_columns(df)

            print("📊 Scaling numeric features...")
            df = self.scale_features(df, scaling_method)

            print("🔀 Splitting dataset...")
            train_df, test_df = self.split_dataset(df, test_size)

            os.makedirs(output_dir, exist_ok=True)
            train_path = os.path.join(output_dir, f"{output_prefix}_train.csv")
            test_path = os.path.join(output_dir, f"{output_prefix}_test.csv")

            print(f"💾 Saving training data to {train_path}")
            train_df.to_csv(train_path, index=False)
            print(f"💾 Saving testing data to {test_path}")
            test_df.to_csv(test_path, index=False)

            print("✅ Preprocessing complete!")
            return train_df, test_df

        except Exception as e:
            print(f"❌ Error: Preprocessing failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Dataset Preprocessor CLI")
    parser.add_argument("input_path", nargs='?', help="Path to the input dataset")
    parser.add_argument("--output_prefix", default=None, help="Output filename prefix")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--missing_strategy", choices=['mean', 'median', 'mode', 'interpolate'], default='mean')
    parser.add_argument("--scaling_method", choices=['standard', 'minmax'], default='standard')
    parser.add_argument("--test_size", type=float, default=0.2)

    args = parser.parse_args()

    # 🔄 Interactive prompts if values are missing
    if not args.input_path:
        args.input_path = input("📎 Enter the input dataset path: ").strip()

    if not args.output_prefix:
        args.output_prefix = input("📤 Enter output prefix [default = preprocessed]: ").strip() or "preprocessed"

    if not args.output_dir:
        args.output_dir = input("📁 Enter output directory [default = preprocessed_data]: ").strip() or "preprocessed_data"

    preprocessor = DatasetPreprocessor(output_dir=args.output_dir)
    preprocessor.preprocess_and_save(
        input_path=args.input_path,
        output_prefix=args.output_prefix,
        output_dir=args.output_dir,
        missing_strategy=args.missing_strategy,
        scaling_method=args.scaling_method,
        test_size=args.test_size
    )

if __name__ == "__main__":
    main()
