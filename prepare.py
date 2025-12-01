import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def prepare_dataset():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load raw dataset
    df = pd.read_csv(config["paths"]["raw_data"])

    # Handle Age (mean or median)
    if "Age" in df.columns:
        strategy = config["preprocessing"]["fill_age_with"]
        if strategy == "median":
            df["Age"].fillna(df["Age"].median(), inplace=True)
        elif strategy == "mean":
            df["Age"].fillna(df["Age"].mean(), inplace=True)

    # Drop unused columns
    drop_cols = config["preprocessing"]["drop_columns"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Encode categorical features
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = df[col].astype("category").cat.codes

    # Separate input and output
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"]   # ✔ FIXED
    )

    # Feature scaling
    if config["preprocessing"]["scale_features"]:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Save processed data
    np.save(config["paths"]["X_train"], X_train)
    np.save(config["paths"]["X_test"], X_test)
    np.save(config["paths"]["y_train"], y_train.values)
    np.save(config["paths"]["y_test"], y_test.values)

    print("Dataset preparation done!")

if __name__ == "__main__":
    prepare_dataset()
