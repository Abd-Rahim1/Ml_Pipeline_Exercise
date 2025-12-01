import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_model(name, params):
    if name == "LogisticRegression":
        return LogisticRegression(**params)
    if name == "DecisionTree":
        return DecisionTreeClassifier(**params)
    if name == "SVM":
        return SVC(**params)
    
    raise ValueError(f"Unknown model: {name}")

def train_and_evaluate():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load prepared data
    X_train = np.load(config["paths"]["X_train"])
    X_test = np.load(config["paths"]["X_test"])
    y_train = np.load(config["paths"]["y_train"])
    y_test = np.load(config["paths"]["y_test"])

    results = {}

    for model_name, params in config["models"].items():
        model = load_model(model_name, params)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[model_name] = {}

        if "accuracy" in config["metrics"]["compute"]:
            results[model_name]["accuracy"] = accuracy_score(y_test, y_pred)
        if "precision" in config["metrics"]["compute"]:
            results[model_name]["precision"] = precision_score(y_test, y_pred)
        if "recall" in config["metrics"]["compute"]:
            results[model_name]["recall"] = recall_score(y_test, y_pred)
        if "f1" in config["metrics"]["compute"]:
            results[model_name]["f1"] = f1_score(y_test, y_pred)
        if "confusion_matrix" in config["metrics"]["compute"]:
            results[model_name]["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()

        print(f"\n{model_name} Results:", results[model_name])

    # Select best model by accuracy
    best_model = max(results, key=lambda x: results[x].get("accuracy", 0))

    # Save metrics
    with open(config["paths"]["metrics_output"], "w") as f:
        f.write("MODEL EVALUATION METRICS\n")
        f.write("="*40 + "\n\n")

        for name, metrics in results.items():
            f.write(f"{name}:\n")
            for m, v in metrics.items():
                f.write(f"  {m}: {v}\n")
            f.write("\n")

        f.write(f"BEST MODEL: {best_model}\n")

    print(f"\nMetrics saved to {config['paths']['metrics_output']}")
    print(f"Best Model: {best_model}")

if __name__ == "__main__":
    train_and_evaluate()
