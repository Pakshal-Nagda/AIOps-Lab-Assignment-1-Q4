import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=3)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--data_path", type=str, default="data")
    args = parser.parse_args()

    mlflow.set_tracking_uri("http://localhost:5000")

    X_train = pd.read_csv(os.path.join(args.data_path, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(args.data_path, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(args.data_path, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(args.data_path, "y_test.csv")).values.ravel()

    with mlflow.start_run(run_name=f"project-run-n{args.n_estimators}-d{args.max_depth}"):
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)

        model = RandomForestClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        mlflow.sklearn.log_model(model, name="model")

        print(f"accuracy={acc:.4f}  f1_macro={f1:.4f}  run_id={mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
