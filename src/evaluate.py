"""Evaluate the saved model and write diagnostic plots."""
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.preprocessing import load_and_split_data  # noqa: E402


def evaluate_model(data_path: str | Path | None = None, model_path: str | Path | None = None) -> dict:
    data_path = Path(data_path or PROJECT_ROOT / "data" / "student_data.csv")
    model_path = Path(model_path or PROJECT_ROOT / "models" / "best_model.pkl")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    _, X_test, _, y_test = load_and_split_data(data_path, bundle.get("random_state", 42))
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, zero_division=0),
    }

    output_dir = PROJECT_ROOT / "models"
    output_dir.mkdir(exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"{bundle['model_name']} (AUC={metrics['roc_auc']:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Dropout prediction ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(5, 4))
    sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    for key, value in metrics.items():
        if key != "classification_report":
            print(f"{key}: {value}")
    print(metrics["classification_report"])
    return metrics


if __name__ == "__main__":
    evaluate_model()
