"""SHAP-based global and individual explanations for the saved model."""
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.preprocessing import FEATURES  # noqa: E402


def explain_model(data_path: str | Path | None = None, model_path: str | Path | None = None, student_index: int = 0):
    data_path = Path(data_path or PROJECT_ROOT / "data" / "student_data.csv")
    model_path = Path(model_path or PROJECT_ROOT / "models" / "best_model.pkl")
    bundle = joblib.load(model_path)
    pipeline = bundle["model"]
    data = pd.read_csv(data_path)
    X = data[FEATURES]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()

    # TreeExplainer is fast for the tree candidates; the generic explainer covers logistic regression.
    if classifier.__class__.__name__ in {"RandomForestClassifier", "GradientBoostingClassifier"}:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
    else:
        background = shap.sample(transformed, min(100, len(transformed)), random_state=42)
        explainer = shap.Explainer(classifier.predict_proba, background, feature_names=feature_names)
        explanation = explainer(transformed[: min(200, len(transformed))])
        shap_values = explanation.values[:, :, 1]
        expected_value = explanation.base_values[:, 1].mean()

    importance = pd.DataFrame({"feature": feature_names, "mean_abs_shap": abs(shap_values).mean(axis=0)})
    importance = importance.sort_values("mean_abs_shap", ascending=False)
    output_dir = PROJECT_ROOT / "models"
    output_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, transformed, feature_names=feature_names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()

    student_index = min(max(student_index, 0), len(X) - 1)
    individual = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": shap_values[student_index],
            "absolute_shap": abs(shap_values[student_index]),
        }
    ).sort_values("absolute_shap", ascending=False)
    return importance, individual, expected_value


if __name__ == "__main__":
    global_importance, individual_explanation, _ = explain_model()
    print(global_importance.head(15).to_string(index=False))
    print("\nTop factors for first student:")
    print(individual_explanation.head(10).to_string(index=False))
