"""Train candidate classifiers with SMOTE and save the best model bundle."""
from pathlib import Path
import sys

import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import FEATURES, load_and_split_data, build_preprocessor  # noqa: E402

RANDOM_STATE = 42


def train_models(data_path: str | Path | None = None) -> dict:
    data_path = Path(data_path or PROJECT_ROOT / "data" / "student_data.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run data_generation.py first.")

    X_train, X_test, y_train, y_test = load_and_split_data(data_path, RANDOM_STATE)
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight=None, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=180, max_depth=3),
    }

    results = {}
    fitted_models = {}
    for name, estimator in candidates.items():
        pipeline = ImbPipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        results[name] = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
        fitted_models[name] = pipeline

    best_name = max(results, key=results.get)
    output_dir = PROJECT_ROOT / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": fitted_models[best_name],
        "model_name": best_name,
        "features": FEATURES,
        "random_state": RANDOM_STATE,
        "validation_roc_auc": results[best_name],
    }
    joblib.dump(bundle, output_dir / "best_model.pkl")
    print("Validation ROC-AUC by model:")
    for name, score in results.items():
        print(f"  {name}: {score:.4f}")
    print(f"Saved best model ({best_name}) to {output_dir / 'best_model.pkl'}")
    return bundle


if __name__ == "__main__":
    train_models()
