"""Feature preparation shared by training, evaluation, and the app."""
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "dropout"
ID_COLUMN = "student_id"
NUMERIC_FEATURES = [
    "age",
    "attendance_rate",
    "avg_grade",
    "assignment_submission_rate",
    "lms_logins_per_week",
    "participation_score",
    "disciplinary_incidents",
    "financial_aid",
    "first_generation",
    "distance_from_school_km",
    "previous_failures",
]
CATEGORICAL_FEATURES = ["gender", "socioeconomic_status"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Build a fitted-later transformer with robust handling for missing values."""
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def load_and_split_data(data_path: str | Path, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    data = pd.read_csv(data_path)
    missing_columns = set(FEATURES + [TARGET]) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")
    X = data[FEATURES]
    y = data[TARGET].astype(int)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
