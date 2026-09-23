"""Generate a realistic synthetic student dropout dataset."""
from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_STATE = 42


def generate_student_data(n_students: int = 5000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Create synthetic student records with a noisy dropout mechanism."""
    rng = np.random.default_rng(random_state)

    socioeconomic_status = rng.choice(["Low", "Medium", "High"], n_students, p=[0.30, 0.50, 0.20])
    gender = rng.choice(["Female", "Male", "Non-binary"], n_students, p=[0.48, 0.48, 0.04])
    age = np.clip(np.round(rng.normal(20.5, 2.1, n_students)), 17, 30).astype(int)
    financial_aid = rng.binomial(1, np.where(socioeconomic_status == "Low", 0.72, 0.34))
    first_generation = rng.binomial(1, np.where(socioeconomic_status == "Low", 0.58, 0.24))

    status_effect = pd.Series(socioeconomic_status).map({"Low": -8, "Medium": 0, "High": 5}).to_numpy()
    attendance_rate = np.clip(rng.normal(82 + status_effect * 0.25, 11, n_students), 35, 100)
    avg_grade = np.clip(0.48 * attendance_rate + rng.normal(35 + status_effect * 0.35, 10, n_students), 25, 100)
    assignment_submission_rate = np.clip(
        (avg_grade / 115) + rng.normal(0.08, 0.12, n_students), 0.05, 1.0
    )
    lms_logins_per_week = np.clip(
        np.round(2.5 + attendance_rate / 18 + assignment_submission_rate * 3 + rng.normal(0, 1.8, n_students)),
        0,
        25,
    )
    participation_score = np.clip(
        0.045 * attendance_rate + 3.2 * assignment_submission_rate + rng.normal(-1.2, 1.4, n_students),
        0,
        10,
    )
    disciplinary_incidents = np.clip(
        rng.poisson(np.clip(1.8 - attendance_rate / 125 + (socioeconomic_status == "Low") * 0.35, 0.1, 2.0)),
        0,
        8,
    )
    distance_from_school_km = np.clip(rng.gamma(2.1, 4.0, n_students), 0.2, 35)
    previous_failures = np.clip(
        rng.poisson(np.clip(1.1 + (100 - avg_grade) / 55, 0.1, 4.0)), 0, 6
    )

    # The target is probabilistic: adverse signals raise risk, but noise prevents perfection.
    logit = (
        -4.0
        + 0.055 * (65 - attendance_rate)
        + 0.045 * (62 - avg_grade)
        + 1.7 * (0.62 - assignment_submission_rate)
        + 0.10 * (5 - lms_logins_per_week)
        + 0.14 * (4.5 - participation_score)
        + 0.24 * disciplinary_incidents
        + 0.35 * (1 - financial_aid)
        + 0.34 * first_generation
        + 0.035 * (distance_from_school_km - 8)
        + 0.62 * previous_failures
        + rng.normal(0, 0.75, n_students)
    )
    dropout_probability = 1 / (1 + np.exp(-logit))
    dropout = rng.binomial(1, dropout_probability)

    return pd.DataFrame(
        {
            "student_id": [f"STU-{i:05d}" for i in range(1, n_students + 1)],
            "age": age,
            "gender": gender,
            "socioeconomic_status": socioeconomic_status,
            "attendance_rate": np.round(attendance_rate, 2),
            "avg_grade": np.round(avg_grade, 2),
            "assignment_submission_rate": np.round(assignment_submission_rate, 3),
            "lms_logins_per_week": lms_logins_per_week.astype(int),
            "participation_score": np.round(participation_score, 2),
            "disciplinary_incidents": disciplinary_incidents.astype(int),
            "financial_aid": financial_aid,
            "first_generation": first_generation,
            "distance_from_school_km": np.round(distance_from_school_km, 2),
            "previous_failures": previous_failures.astype(int),
            "dropout": dropout,
        }
    )


def main() -> None:
    output_path = Path(__file__).resolve().parent / "data" / "student_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = generate_student_data()
    data.to_csv(output_path, index=False)
    print(f"Saved {len(data):,} students to {output_path}")
    print(f"Dropout rate: {data['dropout'].mean():.1%}")


if __name__ == "__main__":
    main()
