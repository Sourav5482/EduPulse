"""Package entry point for generating the project's synthetic dataset."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_generation import generate_student_data  # noqa: E402


def main() -> None:
    output_path = PROJECT_ROOT / "data" / "student_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = generate_student_data()
    data.to_csv(output_path, index=False)
    print(f"Saved {len(data):,} students to {output_path}")
    print(f"Dropout rate: {data['dropout'].mean():.1%}")


if __name__ == "__main__":
    main()
