# Student Dropout Prediction & Intervention

An end-to-end, support-oriented machine learning project that identifies students who may benefit from earlier outreach and recommends human-led interventions. The included dataset is synthetic and contains 5,000 student records.

## Project structure

```text
data/                  Generated CSV and model diagnostics
notebooks/              Space for exploratory notebooks
src/
  data_generation.py    Synthetic data generator
  preprocessing.py      Shared preprocessing and split logic
  train.py              SMOTE training and model selection
  evaluate.py           Metrics and diagnostic plots
  explain.py            SHAP global and local explanations
app/
  app.py                Streamlit dashboard
models/
  best_model.pkl        Saved preprocessing + classifier bundle
requirements.txt
README.md
```

## Setup

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the pipeline

From the project root:

```bash
python data_generation.py
python -m src.train
python -m src.evaluate
python -m src.explain
streamlit run app/app.py
```

The dashboard opens at the local Streamlit URL. The training script compares Logistic Regression, Random Forest, and Gradient Boosting using stratified validation data and SMOTE inside each training pipeline. The highest validation ROC-AUC model is saved to `models/best_model.pkl`.

## Modeling notes

- Numeric features use median imputation and standard scaling.
- Categorical features use most-frequent imputation and one-hot encoding.
- `student_id` is excluded from modeling.
- SMOTE is applied only to the training fold to avoid test contamination.
- Evaluation reports accuracy, precision, recall, F1, ROC-AUC, a confusion matrix, a classification report, and ROC/confusion-matrix plots.
- SHAP provides global importance and individual explanations; transformed categorical feature names are preserved.

## Ethical use

This system is a decision-support aid, not a verdict. It must not be used for punishment, exclusion, grading, or automatic high-stakes decisions. Teachers and counselors should verify context with each student, obtain consent for support where appropriate, protect sensitive information, and monitor error rates and outcomes across gender and socioeconomic groups before using it operationally. Synthetic data is for demonstration only and is not evidence about any real student population.
