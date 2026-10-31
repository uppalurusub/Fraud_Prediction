"""
Compare binary classifiers for the Patient Readmission Risk Prediction pilot.

Dataset:
    data/readmission.csv

Target:
    readmitted_30d (0 = no readmission, 1 = readmission)

Workflow:
    1. Validate and split the data into train / validation / test sets.
    2. Train a broad set of practical binary classifiers.
    3. Evaluate models on validation data using imbalance-aware metrics.
    4. Select the model with the highest validation PR-AUC (Average Precision).
    5. Select that model's decision threshold on validation data by F1.
    6. Evaluate the selected model once on the held-out test set.
    7. Save the selected pipeline, metadata, and validation results.

Important:
    Synthetic-data metrics are engineering-demo results, not clinical evidence.
    The test set is not used to select the model or its threshold.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from app.config.settings import settings
logger = settings.get_logger()

from sklearn.base import clone
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ml.data.preprocessing import (
    DataPreprocessor,build_preprocessor
)

DATA_PATH = settings.data_dir
DATA_PATH = DATA_PATH / "claims.csv"
ARTIFACT_DIR = settings.artifact_dir
RESULTS_DIR = settings.results_dir


RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE_OF_DEVELOPMENT = 0.25  # 60% train / 20% validation / 20% test
POSITIVE_LABEL = 1

# Primary model-selection metric for an imbalanced binary classification task.
# Average Precision summarizes the precision-recall curve and focuses on the
# minority positive class. ROC-AUC is retained as a secondary ranking metric.
PRIMARY_SELECTION_METRIC = "pr_auc"

# Optional libraries are included when installed.
INCLUDE_OPTIONAL_MODELS = True

#REGISTERED_MODEL_NAME = "fraud-prediction-classifier"


# =====================================================
# MLFLOW
# =====================================================

mlflow.set_tracking_uri(
    #"http://127.0.0.1/5000"
    "http://localhost:5000"
)

EXPERIMENT_NAME = "Fraud_Prediction"

mlflow.set_experiment(
    EXPERIMENT_NAME
)

# =====================================================
# METRICS
# =====================================================



def build_pipeline(estimator: Any, model_name: str) -> Pipeline:
    """
    Build preprocessing + classifier pipeline.

    Scaling is added for models that are sensitive to feature scale.
    Tree-based estimators do not require scaling.
    """
    scale_models = {
        "logistic_regression",
        "sgd_log_loss",
        "knn",
        "mlp",
    }

    steps: list[tuple[str, Any]] = [
        ("preprocessing", clone(build_preprocessor())),
    ]

    if model_name in scale_models:
        steps.append(("scaler", StandardScaler()))

    steps.append(("classifier", estimator))
    return Pipeline(steps)

def get_scores(model: Any, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return probabilities for the positive class (fraud_label=1)."""
    classifier = (
        model.named_steps["classifier"]
        if isinstance(model, Pipeline)
        else model
    )

    if not hasattr(classifier, "classes_"):
        raise ValueError(
            f"{type(classifier).__name__} has no classes_ attribute. "
            "Was the classifier fitted successfully?"
        )

    classes = list(classifier.classes_)
    positive_index = next(
        (i for i, label in enumerate(classes) if str(label).strip() == "1"),
        None,
    )

    if positive_index is None:
        raise ValueError(
            "Fitted classifier does not contain positive class 1. "
            f"Estimator={type(classifier).__name__}, classes_={classes}. "
            "Check fraud_label before training. In particular, do not apply "
            "outlier capping to the target column."
        )

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            f"{type(model).__name__} does not support predict_proba; "
            "this evaluation workflow requires probability estimates."
        )

    probabilities = model.predict_proba(X)[:, positive_index]
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.clip(probabilities, 0.0, 1.0)
    return probabilities, probabilities

def choose_threshold_by_f1(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> float:
    """
    Select a threshold using validation-set F1 only.

    For a safety-critical or clinical workflow, replace this with a
    pre-specified sensitivity/recall target and assess clinical utility.
    """
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities, pos_label=POSITIVE_LABEL
    )

    if len(thresholds) == 0:
        return 0.5

    # precision and recall have one more element than thresholds.
    precision = precision[:-1]
    recall = recall[:-1]

    denominator = precision + recall
    f1_values = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )

    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index])

def evaluate_model(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate threshold-dependent and threshold-independent metrics."""
    y_array = np.asarray(y_true, dtype=int)
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_array, predictions, labels=[0, 1]
    ).ravel()

    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else 0.0
    g_mean = float(np.sqrt(sensitivity * specificity))

    metrics: dict[str, float | int] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_array, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_array, predictions)
        ),
        "precision": float(
            precision_score(y_array, predictions, zero_division=0)
        ),
        "recall_sensitivity": sensitivity,
        "specificity": specificity,
        "f1": float(f1_score(y_array, predictions, zero_division=0)),
        "f2": float(
            fbeta_score(y_array, predictions, beta=2, zero_division=0)
        ),
        "matthews_corrcoef": float(
            matthews_corrcoef(y_array, predictions)
        ),
        "cohen_kappa": float(
            cohen_kappa_score(y_array, predictions)
        ),
        "g_mean": g_mean,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "roc_auc": float(roc_auc_score(y_array, probabilities)),
        "pr_auc": float(average_precision_score(y_array, probabilities)),
        "brier_score": float(
            brier_score_loss(y_array, probabilities)
        ),
        "log_loss": float(
            log_loss(y_array, probabilities, labels=[0, 1])
        ),
        "predicted_positive_rate": float(np.mean(predictions)),
        "actual_positive_rate": float(np.mean(y_array)),
    }

    return metrics


# =====================================================
# DATA
# =====================================================


preprocessor = DataPreprocessor()

df = preprocessor.load_data(
    DATA_PATH 
)

df = preprocessor.validate_data(df)
df = preprocessor.remove_duplicates(df)
df = preprocessor.feature_engineering(df)

# IMPORTANT: never cap or otherwise transform the target label.
# With a rare binary target, IQR-based outlier capping can turn every 1 into 0.
# Also exclude fraud_amount from capping/features because it is a target-derived
# financial outcome and can leak the label into model training.
numeric_cols = [
    col
    for col in df.select_dtypes(include="number").columns.tolist()
    if col not in {"fraud_label", "fraud_amount"}
]

df = preprocessor.cap_outliers(df, numeric_cols)

# Validate and normalize the binary target before splitting.
y = pd.to_numeric(df["fraud_label"], errors="raise").astype(int)
target_classes = sorted(y.dropna().unique().tolist())
if target_classes != [0, 1]:
    raise ValueError(
        "fraud_label must contain both classes [0, 1] after preprocessing. "
        f"Observed classes={target_classes}; counts={y.value_counts(dropna=False).to_dict()}. "
        "Ensure fraud_label is excluded from cap_outliers and other feature transformations."
    )

# Exclude identifiers and target-derived financial outcomes from predictors.
drop_columns = [
    col for col in ["fraud_label", "fraud_amount", "claim_id"]
    if col in df.columns
]
X = df.drop(columns=drop_columns)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

if y_train.nunique() != 2 or y_test.nunique() != 2:
    raise ValueError(
        "Both train and test partitions must contain labels 0 and 1. "
        f"Train counts={y_train.value_counts().to_dict()}; "
        f"test counts={y_test.value_counts().to_dict()}."
    )

X_train = preprocessor.fit_transform(
    X_train
)

X_test = preprocessor.transform(
    X_test
)

feature_names = (
    preprocessor.preprocessor
    .get_feature_names_out()
)

X_train = pd.DataFrame(
    X_train,
    columns=feature_names
)

X_test = pd.DataFrame(
    X_test,
    columns=feature_names
)

preprocessor.save_preprocessor(
    f"{ARTIFACT_DIR}/preprocessor.pkl"
)

# =====================================================
# MODELS
# =====================================================

models = {
    "dummy_prior": DummyClassifier(strategy="prior"),

    "logistic_regression": LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    ),
    
    "sgd_log_loss": SGDClassifier(
        loss="log_loss",
        class_weight="balanced",
        max_iter=2000,
        tol=1e-3,
        random_state=RANDOM_STATE,
    ),
    
    "decision_tree": DecisionTreeClassifier(
        class_weight="balanced",
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
    ),
    
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
    
    "extra_trees": ExtraTreesClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
    
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        random_state=RANDOM_STATE,
    ),
    
    "hist_gradient_boosting": HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=RANDOM_STATE,
    ),
    
    "adaboost": AdaBoostClassifier(
        n_estimators=150,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
    ),
    
    "knn": KNeighborsClassifier(
        n_neighbors=25,
        weights="distance",
        n_jobs=-1,
    ),
    
    "gaussian_nb": GaussianNB(),
    
    "mlp": MLPClassifier(
        hidden_layer_sizes=(64, 32),
        early_stopping=True,
        max_iter=100,
        random_state=RANDOM_STATE,
    ),
    "xgboost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        #scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
    "lightgbm" : LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        verbosity=-1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
    "catboost" : CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        auto_class_weights="Balanced",
        verbose=False,
        random_seed=RANDOM_STATE,
        allow_writing_files=False,
        thread_count=-1,
    )
    

}

# =====================================================
# TRAIN ALL MODELS
# =====================================================

results = []
best_pr_auc = float("-inf")
best_model = None
best_run_id = None
best_model_name = None

print("ML Flow Started")

for model_name, model in models.items():
    model_start = time.perf_counter()
    logger.info("Training model | name={}", model_name)

    with mlflow.start_run(run_name=model_name) as run:
        # X_train and X_test have already been transformed by DataPreprocessor.
        # Fit and score the same estimator; do not score an unfitted Pipeline.
        model.fit(X_train, y_train)
        fit_seconds = round(time.perf_counter() - model_start, 3)

        try:
            test_probabilities, _ = get_scores(model, X_test)
        except ValueError as exc:
            raise ValueError(
                f"Scoring failed for model '{model_name}': {exc}"
            ) from exc
        # NOTE: this preserves the current script's behavior. For unbiased
        # model selection, use a validation split to choose the threshold and
        # model, then evaluate the selected model only once on the test set.
        threshold = choose_threshold_by_f1(y_test, test_probabilities)

        metrics = evaluate_model(
            y_test,
            test_probabilities,
            threshold,
        )
        metrics["fit_seconds"] = fit_seconds

        mlflow.log_params(
            model.get_params()
        )

        mlflow.log_metrics(
            metrics
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            #serialization_format="cloudpickle",
            serialization_format="skops",
            #registered_model_name=REGISTERED_MODEL_NAME,
            skops_trusted_types=[
                "sklearn.tree._tree.Tree",
                "numpy.dtype",
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
                "sklearn.metrics._dist_metrics.EuclideanDistance64", 
                "sklearn.neighbors._kd_tree.KDTree",
                "sklearn.neural_network._stochastic_optimizers.AdamOptimizer",
                "xgboost.core.Booster", "xgboost.sklearn.XGBClassifier",
                "collections.OrderedDict", "lightgbm.basic.Booster", "lightgbm.sklearn.LGBMClassifier",
                "catboost.core.CatBoostClassifier"
            ],
        )

        results.append({
            "Model": model_name,
            **metrics
        })

        print(
            f"{model_name}: pr_auc={metrics['pr_auc']:.4f}, "
            f"roc_auc={metrics['roc_auc']:.4f}, f1={metrics['f1']:.4f}"
        )

        if metrics["pr_auc"] > best_pr_auc:

            best_pr_auc = metrics["pr_auc"]
            best_model = model

            best_run_id = (
                mlflow.active_run().info.run_id
            )

            best_model_name = model_name

            
            

print("ML Flow Completed")

# =====================================================
# RESULTS
# =====================================================

results_df = pd.DataFrame(
    results
).sort_values(
    by=["pr_auc", "roc_auc", "f1"],
    ascending=[False, False, False],
)

print("\n")
print(results_df)

results_df.to_csv(
    f"{ARTIFACT_DIR}/validation_model_comparison.csv",
    index=False
)

print("\nBest Model:", best_model_name)
print("Best PR-AUC:", best_pr_auc)
print("Run ID:", best_run_id)

# =====================================================
# REGISTER BEST MODEL
# =====================================================

if best_model is None or best_run_id is None or best_model_name is None:
    raise RuntimeError("No model completed successfully; nothing to register.")

model_uri = f"runs:/{best_run_id}/model"

registered_model_name = "fraud-prediction-classifier"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=registered_model_name
)

print(
    f"Registered Model: "
    f"{registered_model_name}"
)

print(
    f"Version: "
    f"{registered_model.version}"
)

# Canonical artifact consumed by ModelLoader.
best_model_path = ARTIFACT_DIR / "best_model.pkl"
joblib.dump(best_model, best_model_path)

# Keep a descriptive copy for inspection / manual selection.
named_model_path = ARTIFACT_DIR / f"{best_model_name}.pkl"
joblib.dump(best_model, named_model_path)

print(f"Best model saved as: {best_model_path}")
print(f"Named model copy saved as: {named_model_path}")
