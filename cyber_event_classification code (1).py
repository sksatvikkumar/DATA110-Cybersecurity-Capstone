"""
Cyber Event Classification Using Machine Learning Techniques
Based on the project methodology documented in the capstone presentation.

Models:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Artificial Neural Network (MLP)

Expected target column:
    attack_type

Expected dataset columns may include:
    attacker_ip, target_ip, target_system, outcome,
    security_tools_used, user_role, location, industry,
    mitigation_method, data_compromised_GB,
    attack_duration_min, attack_severity,
    response_time_min, timestamp, attack_type

The loader can read CSV or Excel files from the data/ folder.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET = "attack_type"
TEST_SIZE = 0.20


def load_dataset():
    """Load the first CSV/XLSX file found in data/."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    csv_files = sorted(data_dir.glob("*.csv"))
    excel_files = sorted(data_dir.glob("*.xlsx"))

    if csv_files:
        path = csv_files[0]
        df = pd.read_csv(path)
    elif excel_files:
        path = excel_files[0]
        df = pd.read_excel(path)
    else:
        raise FileNotFoundError(
            "No dataset found. Put your dataset inside the 'data' folder "
            "as a CSV or XLSX file."
        )

    print(f"Loaded dataset: {path}")
    print(f"Shape: {df.shape}")
    return df


def clean_and_engineer_features(df):
    """Perform the cleaning and feature engineering described in the project."""
    df = df.copy()

    # Standardize column names.
    df.columns = [str(c).strip() for c in df.columns]

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' was not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Remove completely duplicated rows.
    df = df.drop_duplicates().reset_index(drop=True)

    # Trim whitespace in text columns.
    text_columns = df.select_dtypes(include=["object", "string"]).columns
    for col in text_columns:
        df[col] = df[col].astype("string").str.strip()

    # Parse timestamp and create temporal features.
    if "timestamp" in df.columns:
        dt = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour"] = dt.dt.hour
        df["day_of_week"] = dt.dt.dayofweek
        df["month"] = dt.dt.month
        df["is_business_hours"] = (
            (df["hour"] >= 9) & (df["hour"] < 17)
        ).astype(int)
        df = df.drop(columns=["timestamp"])

    # Remove near-unique identifiers to avoid memorization.
    identifier_columns = [
        col for col in ["attacker_ip", "target_ip"] if col in df.columns
    ]
    if identifier_columns:
        df = df.drop(columns=identifier_columns)

    # Remove rows with missing target.
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    return df


def build_preprocessor(X):
    """Create numeric/categorical preprocessing."""
    numeric_features = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def evaluate_model(name, model, X_test, y_test, label_encoder):
    """Calculate the project's evaluation metrics."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    try:
        roc_auc = roc_auc_score(
            y_test,
            probabilities,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        roc_auc = np.nan

    print(f"\n{name}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
    }


def main():
    # ------------------------------------------------------------
    # 1. LOAD DATA
    # ------------------------------------------------------------
    df = load_dataset()
    print("\nFirst five rows:")
    print(df.head())

    # ------------------------------------------------------------
    # 2. CLEANING + FEATURE ENGINEERING
    # ------------------------------------------------------------
    df = clean_and_engineer_features(df)

    print("\nCleaned dataset shape:", df.shape)
    print("\nMissing values:")
    print(df.isnull().sum().sum())

    print("\nTarget distribution:")
    print(df[TARGET].value_counts())

    # ------------------------------------------------------------
    # 3. PREPARE X AND y
    # ------------------------------------------------------------
    X = df.drop(columns=[TARGET])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET])

    # Stratified split keeps the class proportions similar.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTraining rows:", len(X_train))
    print("Testing rows :", len(X_test))

    preprocessor = build_preprocessor(X_train)

    # ------------------------------------------------------------
    # 4. DEFINE FOUR MODELS
    # ------------------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "ANN": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=100,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
    }

    # ------------------------------------------------------------
    # 5. TRAIN BASE MODELS
    # ------------------------------------------------------------
    results = []
    trained_models = {}

    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)

        results.append(
            evaluate_model(
                name,
                pipeline,
                X_test,
                y_test,
                label_encoder,
            )
        )
        trained_models[name] = pipeline

    # ------------------------------------------------------------
    # 6. HYPERPARAMETER TUNING
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING")
    print("=" * 70)

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    rf_grid = {
        "model__n_estimators": [100, 150],
        "model__max_depth": [10, 20],
    }

    rf_search = GridSearchCV(
        rf_pipeline,
        rf_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
    )

    rf_search.fit(X_train, y_train)

    print("\nBest Random Forest parameters:")
    print(rf_search.best_params_)
    print(f"Best CV accuracy: {rf_search.best_score_:.4f}")

    dt_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]
    )

    dt_grid = {
        "model__max_depth": [5, 10, 15, 20],
        "model__min_samples_split": [2, 5, 10],
    }

    dt_search = GridSearchCV(
        dt_pipeline,
        dt_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
    )

    dt_search.fit(X_train, y_train)

    print("\nBest Decision Tree parameters:")
    print(dt_search.best_params_)
    print(f"Best CV accuracy: {dt_search.best_score_:.4f}")

    # ------------------------------------------------------------
    # 7. EVALUATE TUNED MODELS
    # ------------------------------------------------------------
    results.append(
        evaluate_model(
            "Tuned Random Forest",
            rf_search.best_estimator_,
            X_test,
            y_test,
            label_encoder,
        )
    )

    results.append(
        evaluate_model(
            "Tuned Decision Tree",
            dt_search.best_estimator_,
            X_test,
            y_test,
            label_encoder,
        )
    )

    # ------------------------------------------------------------
    # 8. SAVE RESULTS
    # ------------------------------------------------------------
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="Accuracy", ascending=False
    )

    results_df.to_csv(
        results_dir / "model_comparison.csv",
        index=False,
    )

    print("\nFinal model comparison:")
    print(results_df.to_string(index=False))

    # ------------------------------------------------------------
    # 9. SAVE BEST MODEL
    # ------------------------------------------------------------
    best_name = results_df.iloc[0]["Model"]

    if best_name == "Tuned Random Forest":
        best_model = rf_search.best_estimator_
    elif best_name == "Tuned Decision Tree":
        best_model = dt_search.best_estimator_
    else:
        best_model = trained_models[best_name]

    print(f"\nBest model: {best_name}")

    try:
        import joblib

        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        joblib.dump(
            {
                "model": best_model,
                "label_encoder": label_encoder,
            },
            models_dir / "best_model.pkl",
        )

        print("Saved: models/best_model.pkl")
    except ImportError:
        print(
            "joblib is not installed, so the model was not saved. "
            "Install joblib and run again."
        )


if __name__ == "__main__":
    main()
