"""
preprocess.py
"""

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder,
    RobustScaler
)
from app.config.settings import settings
logger = settings.get_logger()

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

NUMERIC_FEATURES = [
    "claim_amount",
    "claim_age_days",
    "prior_claim_count",
    "duplicate_claim_flag",
]

CATEGORICAL_FEATURES = [
    "provider_type",
    "diagnosis_code",
    "procedure_code",
]

ID_COLUMN = "claim_id"
#EXCLUDED_COLUMNS = {"fraud_amount"}

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_COLUMN = "fraud_label"

# ---------------------------------------------------------
# Preprocessing pipeline factory
# ---------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """
    Create an unfitted preprocessing transformer.

    Numeric features:
        Median imputation.

    Categorical features:
        Most-frequent imputation and one-hot encoding.

    The returned transformer must be fitted only on training data.
    """

    logger.info(
        "Building preprocessing pipeline | "
        "numeric_features={} | categorical_features={}",
        len(NUMERIC_FEATURES),
        len(CATEGORICAL_FEATURES),
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    logger.success(
        "Preprocessing pipeline created successfully."
    )

    return preprocessor


class DataPreprocessor:

    def __init__(self):

        self.preprocessor = None

        self.numeric_columns = []

        self.categorical_columns = []

    ###################################################
    # Load Data
    ###################################################

    def load_data(
        self,
        file_path: str
    ) -> pd.DataFrame:

        try:

            df = pd.read_csv(
                file_path
            )

            df = df.drop(
                columns=["claim_id"]
            )

            print(
                f"Dataset Loaded Successfully"
            )

            print(
                f"Rows : {df.shape[0]}"
            )

            print(
                f"Columns : {df.shape[1]}"
            )

            return df

        except Exception as e:

            raise Exception(
                f"Data Loading Failed : {e}"
            )

    ###################################################
    # Dataset Summary
    ###################################################

    def dataset_summary(
        self,
        df
    ):

        print("\nShape")

        print(df.shape)

        print("\nColumns")

        print(df.columns.tolist())

        #print("\nData Types")

        #print(df.dtypes)

    ###################################################
    # Missing Value Report
    ###################################################

    def missing_value_report(
        self,
        df
    ):

        report = pd.DataFrame({

            "column": df.columns,

            "missing_count":
                df.isnull().sum(),

            "missing_percent":

                round(
                    (
                        df.isnull().sum()
                        /
                        len(df)
                    ) * 100,
                    2
                )

        })

        return report

    ###################################################
    # Duplicate Removal
    ###################################################

    def remove_duplicates(
        self,
        df
    ):

        before = len(df)

        df = df.drop_duplicates()

        after = len(df)

        """
        print(
            f"Duplicates Removed : "
            f"{before-after}"
        )"""

        return df

    ###################################################
    # Data Validation
    ###################################################

    def validate_data(
        self,
        df
    ):

        if "age" in df.columns:

            invalid_age = len(

                df[
                    (
                        df["age"] < 0
                    )
                    |
                    (
                        df["age"] > 120
                    )
                ]

            )

            """
            print(
                f"Invalid Age Records: "
                f"{invalid_age}"
            )"""

        return df

    ###################################################
    # Feature Engineering
    ###################################################

    def feature_engineering(
        self,
        df
    ):

        df = df.copy()

        

        return df

    ###################################################
    # Outlier Detection
    ###################################################

    def detect_outliers(
        self,
        df,
        columns
    ):

        results = {}

        for col in columns:

            Q1 = df[col].quantile(
                0.25
            )

            Q3 = df[col].quantile(
                0.75
            )

            IQR = Q3 - Q1

            lower = (
                Q1
                -
                1.5 * IQR
            )

            upper = (
                Q3
                +
                1.5 * IQR
            )

            count = len(

                df[
                    (
                        df[col] < lower
                    )
                    |
                    (
                        df[col] > upper
                    )
                ]

            )

            results[col] = count

        return results

    ###################################################
    # Outlier Treatment
    ###################################################

    def cap_outliers(
        self,
        df,
        columns
    ):

        df = df.copy()

        for col in columns:

            Q1 = df[col].quantile(
                0.25
            )

            Q3 = df[col].quantile(
                0.75
            )

            IQR = Q3 - Q1

            lower = (
                Q1
                -
                1.5 * IQR
            )

            upper = (
                Q3
                +
                1.5 * IQR
            )

            df[col] = np.clip(

                df[col],

                lower,

                upper

            )

        return df

    ###################################################
    # Build Pipeline
    ###################################################

    def build_pipeline(
        self,
        X
    ):

        self.numeric_columns = (

            X.select_dtypes(
                include=np.number
            )
            .columns
            .tolist()

        )

        self.categorical_columns = (

            X.select_dtypes(
                exclude=np.number
            )
            .columns
            .tolist()

        )

        numeric_pipeline = Pipeline([

            (
                "imputer",

                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",

                RobustScaler()
            )

        ])

        categorical_pipeline = Pipeline([

            (
                "imputer",

                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",

                
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )

        ])

        self.preprocessor = (

            ColumnTransformer(

                transformers=[

                    (
                        "num",
                        numeric_pipeline,
                        self.numeric_columns
                    ),

                    (
                        "cat",
                        categorical_pipeline,
                        self.categorical_columns
                    )

                ]

            )

        )

        return self.preprocessor

    ###################################################
    # Fit Transform
    ###################################################

    def fit_transform(
        self,
        X
    ):

        self.build_pipeline(X)

        return (

            self.preprocessor
            .fit_transform(X)

        )

    ###################################################
    # Transform
    ###################################################

    def transform(
        self,
        X
    ):

        return (

            self.preprocessor
            .transform(X)

        )

    ###################################################
    # Save Preprocessor
    ###################################################

    def save_preprocessor(
        self,
        path
    ):

        joblib.dump(

            self.preprocessor,

            path

        )

    ###################################################
    # Load Preprocessor
    ###################################################

    def load_preprocessor(
        self,
        path
    ):

        self.preprocessor = (

            joblib.load(path)

        )

        return self.preprocessor