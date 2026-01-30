"""
Build ONLY the missing prediction artifacts (no model training).

Why this exists:
- The Streamlit app needs, for each granularity level:
  - a fitted preprocessor (ColumnTransformer)
  - a fitted label encoder (LabelEncoder)
  - an already-trained XGBoost model (.joblib)

This script:
- (re)creates the transformed modeling CSVs if missing (Category / FamilyLevel1)
- fits and saves the preprocessors + label encoders
- DOES NOT modify or retrain any xgb_model*.joblib files

Outputs (expected by the Streamlit app):

FamilyLevel2:
- models/preprocessor.joblib
- models/label_encoder.joblib
- uses existing models/xgb_model.joblib
- data/transformed/df_model_final_family_2.csv (must exist)

FamilyLevel1:
- models/preprocessor_family_level1.joblib
- models/label_encoder_family_level1.joblib
- uses existing models/xgb_model_family_level1.joblib
- data/transformed/df_model_final_family_level1.csv (created if missing)

Category:
- models/preprocessor_category.joblib
- models/label_encoder_category.joblib
- uses existing models/xgb_model_category.joblib
- data/transformed/df_model_final_category.csv (must exist / can be created)
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
TRANSFORMED_DIR = BASE_DIR / "data" / "transformed"
MODEL_DIR = BASE_DIR / "models"


def _load_raw_joined() -> pd.DataFrame:
    required = [
        RAW_DIR / "clients.csv",
        RAW_DIR / "products.csv",
        RAW_DIR / "stores.csv",
        RAW_DIR / "transactions.csv",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raw files:\n" + "\n".join([f"- {p}" for p in missing])
        )

    clients = pd.read_csv(RAW_DIR / "clients.csv")
    products = pd.read_csv(RAW_DIR / "products.csv")
    stores = pd.read_csv(RAW_DIR / "stores.csv")
    tx = pd.read_csv(RAW_DIR / "transactions.csv")

    # Basic cleanup (aligned with dataset_builder.py)
    tx["SaleTransactionDate"] = pd.to_datetime(
        tx["SaleTransactionDate"], errors="coerce", utc=False, dayfirst=False
    )
    tx.drop_duplicates(inplace=True)
    tx["Quantity"] = pd.to_numeric(tx["Quantity"], errors="coerce")
    tx["SalesNetAmountEuro"] = pd.to_numeric(tx["SalesNetAmountEuro"], errors="coerce")
    tx = tx.dropna(subset=["ClientID", "ProductID", "SaleTransactionDate"])

    if "ClientGender" in clients.columns:
        clients["ClientGender"] = clients["ClientGender"].fillna("N/A")
    if "Age" in clients.columns:
        clients = clients.drop(columns=["Age"])

    df = (
        tx.merge(products, on="ProductID", how="left")
        .merge(stores, on="StoreID", how="left")
        .merge(clients, on="ClientID", how="left")
    )
    df["SaleTransactionDate"] = pd.to_datetime(df["SaleTransactionDate"])
    df = df.sort_values(["ClientID", "SaleTransactionDate"])
    return df


def _build_model_df(df: pd.DataFrame, target_level: str) -> pd.DataFrame:
    if target_level not in df.columns:
        raise ValueError(
            f"target_level='{target_level}' not found in data columns. "
            f"Available: {sorted(df.columns.tolist())[:30]}..."
        )

    work = df.copy()
    work["target_group"] = work[target_level]
    work["next_target_group"] = work.groupby("ClientID")["target_group"].shift(-1)
    df_model = work[work["next_target_group"].notna()].copy()

    # Feature engineering (aligned with feature_engineering.py)
    df_model["days_since_last_purchase"] = (
        df_model["SaleTransactionDate"]
        - df_model.groupby("ClientID")["SaleTransactionDate"].shift(1)
    ).dt.days
    df_model["nb_purchases_so_far"] = df_model.groupby("ClientID").cumcount()
    df_model["cum_quantity"] = df_model.groupby("ClientID")["Quantity"].cumsum()
    df_model["cum_spent"] = df_model.groupby("ClientID")["SalesNetAmountEuro"].cumsum()

    pref = df_model.groupby(["ClientID", target_level]).size().unstack(fill_value=0)
    pref_ratio = pref.div(pref.sum(axis=1), axis=0)
    pref_ratio.columns = [f"pref_{c}" for c in pref_ratio.columns]
    df_model = df_model.merge(pref_ratio, left_on="ClientID", right_index=True, how="left")

    df_model["last_target_group"] = df_model.groupby("ClientID")[target_level].shift(0)

    if "StoreCountry" in df_model.columns and "ClientCountry" in df_model.columns:
        df_model["same_country"] = (df_model["StoreCountry"] == df_model["ClientCountry"]).astype(
            int
        )

    df_model["SaleTransactionDate"] = pd.to_datetime(df_model["SaleTransactionDate"])
    return df_model


def _fit_and_save_artifacts(
    df_model: pd.DataFrame,
    *,
    preprocessor_path: Path,
    label_encoder_path: Path,
) -> None:
    TARGET_COL = "next_target_group"
    DATE_COL = "SaleTransactionDate"
    ID_COL = "ClientID"

    # Label encoder
    le = LabelEncoder()
    le.fit(df_model[TARGET_COL])

    # Preprocessor (same column selection logic as the notebook)
    exclude = [ID_COL, DATE_COL, TARGET_COL, "y", "Unnamed: 0.1", "Unnamed: 0"]
    candidate = [c for c in df_model.columns if c not in exclude]
    numeric_cols = df_model[candidate].select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in candidate if c not in numeric_cols]

    numeric_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ord",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    # Fit preprocessor
    preprocessor.fit(df_model[numeric_cols + categorical_cols])

    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(le, label_encoder_path)
    print(f"Saved: {preprocessor_path.name}, {label_encoder_path.name}")


def main() -> None:
    TRANSFORMED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure transformed CSVs exist (FamilyLevel2 expected to exist already)
    family_l2_csv = TRANSFORMED_DIR / "df_model_final_family_2.csv"
    category_csv = TRANSFORMED_DIR / "df_model_final_category.csv"
    family_l1_csv = TRANSFORMED_DIR / "df_model_final_family_level1.csv"

    if not family_l2_csv.exists():
        raise FileNotFoundError(
            f"Missing {family_l2_csv}. Run scripts/dataset_builder.py and scripts/feature_engineering.py first."
        )

    # Category and FamilyLevel1 can be created if missing
    if not category_csv.exists() or not family_l1_csv.exists():
        print("Building missing transformed datasets from raw files...")
        df = _load_raw_joined()
        if not category_csv.exists():
            df_cat = _build_model_df(df, target_level="Category")
            df_cat.to_csv(category_csv, index=False)
            print(f"Saved dataset: {category_csv.relative_to(BASE_DIR)} (rows={len(df_cat)})")
        if not family_l1_csv.exists():
            df_l1 = _build_model_df(df, target_level="FamilyLevel1")
            df_l1.to_csv(family_l1_csv, index=False)
            print(f"Saved dataset: {family_l1_csv.relative_to(BASE_DIR)} (rows={len(df_l1)})")

    # Fit and save artifacts for each level (no model training)
    print("\nFitting artifacts for FamilyLevel2...")
    df_l2 = pd.read_csv(family_l2_csv, parse_dates=["SaleTransactionDate"])
    _fit_and_save_artifacts(
        df_l2,
        preprocessor_path=MODEL_DIR / "preprocessor.joblib",
        label_encoder_path=MODEL_DIR / "label_encoder.joblib",
    )

    print("\nFitting artifacts for Category...")
    df_cat = pd.read_csv(category_csv, parse_dates=["SaleTransactionDate"])
    _fit_and_save_artifacts(
        df_cat,
        preprocessor_path=MODEL_DIR / "preprocessor_category.joblib",
        label_encoder_path=MODEL_DIR / "label_encoder_category.joblib",
    )

    print("\nFitting artifacts for FamilyLevel1...")
    df_l1 = pd.read_csv(family_l1_csv, parse_dates=["SaleTransactionDate"])
    _fit_and_save_artifacts(
        df_l1,
        preprocessor_path=MODEL_DIR / "preprocessor_family_level1.joblib",
        label_encoder_path=MODEL_DIR / "label_encoder_family_level1.joblib",
    )

    print("\n✅ Done. No model files were retrained or modified.")


if __name__ == "__main__":
    main()


