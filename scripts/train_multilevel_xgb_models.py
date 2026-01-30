"""
Train and save XGBoost recommendation models for multiple granularities:
- Category
- FamilyLevel1
- FamilyLevel2

This script generates the transformed modeling datasets (df_model_final_*.csv)
and saves ALL artifacts required by the Streamlit app:

FamilyLevel2 (most detailed):
- models/preprocessor.joblib
- models/label_encoder.joblib
- models/xgb_model.joblib
- data/transformed/df_model_final_family_2.csv

FamilyLevel1:
- models/preprocessor_family_level1.joblib
- models/label_encoder_family_level1.joblib
- models/xgb_model_family_level1.joblib
- data/transformed/df_model_final_family_level1.csv

Category:
- models/preprocessor_category.joblib
- models/label_encoder_category.joblib
- models/xgb_model_category.joblib
- data/transformed/df_model_final_category.csv
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
TRANSFORMED_DIR = BASE_DIR / "data" / "transformed"
MODEL_DIR = BASE_DIR / "models"


def _load_raw() -> pd.DataFrame:
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

    # Basic cleanup (similar to dataset_builder.py)
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
    # target_level must be one of: Category, FamilyLevel1, FamilyLevel2
    if target_level not in df.columns:
        raise ValueError(
            f"target_level='{target_level}' not found in data columns. "
            f"Available columns include: {sorted(df.columns.tolist())[:30]}..."
        )

    work = df.copy()
    work["target_group"] = work[target_level]
    work["next_target_group"] = work.groupby("ClientID")["target_group"].shift(-1)

    # Keep only rows where next target exists (prediction opportunity)
    df_model = work[work["next_target_group"].notna()].copy()

    # Feature engineering (adapted from feature_engineering.py)
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

    df_model = df_model.merge(
        pref_ratio, left_on="ClientID", right_index=True, how="left"
    )

    # Keep as feature: "last target group" at current time
    df_model["last_target_group"] = df_model.groupby("ClientID")[target_level].shift(0)

    if "StoreCountry" in df_model.columns and "ClientCountry" in df_model.columns:
        df_model["same_country"] = (df_model["StoreCountry"] == df_model["ClientCountry"]).astype(
            int
        )

    # Ensure datetime type (for parsing downstream if needed)
    df_model["SaleTransactionDate"] = pd.to_datetime(df_model["SaleTransactionDate"])
    return df_model


def _train_and_save(df_model: pd.DataFrame, *, model_prefix: str) -> None:
    """
    model_prefix:
      - "" for FamilyLevel2 (legacy names)
      - "family_level1"
      - "category"
    """
    TARGET_COL = "next_target_group"
    DATE_COL = "SaleTransactionDate"
    ID_COL = "ClientID"

    # Encode target labels
    le = LabelEncoder()
    y = le.fit_transform(df_model[TARGET_COL])

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

    X = preprocessor.fit_transform(df_model[numeric_cols + categorical_cols])

    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=300,
        learning_rate=0.1,
        max_depth=0,
        max_leaves=32,
        grow_policy="lossguide",
        min_child_weight=10,
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=0.2,
        tree_method="hist",
        max_bin=128,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y)

    # Artifact paths (legacy for family_level2)
    if model_prefix == "":
        preproc_path = MODEL_DIR / "preprocessor.joblib"
        le_path = MODEL_DIR / "label_encoder.joblib"
        model_path = MODEL_DIR / "xgb_model.joblib"
    else:
        preproc_path = MODEL_DIR / f"preprocessor_{model_prefix}.joblib"
        le_path = MODEL_DIR / f"label_encoder_{model_prefix}.joblib"
        model_path = MODEL_DIR / f"xgb_model_{model_prefix}.joblib"

    joblib.dump(preprocessor, preproc_path)
    joblib.dump(le, le_path)
    joblib.dump(model, model_path)

    print(f"Saved: {preproc_path.name}, {le_path.name}, {model_path.name}")


def main() -> None:
    TRANSFORMED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    df = _load_raw()

    levels = [
        ("Category", "category", TRANSFORMED_DIR / "df_model_final_category.csv"),
        ("FamilyLevel1", "family_level1", TRANSFORMED_DIR / "df_model_final_family_level1.csv"),
        ("FamilyLevel2", "", TRANSFORMED_DIR / "df_model_final_family_2.csv"),
    ]

    for target_level, model_prefix, out_csv in levels:
        print(f"\n=== Building dataset for {target_level} ===")
        df_model = _build_model_df(df, target_level=target_level)
        df_model.to_csv(out_csv, index=False)
        print(f"Saved dataset: {out_csv.relative_to(BASE_DIR)} (rows={len(df_model)})")

        print(f"Training model for {target_level}...")
        _train_and_save(df_model, model_prefix=model_prefix)

    print("\n✅ All levels trained and artifacts saved.")


if __name__ == "__main__":
    main()


