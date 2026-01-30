"""
Script utilitaire pour entraîner le modèle XGBoost de recommandation
et sauvegarder tous les artefacts nécessaires à l'app Streamlit :
- models/preprocessor.joblib
- models/label_encoder.joblib
- models/xgb_model.joblib
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "transformed" / "df_model_final_family_2.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Fichier de données introuvable : {DATA_PATH}. "
            "Exécutez d'abord scripts/dataset_builder.py puis scripts/feature_engineering.py."
        )

    print(f"Chargement des données depuis {DATA_PATH}...")
    df_model = pd.read_csv(DATA_PATH)

    # Colonnes de base utilisées dans le notebook
    TARGET_COL = "next_target_group"
    DATE_COL = "SaleTransactionDate"
    ID_COL = "ClientID"

    # Encodage de la cible
    print("Encodage de la cible...")
    label_encoder = LabelEncoder()
    df_model["y"] = label_encoder.fit_transform(df_model[TARGET_COL])

    # Construction de la liste des features
    exclude = [ID_COL, DATE_COL, TARGET_COL, "y", "Unnamed: 0.1", "Unnamed: 0"]
    candidate = [c for c in df_model.columns if c not in exclude]

    numeric_cols = df_model[candidate].select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in candidate if c not in numeric_cols]

    print(f"{len(numeric_cols)} variables numériques, {len(categorical_cols)} catégorielles.")

    # Pipelines de prétraitement comme dans le notebook
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ord",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
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

    print("Prétraitement des features...")
    X = preprocessor.fit_transform(df_model[numeric_cols + categorical_cols])
    y = df_model["y"].values

    print(f"Dimensions de X : {X.shape}, y : {y.shape}")

    # Modèle XGBoost (hyperparamètres alignés sur le notebook)
    print("Entraînement du modèle XGBoost...")
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

    # Sauvegarde des artefacts
    preproc_path = MODEL_DIR / "preprocessor.joblib"
    le_path = MODEL_DIR / "label_encoder.joblib"
    model_path = MODEL_DIR / "xgb_model.joblib"

    print(f"Sauvegarde du préprocesseur dans {preproc_path}...")
    joblib.dump(preprocessor, preproc_path)
    print(f"Sauvegarde du label encoder dans {le_path}...")
    joblib.dump(label_encoder, le_path)
    print(f"Sauvegarde du modèle dans {model_path}...")
    joblib.dump(model, model_path)

    print("✅ Entraînement terminé et artefacts sauvegardés.")


if __name__ == "__main__":
    main()


