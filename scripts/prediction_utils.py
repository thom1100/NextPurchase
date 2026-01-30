"""
Module pour charger les modèles de prédiction et faire des recommandations de produits.
"""
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from functools import lru_cache

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

LEVEL_FAMILY_L2 = "family_level2"
LEVEL_FAMILY_L1 = "family_level1"
LEVEL_CATEGORY = "category"

_LEVEL_TO_FILES = {
    LEVEL_FAMILY_L2: {
        "model": "xgb_model.joblib",
        "preprocessor": "preprocessor.joblib",
        "label_encoder": "label_encoder.joblib",
        "data": "df_model_final_family_2.csv",
    },
    LEVEL_FAMILY_L1: {
        "model": "xgb_model_family_level1.joblib",
        "preprocessor": "preprocessor_family_level1.joblib",
        "label_encoder": "label_encoder_family_level1.joblib",
        "data": "df_model_final_family_level1.csv",
    },
    LEVEL_CATEGORY: {
        "model": "xgb_model_category.joblib",
        "preprocessor": "preprocessor_category.joblib",
        "label_encoder": "label_encoder_category.joblib",
        "data": "df_model_final_category.csv",
    },
}


@lru_cache(maxsize=8)
def load_model_artifacts(level: str = LEVEL_FAMILY_L2):
    """
    Charge les artefacts du modèle (preprocessor, label_encoder, model).
    Retourne None si les fichiers n'existent pas.
    """
    try:
        cfg = _LEVEL_TO_FILES.get(level)
        if cfg is None:
            return None

        preprocessor_path = MODEL_DIR / cfg["preprocessor"]
        label_encoder_path = MODEL_DIR / cfg["label_encoder"]
        model_path = MODEL_DIR / cfg["model"]
        
        if not all([preprocessor_path.exists(), label_encoder_path.exists(), model_path.exists()]):
            return None
        
        preprocessor = joblib.load(preprocessor_path)
        label_encoder = joblib.load(label_encoder_path)
        model = joblib.load(model_path)

        # Ensure the loaded sklearn wrapper is actually fitted.
        # If the model was joblib-dumped before `.fit()` (or if the pickle is incompatible
        # with the current xgboost version), it may load but have no Booster.
        try:
            if hasattr(model, "get_booster"):
                model.get_booster()
        except Exception as e:
            print(
                "Model file exists but appears unfitted/incompatible with this environment: "
                f"{model_path} ({type(e).__name__}: {e})"
            )
            return None
        
        return preprocessor, label_encoder, model
    except Exception as e:
        print(f"Erreur lors du chargement des modèles: {e}")
        return None


@lru_cache(maxsize=8)
def load_model_data(level: str = LEVEL_FAMILY_L2):
    """
    Charge le fichier df_model_final_family_2.csv qui contient les features.
    Retourne None si le fichier n'existe pas.
    """
    try:
        cfg = _LEVEL_TO_FILES.get(level)
        if cfg is None:
            return None

        data_path = DATA_DIR / "transformed" / cfg["data"]
        if not data_path.exists():
            return None
        df = pd.read_csv(data_path, parse_dates=["SaleTransactionDate"])
        return df
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        return None


@lru_cache(maxsize=1)
def load_product_mapping():
    """
    Charge le mapping ProductID -> FamilyLevel2 pour afficher les noms de produits.
    """
    try:
        products_path = DATA_DIR / "raw" / "products.csv"
        if not products_path.exists():
            return {}
        products = pd.read_csv(products_path)
        # Créer un mapping ProductID -> FamilyLevel2
        mapping = dict(zip(products["ProductID"].values, products["FamilyLevel2"].values))
        return mapping
    except Exception as e:
        print(f"Erreur lors du chargement du mapping produits: {e}")
        return {}


def get_client_features_for_prediction(
    client_id: int, level: str = LEVEL_FAMILY_L2
) -> Optional[pd.DataFrame]:
    """
    Récupère la dernière ligne de features pour un client donné depuis df_model_final_family_2.csv.
    Cette ligne contient toutes les features nécessaires pour la prédiction.
    
    Args:
        client_id: ID du client
        
    Returns:
        DataFrame avec une seule ligne contenant les features, ou None si non trouvé
    """
    df_model = load_model_data(level=level)
    if df_model is None:
        return None
    
    client_id = int(client_id)
    client_rows = df_model[df_model["ClientID"] == client_id].copy()
    
    if client_rows.empty:
        return None
    
    # Prendre la dernière ligne (la plus récente)
    last_row = client_rows.sort_values("SaleTransactionDate").iloc[[-1]]
    return last_row


def predict_top_k_for_client(
    client_id: int,
    campaign_type: str = "mail",
    top_k: Optional[int] = None,
    level: str = LEVEL_FAMILY_L2,
) -> List[Tuple[str, float]]:
    """
    Prédit les top K produits (FamilyLevel2) pour un client donné.
    
    Args:
        client_id: ID du client
        campaign_type: Type de campagne ("telephone" ou "mail")
        top_k: Nombre de produits à recommander (si None, utilise campaign_type)
        
    Returns:
        Liste de tuples (nom_produit, probabilité) triés par probabilité décroissante
    """
    # Déterminer le nombre de produits selon le type de campagne
    if top_k is None:
        top_k = 1 if campaign_type.lower() == "telephone" else 5
    
    # Charger les artefacts du modèle
    artifacts = load_model_artifacts(level=level)
    if artifacts is None:
        raise ValueError(
            "Prediction artifacts are not available for this level. "
            "Make sure the preprocessor + label encoder + model files exist for the selected level."
        )
    
    preprocessor, label_encoder, model = artifacts
    
    # Récupérer les features du client
    client_features = get_client_features_for_prediction(client_id, level=level)
    if client_features is None:
        raise ValueError(
            f"Aucune donnée trouvée pour le client {client_id}. "
            "Le client doit avoir au moins une transaction dans df_model_final_family_2.csv."
        )
    
    # Identifier les colonnes de features (exclure les colonnes non-features)
    # Important: do NOT exclude `target_group` here.
    # The current training setup often uses `target_group` as an input feature,
    # so dropping it will cause ColumnTransformer to fail with
    # "columns are missing: {'target_group'}".
    exclude_cols = [
        "ClientID",
        "SaleTransactionDate",
        "next_target_group",
        "y",
        "Unnamed: 0.1",
        "Unnamed: 0",
    ]
    
    # Colonnes numériques et catégorielles utilisées pour l'entraînement
    # Filtrer les colonnes qui existent réellement dans le DataFrame
    feature_cols = [
        c for c in client_features.columns 
        if c not in exclude_cols and not c.startswith("Unnamed")
    ]
    
    if not feature_cols:
        raise ValueError(
            f"Aucune colonne de feature trouvée pour le client {client_id}. "
            "Vérifiez que le fichier df_model_final_family_2.csv contient les colonnes nécessaires."
        )
    
    # Préparer les features pour la prédiction
    X = client_features[feature_cols]
    
    # Transformer avec le preprocessor
    X_proc = preprocessor.transform(X)
    
    # Faire la prédiction
    proba = model.predict_proba(X_proc)[0]  # shape (n_classes,)
    
    # Obtenir les top K indices
    top_idx = np.argsort(proba)[-top_k:][::-1]
    
    # Convertir les indices en labels (FamilyLevel2)
    labels = label_encoder.inverse_transform(top_idx)
    
    # Retourner les tuples (label, probabilité)
    predictions = list(zip(labels, proba[top_idx]))
    
    return predictions


def get_product_details_from_family(family_level2: str) -> pd.DataFrame:
    """
    Récupère les détails des produits (ProductID, Category, etc.) pour une FamilyLevel2 donnée.
    
    Args:
        family_level2: Nom de la famille de produits (FamilyLevel2)
        
    Returns:
        DataFrame avec les détails des produits de cette famille
    """
    try:
        products_path = DATA_DIR / "raw" / "products.csv"
        if not products_path.exists():
            return pd.DataFrame()
        
        products = pd.read_csv(products_path)
        family_products = products[products["FamilyLevel2"] == family_level2].copy()
        return family_products
    except Exception as e:
        print(f"Erreur lors de la récupération des détails produits: {e}")
        return pd.DataFrame()


def get_product_details_for_level(label: str, level: str) -> pd.DataFrame:
    """
    Return products matching a predicted label for a given granularity level.

    - category: filter products by Category
    - family_level1: filter products by FamilyLevel1
    - family_level2: filter products by FamilyLevel2
    """
    try:
        products_path = DATA_DIR / "raw" / "products.csv"
        if not products_path.exists():
            return pd.DataFrame()
        products = pd.read_csv(products_path)

        col_map = {
            LEVEL_CATEGORY: "Category",
            LEVEL_FAMILY_L1: "FamilyLevel1",
            LEVEL_FAMILY_L2: "FamilyLevel2",
        }
        col = col_map.get(level)
        if col is None or col not in products.columns:
            return pd.DataFrame()
        return products[products[col] == label].copy()
    except Exception:
        return pd.DataFrame()

