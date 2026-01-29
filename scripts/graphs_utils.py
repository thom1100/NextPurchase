import pandas as pd
from pathlib import Path
from functools import lru_cache


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


@lru_cache(maxsize=1)
def load_clients() -> pd.DataFrame:
    clients_path = DATA_DIR / "clients.csv"
    df = pd.read_csv(clients_path)
    return df


@lru_cache(maxsize=1)
def load_transactions() -> pd.DataFrame:
    tx_path = DATA_DIR / "transactions.csv"
    df = pd.read_csv(
        tx_path,
        parse_dates=["SaleTransactionDate"],
    )
    return df


@lru_cache(maxsize=1)
def load_products() -> pd.DataFrame:
    products_path = DATA_DIR / "products.csv"
    df = pd.read_csv(products_path)
    return df


@lru_cache(maxsize=1)
def load_stores() -> pd.DataFrame:
    stores_path = DATA_DIR / "stores.csv"
    df = pd.read_csv(stores_path)
    return df


@lru_cache(maxsize=1)
def load_stocks() -> pd.DataFrame:
    """
    Stocks par pays et produit.
    Colonnes: StoreCountry, ProductID, Quantity
    """
    stocks_path = DATA_DIR / "stocks.csv"
    df = pd.read_csv(stocks_path)
    return df


@lru_cache(maxsize=1)
def load_full_transactions() -> pd.DataFrame:
    """
    Transactions enrichies avec infos client, produit et magasin.
    """
    tx = load_transactions()
    clients = load_clients()
    products = load_products()
    stores = load_stores()

    df = (
        tx.merge(clients, on="ClientID", how="left")
        .merge(products, on="ProductID", how="left")
        .merge(stores, on="StoreID", how="left")
    )
    return df


def get_client_list() -> pd.DataFrame:
    """
    Retourne la liste unique des clients avec quelques métadonnées.
    """
    clients = load_clients().copy()
    # Optionnel : on peut créer un libellé lisible pour l’UI
    clients["Label"] = (
        clients["ClientID"].astype(str)
        + " | "
        + clients["ClientSegment"].astype(str)
        + " | "
        + clients["ClientCountry"].astype(str)
    )
    return clients


def get_client_overview(client_id: int | str) -> dict:
    """
    Retourne des indicateurs de haut niveau pour un client.
    """
    df = load_full_transactions()
    client_id = int(client_id)

    client_tx = df[df["ClientID"] == client_id].copy()

    if client_tx.empty:
        return {
            "has_transactions": False,
            "total_spent": 0.0,
            "nb_transactions": 0,
            "nb_products": 0,
            "first_purchase": None,
            "last_purchase": None,
            "favorite_store_id": None,
            "favorite_store_country": None,
        }

    total_spent = client_tx["SalesNetAmountEuro"].sum()
    nb_transactions = client_tx.shape[0]
    nb_products = client_tx["ProductID"].nunique()
    first_purchase = client_tx["SaleTransactionDate"].min()
    last_purchase = client_tx["SaleTransactionDate"].max()

    # Magasin favori (par montant dépensé)
    store_agg = (
        client_tx.groupby(["StoreID", "StoreCountry"])["SalesNetAmountEuro"]
        .sum()
        .reset_index()
        .sort_values("SalesNetAmountEuro", ascending=False)
    )
    favorite_store_id = store_agg.iloc[0]["StoreID"]
    favorite_store_country = store_agg.iloc[0]["StoreCountry"]

    return {
        "has_transactions": True,
        "total_spent": total_spent,
        "nb_transactions": nb_transactions,
        "nb_products": nb_products,
        "first_purchase": first_purchase,
        "last_purchase": last_purchase,
        "favorite_store_id": favorite_store_id,
        "favorite_store_country": favorite_store_country,
    }


def get_client_recent_purchases(client_id: int | str, n_recent: int = 20) -> pd.DataFrame:
    """
    Retourne les achats récents du client, avec produit, catégorie, magasin et prix.
    """
    df = load_full_transactions()
    client_id = int(client_id)

    client_tx = df[df["ClientID"] == client_id].copy()
    if client_tx.empty:
        return client_tx

    client_tx = client_tx.sort_values("SaleTransactionDate", ascending=False)
    cols = [
        "SaleTransactionDate",
        "ProductID",
        "Category",
        "FamilyLevel1",
        "FamilyLevel2",
        "Universe",
        "StoreID",
        "StoreCountry",
        "Quantity",
        "SalesNetAmountEuro",
    ]
    existing_cols = [c for c in cols if c in client_tx.columns]
    return client_tx[existing_cols].head(n_recent)


def get_client_spend_over_time(client_id: int | str, freq: str = "M") -> pd.DataFrame:
    """
    Retourne une série temps agrégée (par mois par défaut) du montant dépensé par le client.
    """
    tx = load_transactions().copy()
    client_id = int(client_id)
    tx = tx[tx["ClientID"] == client_id]
    if tx.empty:
        return tx

    tx = tx.set_index("SaleTransactionDate")
    agg = (
        tx["SalesNetAmountEuro"]
        .resample(freq)
        .sum()
        .reset_index()
        .rename(columns={"SalesNetAmountEuro": "TotalSpent"})
    )
    return agg


def get_client_favorite_categories(client_id: int | str, top_n: int = 10) -> pd.DataFrame:
    """
    Retourne les catégories préférées du client par montant dépensé.
    """
    df = load_full_transactions()
    client_id = int(client_id)
    client_tx = df[df["ClientID"] == client_id].copy()
    if client_tx.empty:
        return client_tx

    if "Category" not in client_tx.columns:
        return client_tx

    agg = (
        client_tx.groupby("Category")["SalesNetAmountEuro"]
        .sum()
        .reset_index()
        .sort_values("SalesNetAmountEuro", ascending=False)
        .head(top_n)
    )
    return agg


# ---------- VUES MAGASIN ----------


def get_store_list() -> pd.DataFrame:
    """
    Retourne uniquement les magasins qui ont au moins une transaction,
    pour éviter d'afficher des magasins à 0 partout dans l'app.
    """
    stores = load_stores().copy()
    tx = load_transactions().copy()

    used_store_ids = tx["StoreID"].unique()
    stores = stores[stores["StoreID"].isin(used_store_ids)]

    stores["Label"] = (
        stores["StoreID"].astype(str) + " | " + stores["StoreCountry"].astype(str)
    )
    return stores


def get_store_overview(store_id: int | str) -> dict:
    """
    KPIs pour un magasin spécifique.
    """
    df = load_full_transactions()
    store_id = int(store_id)
    store_tx = df[df["StoreID"] == store_id].copy()

    if store_tx.empty:
        return {
            "has_transactions": False,
            "total_spent": 0.0,
            "nb_transactions": 0,
            "nb_clients": 0,
            "nb_products": 0,
            "first_purchase": None,
            "last_purchase": None,
            "country": None,
        }

    total_spent = store_tx["SalesNetAmountEuro"].sum()
    nb_transactions = store_tx.shape[0]
    nb_clients = store_tx["ClientID"].nunique()
    nb_products = store_tx["ProductID"].nunique()
    first_purchase = store_tx["SaleTransactionDate"].min()
    last_purchase = store_tx["SaleTransactionDate"].max()
    country = store_tx["StoreCountry"].iloc[0] if "StoreCountry" in store_tx.columns else None

    return {
        "has_transactions": True,
        "total_spent": total_spent,
        "nb_transactions": nb_transactions,
        "nb_clients": nb_clients,
        "nb_products": nb_products,
        "first_purchase": first_purchase,
        "last_purchase": last_purchase,
        "country": country,
    }


def get_store_top_products(store_id: int | str, top_n: int = 20) -> pd.DataFrame:
    df = load_full_transactions()
    store_id = int(store_id)
    store_tx = df[df["StoreID"] == store_id].copy()
    if store_tx.empty:
        return store_tx

    agg = (
        store_tx.groupby(["ProductID", "Category"])["SalesNetAmountEuro"]
        .sum()
        .reset_index()
        .sort_values("SalesNetAmountEuro", ascending=False)
        .head(top_n)
    )
    return agg


def get_store_top_clients(store_id: int | str, top_n: int = 20) -> pd.DataFrame:
    df = load_full_transactions()
    store_id = int(store_id)
    store_tx = df[df["StoreID"] == store_id].copy()
    if store_tx.empty:
        return store_tx

    agg = (
        store_tx.groupby("ClientID")["SalesNetAmountEuro"]
        .sum()
        .reset_index()
        .sort_values("SalesNetAmountEuro", ascending=False)
        .head(top_n)
    )
    return agg


# ---------- VUES PRODUIT ----------


def get_product_list() -> pd.DataFrame:
    products = load_products().copy()
    products["Label"] = (
        products["ProductID"].astype(str)
        + " | "
        + products["Category"].astype(str)
        + " | "
        + products["FamilyLevel1"].astype(str)
    )
    return products


def get_product_overview(product_id: int | str) -> dict:
    """
    KPIs globaux pour un produit.
    """
    df = load_full_transactions()
    product_id = int(product_id)
    prod_tx = df[df["ProductID"] == product_id].copy()

    if prod_tx.empty:
        return {
            "has_transactions": False,
            "total_revenue": 0.0,
            "total_quantity": 0.0,
            "nb_clients": 0,
            "nb_stores": 0,
            "first_purchase": None,
            "last_purchase": None,
            "category": None,
            "family1": None,
            "family2": None,
            "universe": None,
        }

    total_revenue = prod_tx["SalesNetAmountEuro"].sum()
    total_quantity = prod_tx["Quantity"].sum()
    nb_clients = prod_tx["ClientID"].nunique()
    nb_stores = prod_tx["StoreID"].nunique()
    first_purchase = prod_tx["SaleTransactionDate"].min()
    last_purchase = prod_tx["SaleTransactionDate"].max()

    category = prod_tx["Category"].iloc[0] if "Category" in prod_tx.columns else None
    family1 = prod_tx["FamilyLevel1"].iloc[0] if "FamilyLevel1" in prod_tx.columns else None
    family2 = prod_tx["FamilyLevel2"].iloc[0] if "FamilyLevel2" in prod_tx.columns else None
    universe = prod_tx["Universe"].iloc[0] if "Universe" in prod_tx.columns else None

    return {
        "has_transactions": True,
        "total_revenue": total_revenue,
        "total_quantity": total_quantity,
        "nb_clients": nb_clients,
        "nb_stores": nb_stores,
        "first_purchase": first_purchase,
        "last_purchase": last_purchase,
        "category": category,
        "family1": family1,
        "family2": family2,
        "universe": universe,
    }


def get_product_stock_by_country(product_id: int | str) -> pd.DataFrame:
    """
    Stock disponible par pays pour un produit donné.
    """
    stocks = load_stocks().copy()
    product_id = int(product_id)
    prod_stock = stocks[stocks["ProductID"] == product_id].copy()
    if prod_stock.empty:
        return prod_stock

    agg = (
        prod_stock.groupby("StoreCountry")["Quantity"]
        .sum()
        .reset_index()
        .sort_values("Quantity", ascending=False)
    )
    return agg


def get_product_sales_over_time(product_id: int | str, freq: str = "M") -> pd.DataFrame:
    tx = load_transactions().copy()
    product_id = int(product_id)
    tx = tx[tx["ProductID"] == product_id]
    if tx.empty:
        return tx

    tx = tx.set_index("SaleTransactionDate")
    agg = (
        tx[["Quantity", "SalesNetAmountEuro"]]
        .resample(freq)
        .sum()
        .reset_index()
        .rename(columns={"SalesNetAmountEuro": "TotalRevenue"})
    )
    return agg


def get_product_top_clients(product_id: int | str, top_n: int = 20) -> pd.DataFrame:
    df = load_full_transactions()
    product_id = int(product_id)
    prod_tx = df[df["ProductID"] == product_id].copy()
    if prod_tx.empty:
        return prod_tx

    agg = (
        prod_tx.groupby("ClientID")["SalesNetAmountEuro"]
        .sum()
        .reset_index()
        .sort_values("SalesNetAmountEuro", ascending=False)
        .head(top_n)
    )
    return agg


def get_product_top_countries(product_id: int | str, top_n: int = 20) -> pd.DataFrame:
    df = load_full_transactions()
    product_id = int(product_id)
    prod_tx = df[df["ProductID"] == product_id].copy()
    if prod_tx.empty:
        return prod_tx

    agg = (
        prod_tx.groupby("StoreCountry")["SalesNetAmountEuro"]
        .sum()
        .reset_index()
        .sort_values("SalesNetAmountEuro", ascending=False)
        .head(top_n)
    )
    return agg



