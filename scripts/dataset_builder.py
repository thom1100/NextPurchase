import pandas as pd

clients = pd.read_csv("../data/raw/clients.csv")
products = pd.read_csv('../data/raw/products.csv')
stocks = pd.read_csv("../data/raw/stocks.csv")
stores = pd.read_csv("../data/raw/stores.csv")
transactions = pd.read_csv("../data/raw/transactions.csv")
import pandas as pd

# --- transactions ---
transactions["SaleTransactionDate"] = pd.to_datetime(
    transactions["SaleTransactionDate"],
    errors="coerce",
    utc=False,               # keep naive timestamps unless you have tz info
    dayfirst=False           # set True if your CSV is DD/MM/YYYY
)

transactions.drop_duplicates(inplace = True)
transactions["Quantity"] = pd.to_numeric(transactions["Quantity"], errors="coerce")
transactions["SalesNetAmountEuro"] = pd.to_numeric(transactions["SalesNetAmountEuro"], errors="coerce")
transactions = transactions.dropna(subset=["ClientID", "ProductID", "SaleTransactionDate"])

clients["ClientGender"] = clients["ClientGender"].fillna("N/A")
clients.drop(columns = "Age", inplace = True)

df = (
    transactions
    .merge(products, on="ProductID", how="left")
    .merge(stores, on="StoreID", how="left")
    .merge(clients, on="ClientID", how="left")
)

df["SaleTransactionDate"] = pd.to_datetime(df["SaleTransactionDate"])
df = df.sort_values(["ClientID", "SaleTransactionDate"])


TARGET_LEVEL = "FamilyLevel2"
# alternatives: "Category", "FamilyLevel1"


df["target_group"] = df[TARGET_LEVEL]


df["next_target_group"] = (
    df.groupby("ClientID")["target_group"].shift(-1)
)

df_model = df[df["next_target_group"].notna()].copy()

df.to_csv("../data/transformed/df_family_2.csv")
df_model.to_csv("../data/transformed/df_model_family_2.csv")
