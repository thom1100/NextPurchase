import pandas as pd
from sklearn.preprocessing import LabelEncoder


TARGET_LEVEL = "FamilyLevel2"
# alternatives: "Category", "FamilyLevel1"


df_model = pd.read_csv("../data/transformed/df_model_family_2.csv")
stocks = pd.read_csv("../data/raw/stocks.csv")

df_model["SaleTransactionDate"] = pd.to_datetime(df_model["SaleTransactionDate"])

df_model["days_since_last_purchase"] = (
    df_model["SaleTransactionDate"]
    - df_model.groupby("ClientID")["SaleTransactionDate"].shift(1)
).dt.days

df_model["nb_purchases_so_far"] = (
    df_model.groupby("ClientID").cumcount()
)

df_model["cum_quantity"] = (
    df_model.groupby("ClientID")["Quantity"].cumsum()
)

df_model["cum_spent"] = (
    df_model.groupby("ClientID")["SalesNetAmountEuro"].cumsum()
)

pref = (
    df_model
    .groupby(["ClientID", TARGET_LEVEL])
    .size()
    .unstack(fill_value=0)
)

pref_ratio = pref.div(pref.sum(axis=1), axis=0)
pref_ratio.columns = [f"pref_{c}" for c in pref_ratio.columns]

df_model = df_model.merge(
    pref_ratio,
    left_on="ClientID",
    right_index=True,
    how="left"
)

df_model["last_target_group"] = (
    df_model.groupby("ClientID")[TARGET_LEVEL].shift(0)
)

df_model["same_country"] = (
    df_model["StoreCountry"] == df_model["ClientCountry"]
).astype(int)


le = LabelEncoder()
df_model["y"] = le.fit_transform(df_model["next_target_group"])

df_model.to_csv("../data/transformed/df_model_final_family_2.csv")


# Not added yet, for next step, to bias the model towards available products
# stock_agg = (
#     stocks
#     .groupby(["StoreCountry", "ProductID"])["Quantity"]
#     .sum()
#     .reset_index()
# )

