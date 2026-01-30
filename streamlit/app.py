import sys
from pathlib import Path

import streamlit as st
import pandas as pd


# S'assurer que le dossier scripts est dans le path pour l'import
BASE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = BASE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from graphs_utils import (  # noqa: E402
    get_client_list,
    get_client_overview,
    get_client_recent_purchases,
    get_client_spend_over_time,
    get_client_favorite_categories,
    get_store_list,
    get_store_overview,
    get_store_top_products,
    get_store_top_clients,
    get_product_list,
    get_product_overview,
    get_product_stock_by_country,
    get_product_sales_over_time,
    get_product_top_clients,
    get_product_top_countries,
)
from prediction_utils import (  # noqa: E402
    predict_top_k_for_client,
    get_product_details_for_level,
    LEVEL_CATEGORY,
    LEVEL_FAMILY_L1,
    LEVEL_FAMILY_L2,
    load_model_artifacts,
    load_model_data,
)


st.set_page_config(
    page_title="NextPurchase - 360° View",
    page_icon="🛒",
    layout="wide",
)


@st.cache_data
def _get_client_list_cached() -> pd.DataFrame:
    return get_client_list()


@st.cache_data
def _get_store_list_cached() -> pd.DataFrame:
    return get_store_list()


@st.cache_data
def _get_product_list_cached() -> pd.DataFrame:
    return get_product_list()


st.title("🛍️ NextPurchase – 360° Customer / Store / Product View")
st.markdown(
    """
Interactive analysis of your transaction data:

- **Customer** tab: 360° customer profile (favorite store, recent purchases, spend over time, favorite categories, **product recommendations** powered by XGBoost).
- **Store** tab: store performance (revenue, customers, key products, top customers).
- **Product** tab: stock by country, sales over time, top customers and countries.
"""
)

tab_product, tab_store, tab_client = st.tabs(
    ["📦 Product", "🏬 Store", "👤 Customer"]
)


# ========================== CUSTOMER TAB ==========================
with tab_client:
    clients_df = _get_client_list_cached()

    if clients_df.empty:
        st.error("No customer found in the data.")
    else:
        col_select, col_filters = st.columns([2, 1])

        with col_filters:
            st.subheader("Filters")
            available_segments = sorted(clients_df["ClientSegment"].dropna().unique())
            selected_segments = st.multiselect(
                "Customer segment",
                options=available_segments,
                default=available_segments[:5]
                if len(available_segments) > 5
                else available_segments,
            )

            available_countries = sorted(
                clients_df["ClientCountry"].dropna().unique()
            )
            selected_countries = st.multiselect(
                "Country",
                options=available_countries,
                default=available_countries,
                key="client_countries",
            )

        filtered_clients = clients_df.copy()
        if selected_segments:
            filtered_clients = filtered_clients[
                filtered_clients["ClientSegment"].isin(selected_segments)
            ]
        if selected_countries:
            filtered_clients = filtered_clients[
                filtered_clients["ClientCountry"].isin(selected_countries)
            ]

        with col_select:
            st.subheader("Customer selection")

            if filtered_clients.empty:
                st.warning("No customer matches the selected filters.")
                client_id = None
            else:
                # Limit the number of displayed rows to keep the UI fast
                max_rows = st.slider(
                    "Max number of customers displayed",
                    min_value=50,
                    max_value=1000,
                    value=200,
                    step=50,
                    help="Adjust to display more or fewer rows in the table below.",
                )

                display_clients = (
                    filtered_clients.sort_values(
                        ["TotalSpent", "ClientSegment", "ClientCountry", "ClientID"],
                        ascending=[False, True, True, True],
                    )
                    .head(max_rows)
                    .copy()
                )

                st.markdown("**Filtered customers list (sorted by total spend)**")
                st.dataframe(
                    display_clients[
                        ["ClientID", "ClientSegment", "ClientCountry", "TotalSpent"]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

                options = display_clients["Label"].tolist()
                selection = st.selectbox(
                    "Select a customer from the list above", options=options
                )
                client_id = int(selection.split(" | ")[0])

        if client_id is not None:
            overview = get_client_overview(client_id)

            st.markdown("---")
            st.subheader("Customer summary")

            info_cols = st.columns(4)

            with info_cols[0]:
                st.metric("ClientID", value=str(client_id))
                seg = clients_df.loc[
                    clients_df["ClientID"] == client_id, "ClientSegment"
                ]
                st.metric(
                    "Segment", value=str(seg.iloc[0]) if not seg.empty else "N/A"
                )

            with info_cols[1]:
                country = clients_df.loc[
                    clients_df["ClientID"] == client_id, "ClientCountry"
                ]
                st.metric(
                    "Country", value=str(country.iloc[0]) if not country.empty else "N/A"
                )
                st.metric("Transactions", value=int(overview["nb_transactions"]))

            with info_cols[2]:
                st.metric(
                    "Total amount spent (€)",
                    value=f"{overview['total_spent']:,.2f}".replace(",", " "),
                )
                st.metric(
                    "Distinct products", value=int(overview["nb_products"])
                )

            with info_cols[3]:
                if overview["first_purchase"] is not None:
                    st.metric(
                        "First purchase",
                        value=overview["first_purchase"].strftime("%Y-%m-%d"),
                    )
                if overview["last_purchase"] is not None:
                    st.metric(
                        "Last purchase",
                        value=overview["last_purchase"].strftime("%Y-%m-%d"),
                    )
                st.metric(
                    "Favorite store (ID)",
                    value=str(overview["favorite_store_id"])
                    if overview["favorite_store_id"] is not None
                    else "N/A",
                )

            if overview["has_transactions"]:
                st.markdown("---")
                st.subheader("Purchase details")

                tab_recent, tab_time, tab_categories, tab_recommendations = st.tabs(
                    [
                        "🧾 Recent purchases",
                        "📈 Spend over time",
                        "🏷️ Favorite categories",
                        "🎯 Product recommendations",
                    ]
                )

                with tab_recent:
                    n_recent = st.slider(
                        "Number of recent purchases to show",
                        min_value=5,
                        max_value=50,
                        value=20,
                        step=5,
                    )
                    recent_df = get_client_recent_purchases(
                        client_id, n_recent=n_recent
                    )
                    if recent_df.empty:
                        st.info("No purchases found for this customer.")
                    else:
                        display_df = recent_df.copy()
                        if "SaleTransactionDate" in display_df.columns:
                            display_df["SaleTransactionDate"] = display_df[
                                "SaleTransactionDate"
                            ].dt.strftime("%Y-%m-%d")
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                with tab_time:
                    freq_label = st.radio(
                        "Aggregation frequency",
                        options=["Monthly", "Weekly", "Daily"],
                        horizontal=True,
                    )
                    freq_map = {
                        "Monthly": "M",
                        "Weekly": "W",
                        "Daily": "D",
                    }
                    spend_df = get_client_spend_over_time(
                        client_id, freq=freq_map[freq_label]
                    )

                    if spend_df.empty:
                        st.info(
                            "No time series available for this customer."
                        )
                    else:
                        st.line_chart(
                            spend_df.set_index("SaleTransactionDate")["TotalSpent"],
                            use_container_width=True,
                        )

                with tab_categories:
                    top_n = st.slider(
                        "Top N categories", min_value=3, max_value=20, value=10
                    )
                    cat_df = get_client_favorite_categories(client_id, top_n=top_n)
                    if cat_df.empty:
                        st.info(
                            "No category information available for this customer."
                        )
                    else:
                        st.bar_chart(
                            cat_df.set_index("Category")["SalesNetAmountEuro"],
                            use_container_width=True,
                        )

                with tab_recommendations:
                    st.markdown(
                        """
                        **Product recommendations based on the XGBoost model**
                        
                        The model predicts the product families (FamilyLevel2) that this customer is
                        most likely to buy on their next purchase.
                        """
                    )

                    # Recommendation granularity (model selection)
                    level_label = st.selectbox(
                        "Recommendation level",
                        options=[
                            ("Family level 2 (most detailed)", LEVEL_FAMILY_L2),
                            ("Family level 1", LEVEL_FAMILY_L1),
                            ("Category", LEVEL_CATEGORY),
                        ],
                        format_func=lambda x: x[0],
                    )
                    selected_level = level_label[1]

                    # Vérifier si les modèles sont disponibles
                    artifacts = load_model_artifacts(level=selected_level)
                    model_data = load_model_data(level=selected_level)

                    if artifacts is None or model_data is None:
                        # Expected artifacts per level
                        expected = {
                            LEVEL_FAMILY_L2: [
                                "models/preprocessor.joblib",
                                "models/label_encoder.joblib",
                                "models/xgb_model.joblib",
                                "data/transformed/df_model_final_family_2.csv",
                            ],
                            LEVEL_FAMILY_L1: [
                                "models/preprocessor_family_level1.joblib",
                                "models/label_encoder_family_level1.joblib",
                                "models/xgb_model_family_level1.joblib",
                                "data/transformed/df_model_final_family_level1.csv",
                            ],
                            LEVEL_CATEGORY: [
                                "models/preprocessor_category.joblib",
                                "models/label_encoder_category.joblib",
                                "models/xgb_model_category.joblib",
                                "data/transformed/df_model_final_category.csv",
                            ],
                        }
                        expected_lines = "\n".join(
                            [f"- `{p}`" for p in expected.get(selected_level, [])]
                        )
                        st.warning(
                            f"""
                            ⚠️ **Prediction models are not available.**
                            
                            Your selected level requires these files:
                            {expected_lines}

                            If these files exist but you still see this warning, your `xgb_model*.joblib`
                            might be **unfitted** (saved before `.fit()`) or **incompatible** with your
                            current `xgboost` version. In that case, re-export the model from the original
                            training environment (preferred: `model.save_model(...)`) or retrain once with
                            the current environment.
                            
                            To enable recommendations (without retraining your existing models):
                            
                            1. Ensure the transformed CSVs exist:
                               - `data/transformed/df_model_final_family_2.csv`
                               - `data/transformed/df_model_final_category.csv`
                               - `data/transformed/df_model_final_family_level1.csv` (can be generated)
                            
                            2. Run the artifacts-only script:
                               - `python scripts/build_artifacts_existing_models.py`
                            
                            This script will create the missing preprocessors + label encoders required
                            by Streamlit, and will NOT retrain or overwrite your `xgb_model*.joblib` files.
                            """
                        )
                    else:
                        # Récupérer les opt-ins pour ce client
                        client_row = clients_df[clients_df["ClientID"] == client_id]
                        if client_row.empty:
                            st.warning("Unable to retrieve opt-in information for this customer.")
                        else:
                            opt_email = bool(client_row["ClientOptINEmail"].iloc[0])
                            opt_phone = bool(client_row["ClientOptINPhone"].iloc[0])

                            if not opt_email and not opt_phone:
                                st.info(
                                    "This customer has **no Email or Phone opt-in**. "
                                    "No campaign recommendation can be suggested."
                                )
                            else:
                                # Déterminer le canal et le nombre de produits par défaut
                                campaign_type_value = None

                                if opt_email and not opt_phone:
                                    campaign_type_value = "mail"
                                    st.markdown(
                                        "This customer is **Email opt-in only**. "
                                        "The model will automatically propose the **Top 5 products** for an Email campaign."
                                    )
                                elif opt_phone and not opt_email:
                                    campaign_type_value = "telephone"
                                    st.markdown(
                                        "This customer is **Phone/SMS opt-in only**. "
                                        "The model will automatically propose the **Top 1 product** for an SMS campaign."
                                    )
                                else:
                                    st.markdown(
                                        "This customer is opt-in for **Email and Phone**. "
                                        "Please choose the campaign channel:"
                                    )
                                    channel_choice = st.radio(
                                        "Campaign channel",
                                        options=[
                                            "📧 Email (Top 5 products)",
                                            "📞 SMS (Top 1 product)",
                                        ],
                                        horizontal=True,
                                    )
                                    campaign_type_value = (
                                        "mail"
                                        if "Email" in channel_choice
                                        else "telephone"
                                    )

                                if campaign_type_value is not None:
                                    top_k = 5 if campaign_type_value == "mail" else 1

                                    if st.button(
                                        "🔮 Generate recommendations",
                                        type="primary",
                                    ):
                                        try:
                                            with st.spinner(
                                                "Computing recommendations..."
                                            ):
                                                predictions = predict_top_k_for_client(
                                                    client_id,
                                                    campaign_type=campaign_type_value,
                                                    top_k=top_k,
                                                    level=selected_level,
                                                )

                                            if predictions:
                                                canal_label = (
                                                    "Email (Top 5 products)"
                                                    if campaign_type_value == "mail"
                                                    else "SMS (Top 1 product)"
                                                )
                                                st.success(
                                                    f"✅ {len(predictions)} recommendation(s) generated for channel **{canal_label}**"
                                                )

                                                # Afficher les recommandations
                                                st.markdown("### Recommended products")

                                                for idx, (
                                                    family_name,
                                                    proba,
                                                ) in enumerate(predictions, 1):
                                                    with st.expander(
                                                        f"#{idx} {family_name} (Probability: {proba:.2%})",
                                                        expanded=(idx == 1),
                                                    ):
                                                        # Récupérer les détails des produits de cette famille
                                                        product_details = (
                                                            get_product_details_for_level(
                                                                family_name, selected_level
                                                            )
                                                        )

                                                        if not product_details.empty:
                                                            st.markdown(
                                                                f"**Product family:** {family_name}"
                                                            )
                                                            st.markdown(
                                                                f"**Purchase probability:** {proba:.2%}"
                                                            )

                                                            # Afficher les produits de cette famille
                                                            display_cols = [
                                                                "ProductID",
                                                                "Category",
                                                                "FamilyLevel1",
                                                                "FamilyLevel2",
                                                            ]
                                                            available_cols = [
                                                                c
                                                                for c in display_cols
                                                                if c
                                                                in product_details.columns
                                                            ]

                                                            if available_cols:
                                                                st.markdown(
                                                                    "**Available products in this family:**"
                                                                )
                                                                st.dataframe(
                                                                    product_details[
                                                                        available_cols
                                                                    ],
                                                                    use_container_width=True,
                                                                    hide_index=True,
                                                                )
                                                        else:
                                                            st.info(
                                                                f"Family: {family_name}"
                                                            )
                                                            st.markdown(
                                                                f"Probability: {proba:.2%}"
                                                            )

                                                # Graphique des probabilités
                                                if len(predictions) > 1:
                                                    st.markdown(
                                                        "### Prediction probabilities"
                                                    )
                                                    proba_df = pd.DataFrame(
                                                        predictions,
                                                        columns=[
                                                            "Label",
                                                            "Probability",
                                                        ],
                                                    )
                                                    proba_df = proba_df.set_index("Label")
                                                    st.bar_chart(
                                                        proba_df,
                                                        use_container_width=True,
                                                    )

                                        except ValueError as e:
                                            st.error(f"❌ Erreur : {str(e)}")
                                        except Exception as e:
                                            st.error(
                                                "❌ Une erreur inattendue s'est produite."
                                            )
                                            st.exception(e)


# ========================== STORE TAB ==========================
with tab_store:
    stores_df = _get_store_list_cached()

    if stores_df.empty:
        st.error("No store found in the data.")
    else:
        st.subheader("Store selection")
        available_countries = sorted(stores_df["StoreCountry"].dropna().unique())
        selected_countries_store = st.multiselect(
            "Country",
            options=available_countries,
            default=available_countries,
            key="store_countries",
        )

        filtered_stores = stores_df.copy()
        if selected_countries_store:
            filtered_stores = filtered_stores[
                filtered_stores["StoreCountry"].isin(selected_countries_store)
            ]

        if filtered_stores.empty:
            st.warning("No store matches the selected filters.")
        else:
            selection_store = st.selectbox(
                "Store",
                options=filtered_stores["Label"].tolist(),
            )
            store_id = int(selection_store.split(" | ")[0])

            overview_store = get_store_overview(store_id)

            st.markdown("---")
            st.subheader("Store summary")

            cols_store = st.columns(4)
            with cols_store[0]:
                st.metric("StoreID", value=str(store_id))
                st.metric("Country", value=str(overview_store["country"]))
            with cols_store[1]:
                st.metric(
                    "Total revenue (€)",
                    value=f"{overview_store['total_spent']:,.2f}".replace(",", " "),
                )
                st.metric(
                    "Transactions", value=int(overview_store["nb_transactions"])
                )
            with cols_store[2]:
                st.metric(
                    "Distinct customers", value=int(overview_store["nb_clients"])
                )
                st.metric(
                    "Distinct products", value=int(overview_store["nb_products"])
                )
            with cols_store[3]:
                if overview_store["first_purchase"] is not None:
                    st.metric(
                        "First purchase",
                        value=overview_store["first_purchase"].strftime("%Y-%m-%d"),
                    )
                if overview_store["last_purchase"] is not None:
                    st.metric(
                        "Last purchase",
                        value=overview_store["last_purchase"].strftime("%Y-%m-%d"),
                    )

            if overview_store["has_transactions"]:
                st.markdown("---")
                st.subheader("Key products & customers")
                col_p, col_c = st.columns(2)

                with col_p:
                    top_n_prod = st.slider(
                        "Top N products (by revenue)", 3, 30, 10, key="store_top_products"
                    )
                    prod_df = get_store_top_products(store_id, top_n=top_n_prod)
                    if prod_df.empty:
                        st.info("No product found for this store.")
                    else:
                        st.dataframe(
                            prod_df, use_container_width=True, hide_index=True
                        )

                with col_c:
                    top_n_clients = st.slider(
                        "Top N customers (by revenue)", 3, 30, 10, key="store_top_clients"
                    )
                    cli_df = get_store_top_clients(store_id, top_n=top_n_clients)
                    if cli_df.empty:
                        st.info("No customer found for this store.")
                    else:
                        st.dataframe(
                            cli_df, use_container_width=True, hide_index=True
                        )


# ========================== PRODUCT TAB ==========================
with tab_product:
    products_df = _get_product_list_cached()

    if products_df.empty:
        st.error("No product found in the data.")
    else:
        st.subheader("Product selection")

        available_categories = sorted(products_df["Category"].dropna().unique())
        selected_categories = st.multiselect(
            "Categories",
            options=available_categories,
            default=available_categories[:10]
            if len(available_categories) > 10
            else available_categories,
        )

        filtered_products = products_df.copy()
        if selected_categories:
            filtered_products = filtered_products[
                filtered_products["Category"].isin(selected_categories)
            ]

        if filtered_products.empty:
            st.warning("No product matches the selected filters.")
        else:
            # Default product for demos
            DEFAULT_PRODUCT_ID = 8842374365833057329

            product_options = filtered_products["Label"].tolist()
            default_index = 0
            try:
                default_label = next(
                    lbl
                    for lbl in product_options
                    if lbl.startswith(f"{DEFAULT_PRODUCT_ID} | ")
                )
                default_index = product_options.index(default_label)
            except StopIteration:
                # Default product not in current filtered list (e.g., category filter)
                default_index = 0

            selection_product = st.selectbox(
                "Product",
                options=product_options,
                index=default_index,
            )
            product_id = int(selection_product.split(" | ")[0])

            overview_prod = get_product_overview(product_id)

            st.markdown("---")
            st.subheader("Product summary")

            cols_prod = st.columns(4)
            with cols_prod[0]:
                st.metric("ProductID", value=str(product_id))
                st.metric("Category", value=str(overview_prod["category"]))
            with cols_prod[1]:
                st.metric("Family L1", value=str(overview_prod["family1"]))
                st.metric("Family L2", value=str(overview_prod["family2"]))
            with cols_prod[2]:
                st.metric(
                    "Total revenue (€)",
                    value=f"{overview_prod['total_revenue']:,.2f}".replace(",", " "),
                )
                st.metric(
                    "Quantity sold", value=int(overview_prod["total_quantity"])
                )
            with cols_prod[3]:
                st.metric(
                    "Distinct customers", value=int(overview_prod["nb_clients"])
                )
                st.metric(
                    "Distinct stores", value=int(overview_prod["nb_stores"])
                )

            if overview_prod["has_transactions"]:
                st.markdown("---")
                st.subheader("Stock & product dynamics")

                tab_stock, tab_time_p, tab_buyers = st.tabs(
                    [
                        "📦 Stock by country",
                        "📈 Sales over time",
                        "🧑‍🤝‍🧑 Who buys?",
                    ]
                )

                with tab_stock:
                    stock_df = get_product_stock_by_country(product_id)
                    if stock_df.empty:
                        st.info("No stock information found for this product.")
                    else:
                        st.bar_chart(
                            stock_df.set_index("StoreCountry")["Quantity"],
                            use_container_width=True,
                        )

                with tab_time_p:
                    freq_label_p = st.radio(
                        "Aggregation frequency",
                        options=["Monthly", "Weekly", "Daily"],
                        horizontal=True,
                        key="product_freq",
                    )
                    freq_map_p = {
                        "Monthly": "M",
                        "Weekly": "W",
                        "Daily": "D",
                    }
                    sales_df = get_product_sales_over_time(
                        product_id, freq=freq_map_p[freq_label_p]
                    )
                    if sales_df.empty:
                        st.info(
                            "No time series available for this product."
                        )
                    else:
                        st.line_chart(
                            sales_df.set_index("SaleTransactionDate")[
                                ["Quantity", "TotalRevenue"]
                            ],
                            use_container_width=True,
                        )

                with tab_buyers:
                    col_pc, col_cc = st.columns(2)

                    with col_pc:
                        top_n_cli_p = st.slider(
                            "Top N customers (by revenue)",
                            3,
                            30,
                            10,
                            key="prod_top_clients",
                        )
                        cli_p_df = get_product_top_clients(
                            product_id, top_n=top_n_cli_p
                        )
                        if cli_p_df.empty:
                            st.info("No customer found for this product.")
                        else:
                            st.dataframe(
                                cli_p_df, use_container_width=True, hide_index=True
                            )

                    with col_cc:
                        top_n_ctry_p = st.slider(
                            "Top N countries (by revenue)",
                            3,
                            30,
                            10,
                            key="prod_top_countries",
                        )
                        ctry_p_df = get_product_top_countries(
                            product_id, top_n=top_n_ctry_p
                        )
                        if ctry_p_df.empty:
                            st.info("No country found for this product.")
                        else:
                            st.bar_chart(
                                ctry_p_df.set_index("StoreCountry")[
                                    "SalesNetAmountEuro"
                                ],
                                use_container_width=True,
                            )

