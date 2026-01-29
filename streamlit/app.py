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


st.set_page_config(
    page_title="NextPurchase - Vue 360°",
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


st.title("🛍️ NextPurchase – Vue 360° Client / Magasin / Produit")
st.markdown(
    """
Analyse interactive de vos données de transactions :

- Onglet **Client** : profil complet d’un client, magasin favori, achats récents, évolution des dépenses, catégories préférées.
- Onglet **Magasin** : performance d’un magasin (CA, clients, produits clés, top clients).
- Onglet **Produit** : stock par pays, ventes dans le temps, clients & pays qui achètent le plus.
"""
)

tab_client, tab_store, tab_product = st.tabs(
    ["👤 Client", "🏬 Magasin", "📦 Produit"]
)


# ========================== ONGLET CLIENT ==========================
with tab_client:
    clients_df = _get_client_list_cached()

    if clients_df.empty:
        st.error("Aucun client trouvé dans les données.")
    else:
        col_select, col_filters = st.columns([2, 1])

        with col_filters:
            st.subheader("Filtres")
            available_segments = sorted(clients_df["ClientSegment"].dropna().unique())
            selected_segments = st.multiselect(
                "Segment client",
                options=available_segments,
                default=available_segments[:5]
                if len(available_segments) > 5
                else available_segments,
            )

            available_countries = sorted(
                clients_df["ClientCountry"].dropna().unique()
            )
            selected_countries = st.multiselect(
                "Pays",
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
            st.subheader("Sélection du client")

            client_id_input = st.text_input(
                "Entrer un ClientID (copier-coller) ou choisir dans l’échantillon ci-dessous",
                value="",
                help="Collez ici un ClientID exact pour aller directement au bon client.",
            )

            sample_size = st.slider(
                "Taille de l’échantillon dans la liste déroulante",
                min_value=50,
                max_value=1000,
                value=200,
                step=50,
            )

            if filtered_clients.empty:
                st.warning("Aucun client ne correspond aux filtres sélectionnés.")
                client_id = None
            else:
                sample_clients = filtered_clients.sample(
                    n=min(sample_size, len(filtered_clients)), random_state=42
                )
                options = sample_clients["Label"].tolist()
                default_index = 0
                selection = st.selectbox(
                    "Client (échantillon)", options=options, index=default_index
                )
                sample_client_id = int(selection.split(" | ")[0])

                client_id = None
                if client_id_input.strip():
                    try:
                        client_id = int(client_id_input.strip())
                    except ValueError:
                        st.error("ClientID invalide, merci d’entrer un entier.")
                else:
                    client_id = sample_client_id

        if client_id is not None:
            overview = get_client_overview(client_id)

            st.markdown("---")
            st.subheader("Résumé client")

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
                    "Pays", value=str(country.iloc[0]) if not country.empty else "N/A"
                )
                st.metric("Transactions", value=int(overview["nb_transactions"]))

            with info_cols[2]:
                st.metric(
                    "Montant total dépensé (€)",
                    value=f"{overview['total_spent']:,.2f}".replace(",", " "),
                )
                st.metric(
                    "Nombre de produits distincts", value=int(overview["nb_products"])
                )

            with info_cols[3]:
                if overview["first_purchase"] is not None:
                    st.metric(
                        "1er achat",
                        value=overview["first_purchase"].strftime("%Y-%m-%d"),
                    )
                if overview["last_purchase"] is not None:
                    st.metric(
                        "Dernier achat",
                        value=overview["last_purchase"].strftime("%Y-%m-%d"),
                    )
                st.metric(
                    "Magasin favori (ID)",
                    value=str(overview["favorite_store_id"])
                    if overview["favorite_store_id"] is not None
                    else "N/A",
                )

            if overview["has_transactions"]:
                st.markdown("---")
                st.subheader("Détails des achats")

                tab_recent, tab_time, tab_categories = st.tabs(
                    [
                        "🧾 Achats récents",
                        "📈 Évolution des dépenses",
                        "🏷️ Catégories préférées",
                    ]
                )

                with tab_recent:
                    n_recent = st.slider(
                        "Nombre d'achats récents à afficher",
                        min_value=5,
                        max_value=50,
                        value=20,
                        step=5,
                    )
                    recent_df = get_client_recent_purchases(
                        client_id, n_recent=n_recent
                    )
                    if recent_df.empty:
                        st.info("Aucun achat trouvé pour ce client.")
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
                        "Fréquence d'agrégation",
                        options=["Mensuelle", "Hebdomadaire", "Quotidienne"],
                        horizontal=True,
                    )
                    freq_map = {
                        "Mensuelle": "M",
                        "Hebdomadaire": "W",
                        "Quotidienne": "D",
                    }
                    spend_df = get_client_spend_over_time(
                        client_id, freq=freq_map[freq_label]
                    )

                    if spend_df.empty:
                        st.info(
                            "Pas de série temporelle disponible pour ce client."
                        )
                    else:
                        st.line_chart(
                            spend_df.set_index("SaleTransactionDate")["TotalSpent"],
                            use_container_width=True,
                        )

                with tab_categories:
                    top_n = st.slider(
                        "Top N catégories", min_value=3, max_value=20, value=10
                    )
                    cat_df = get_client_favorite_categories(client_id, top_n=top_n)
                    if cat_df.empty:
                        st.info(
                            "Aucune information de catégorie disponible pour ce client."
                        )
                    else:
                        st.bar_chart(
                            cat_df.set_index("Category")["SalesNetAmountEuro"],
                            use_container_width=True,
                        )


# ========================== ONGLET MAGASIN ==========================
with tab_store:
    stores_df = _get_store_list_cached()

    if stores_df.empty:
        st.error("Aucun magasin trouvé dans les données.")
    else:
        st.subheader("Sélection du magasin")
        available_countries = sorted(stores_df["StoreCountry"].dropna().unique())
        selected_countries_store = st.multiselect(
            "Pays",
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
            st.warning("Aucun magasin ne correspond aux filtres sélectionnés.")
        else:
            selection_store = st.selectbox(
                "Magasin",
                options=filtered_stores["Label"].tolist(),
            )
            store_id = int(selection_store.split(" | ")[0])

            overview_store = get_store_overview(store_id)

            st.markdown("---")
            st.subheader("Résumé magasin")

            cols_store = st.columns(4)
            with cols_store[0]:
                st.metric("StoreID", value=str(store_id))
                st.metric("Pays", value=str(overview_store["country"]))
            with cols_store[1]:
                st.metric(
                    "CA total (€)",
                    value=f"{overview_store['total_spent']:,.2f}".replace(",", " "),
                )
                st.metric(
                    "Transactions", value=int(overview_store["nb_transactions"])
                )
            with cols_store[2]:
                st.metric(
                    "Clients distincts", value=int(overview_store["nb_clients"])
                )
                st.metric(
                    "Produits distincts", value=int(overview_store["nb_products"])
                )
            with cols_store[3]:
                if overview_store["first_purchase"] is not None:
                    st.metric(
                        "1er achat",
                        value=overview_store["first_purchase"].strftime("%Y-%m-%d"),
                    )
                if overview_store["last_purchase"] is not None:
                    st.metric(
                        "Dernier achat",
                        value=overview_store["last_purchase"].strftime("%Y-%m-%d"),
                    )

            if overview_store["has_transactions"]:
                st.markdown("---")
                st.subheader("Produits & clients clés")
                col_p, col_c = st.columns(2)

                with col_p:
                    top_n_prod = st.slider(
                        "Top N produits (par CA)", 3, 30, 10, key="store_top_products"
                    )
                    prod_df = get_store_top_products(store_id, top_n=top_n_prod)
                    if prod_df.empty:
                        st.info("Aucun produit trouvé pour ce magasin.")
                    else:
                        st.dataframe(
                            prod_df, use_container_width=True, hide_index=True
                        )

                with col_c:
                    top_n_clients = st.slider(
                        "Top N clients (par CA)", 3, 30, 10, key="store_top_clients"
                    )
                    cli_df = get_store_top_clients(store_id, top_n=top_n_clients)
                    if cli_df.empty:
                        st.info("Aucun client trouvé pour ce magasin.")
                    else:
                        st.dataframe(
                            cli_df, use_container_width=True, hide_index=True
                        )


# ========================== ONGLET PRODUIT ==========================
with tab_product:
    products_df = _get_product_list_cached()

    if products_df.empty:
        st.error("Aucun produit trouvé dans les données.")
    else:
        st.subheader("Sélection du produit")

        available_categories = sorted(products_df["Category"].dropna().unique())
        selected_categories = st.multiselect(
            "Catégories",
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
            st.warning("Aucun produit ne correspond aux filtres sélectionnés.")
        else:
            selection_product = st.selectbox(
                "Produit",
                options=filtered_products["Label"].tolist(),
            )
            product_id = int(selection_product.split(" | ")[0])

            overview_prod = get_product_overview(product_id)

            st.markdown("---")
            st.subheader("Résumé produit")

            cols_prod = st.columns(4)
            with cols_prod[0]:
                st.metric("ProductID", value=str(product_id))
                st.metric("Catégorie", value=str(overview_prod["category"]))
            with cols_prod[1]:
                st.metric("Famille L1", value=str(overview_prod["family1"]))
                st.metric("Famille L2", value=str(overview_prod["family2"]))
            with cols_prod[2]:
                st.metric(
                    "CA total (€)",
                    value=f"{overview_prod['total_revenue']:,.2f}".replace(",", " "),
                )
                st.metric(
                    "Quantité vendue", value=int(overview_prod["total_quantity"])
                )
            with cols_prod[3]:
                st.metric(
                    "Clients distincts", value=int(overview_prod["nb_clients"])
                )
                st.metric(
                    "Magasins distincts", value=int(overview_prod["nb_stores"])
                )

            if overview_prod["has_transactions"]:
                st.markdown("---")
                st.subheader("Stock & dynamique du produit")

                tab_stock, tab_time_p, tab_buyers = st.tabs(
                    [
                        "📦 Stock par pays",
                        "📈 Ventes dans le temps",
                        "🧑‍🤝‍🧑 Qui achète ?",
                    ]
                )

                with tab_stock:
                    stock_df = get_product_stock_by_country(product_id)
                    if stock_df.empty:
                        st.info("Pas d'information de stock trouvée pour ce produit.")
                    else:
                        st.bar_chart(
                            stock_df.set_index("StoreCountry")["Quantity"],
                            use_container_width=True,
                        )

                with tab_time_p:
                    freq_label_p = st.radio(
                        "Fréquence d'agrégation",
                        options=["Mensuelle", "Hebdomadaire", "Quotidienne"],
                        horizontal=True,
                        key="product_freq",
                    )
                    freq_map_p = {
                        "Mensuelle": "M",
                        "Hebdomadaire": "W",
                        "Quotidienne": "D",
                    }
                    sales_df = get_product_sales_over_time(
                        product_id, freq=freq_map_p[freq_label_p]
                    )
                    if sales_df.empty:
                        st.info(
                            "Pas de série temporelle disponible pour ce produit."
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
                            "Top N clients (par CA)",
                            3,
                            30,
                            10,
                            key="prod_top_clients",
                        )
                        cli_p_df = get_product_top_clients(
                            product_id, top_n=top_n_cli_p
                        )
                        if cli_p_df.empty:
                            st.info("Aucun client trouvé pour ce produit.")
                        else:
                            st.dataframe(
                                cli_p_df, use_container_width=True, hide_index=True
                            )

                    with col_cc:
                        top_n_ctry_p = st.slider(
                            "Top N pays (par CA)",
                            3,
                            30,
                            10,
                            key="prod_top_countries",
                        )
                        ctry_p_df = get_product_top_countries(
                            product_id, top_n=top_n_ctry_p
                        )
                        if ctry_p_df.empty:
                            st.info("Aucun pays trouvé pour ce produit.")
                        else:
                            st.bar_chart(
                                ctry_p_df.set_index("StoreCountry")[
                                    "SalesNetAmountEuro"
                                ],
                                use_container_width=True,
                            )

