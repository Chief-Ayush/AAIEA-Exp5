"""Catalog explorer + custom catalog upload (dataset verification, lab step 4)."""
import altair as alt
import pandas as pd
import streamlit as st

from ui.common import SERIES, bar_chart, current_store, page_header, page_setup, section, stat_tiles

page_setup("Product Catalog")

from cartiq.catalog import CatalogStore, REQUIRED_COLUMNS, validate_catalog  # noqa: E402

page_header("Product Catalog", "The canonical catalogue and the three marketplace views the SearchAgents query. "
            "Each marketplace holds a noisy, partial copy - that is why the SynchronizerAgent has to resolve conflicts.", eyebrow="Product")
store = current_store()
stt = store.stats()
stat_tiles([("Products", stt["products"]), ("Categories", stt["categories"]), ("Brands", stt["brands"]), ("Average rating", stt["avg_rating"]),
            ("Price range", f"₹{stt['price_min']:,} – ₹{stt['price_max']:,}")])
stat_tiles([(f"{s} records", n, f"{100 * n / stt['products']:.0f}% coverage, noisy copy") for s, n in stt["source_records"].items()])

t1, t2, t3 = st.tabs(["Browse", "Marketplace views", "Upload a catalogue"])
with t1:
    c = st.columns(3)
    cat = c[0].multiselect("Category", store.categories)
    br = c[1].multiselect("Brand", store.brands)
    mx = c[2].slider("Max price", 0, int(stt["price_max"]), int(stt["price_max"]), step=1000)
    df = store.canonical
    if cat: df = df[df["category"].isin(cat)]
    if br: df = df[df["brand"].isin(br)]
    df = df[df["price"] <= mx]
    st.dataframe(df, hide_index=True, width="stretch")
    g1, g2 = st.columns(2)
    cnt = store.canonical.groupby("category").size().reset_index(name="products")
    g1.altair_chart(bar_chart(cnt, "category", "products", "Products per category", fmt="d", y_title="products"), width="stretch")
    med = store.canonical.groupby("category")["price"].median().reset_index(name="median_price")
    g2.altair_chart(bar_chart(med, "category", "median_price", "Median price per category (₹)", fmt=",.0f", y_title="median price (₹)"), width="stretch")
with t2:
    s = st.selectbox("Marketplace", list(store.sources))
    st.dataframe(store.sources[s], hide_index=True, width="stretch")
    st.caption("Compared with the canonical catalogue, ratings differ slightly, some fields are missing, some prices are stale and a few items are wrongly shown as out of stock.")
with t3:
    st.write(f"Upload a CSV with at least: `{', '.join(REQUIRED_COLUMNS)}`. Optional: reviews, ram_gb, storage_gb, battery_hours, screen_inches, weight_kg, warranty_months, availability, features (slash separated).")
    up = st.file_uploader("Catalog CSV", type="csv")
    if up is not None:
        try:
            new = pd.read_csv(up)
            issues = validate_catalog(new)
            if issues:
                st.error("Catalog rejected: " + "; ".join(issues))
            else:
                st.success(f"Valid catalog: {len(new)} products")
                st.dataframe(new.head(20), hide_index=True, width="stretch")
                if st.button("Use this catalog for all pages", type="primary"):
                    st.session_state["custom_store"] = CatalogStore(new)
                    st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read the file: {exc}")
    if st.session_state.get("custom_store") is not None and st.button("Back to the built-in catalog"):
        st.session_state.pop("custom_store", None)
        st.rerun()
