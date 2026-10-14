"""CartIQ entry point: `streamlit run Home.py`.  Defines the navigation; every page lives in pages/."""
import streamlit as st

st.set_page_config(page_title="CartIQ", layout="wide")

from ui.common import page_setup, system_panel  # noqa: E402

page_setup()

nav = st.navigation({
    "Overview": [st.Page("pages/0_Overview.py", title="Overview", icon=":material/dashboard:", default=True)],
    "Product": [st.Page("pages/1_Shopping_Assistant.py", url_path="assistant", title="Shopping Assistant", icon=":material/shopping_cart:"),
                st.Page("pages/2_Catalog_Explorer.py", url_path="catalog", title="Product Catalog", icon=":material/inventory_2:")],
    "Evaluation": [st.Page("pages/3_Batch_Test_Lab.py", url_path="batch-testing", title="Batch Testing", icon=":material/playlist_play:"),
                   st.Page("pages/7_Evaluation_Suite.py", url_path="evaluation-suite", title="Evaluation Suite", icon=":material/fact_check:"),
                   st.Page("pages/4_Performance_Metrics.py", url_path="evaluation-report", title="Evaluation Report", icon=":material/analytics:")],
    "Governance": [st.Page("pages/5_Security_Lab.py", url_path="security", title="Security Testing", icon=":material/shield:"),
                   st.Page("pages/6_Tools_and_Audit.py", url_path="tools", title="Tools and Audit Log", icon=":material/build:")],
})
nav.run()
system_panel()
