from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["Home.py", "pages/0_Overview.py", "pages/1_Shopping_Assistant.py", "pages/2_Catalog_Explorer.py", "pages/3_Batch_Test_Lab.py", "pages/4_Performance_Metrics.py",
         "pages/5_Security_Lab.py", "pages/6_Tools_and_Audit.py", "pages/7_Evaluation_Suite.py"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]


def test_assistant_runs_a_query():
    at = AppTest.from_file(str(ROOT / "pages/1_Shopping_Assistant.py"), default_timeout=90).run()
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("RECOMMENDED" in m.value or "Human review" in m.value for m in list(at.markdown) + list(at.warning))


def test_batch_lab_runs():
    at = AppTest.from_file(str(ROOT / "pages/3_Batch_Test_Lab.py"), default_timeout=180).run()
    run = [b for b in at.button if "Run batch" in b.label][0]
    run.click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "batch" in at.session_state
