import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cartiq.config import Settings  # noqa: E402
from cartiq.pipeline import ShoppingAssistant  # noqa: E402

LAB = "Find laptops under ₹70,000 with at least 16 GB RAM, 512 GB storage and rating above 4.2"


@pytest.fixture(scope="session")
def asst():
    return ShoppingAssistant(Settings(source_latency_s=0.01, analysis_latency_s=0.01, retry_backoff_s=0.0))


@pytest.fixture(scope="session")
def lab_query():
    return LAB
