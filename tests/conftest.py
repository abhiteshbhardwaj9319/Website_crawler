from __future__ import annotations

import os
from pathlib import Path

import pytest

from website_rag.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Isolated settings: temp data dir, fake keys, no .env file."""
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        retrieval_mode="hybrid",  # unit fixtures inject embeddings, not a downloaded reranker
        OPENAI_API_KEY="sk" + "-test-openai-000000000000",
        GROQ_API_KEY="gsk" + "_testgroq000000000000",
    )


def pytest_collection_modifyitems(config, items):  # noqa: ANN001
    if os.environ.get("RAG_LIVE_TESTS") == "1":
        return
    skip_live = pytest.mark.skip(reason="live test; set RAG_LIVE_TESTS=1 to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
