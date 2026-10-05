"""API contract in artifact mode -- the configuration a fresh clone gets.

No dataset, no checkpoint: the service must still boot and answer every endpoint off
the committed JSON, because that is what makes the repo demo itself.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(artifact_paths):
    from api.main import create_app
    from api.state import reset_backend

    reset_backend(artifact_paths)
    with TestClient(create_app()) as c:
        yield c


def test_health_reports_artifact_mode(client):
    body = client.get("/api/health").json()
    assert body["mode"] == "artifact"
    assert body["status"] == "ready"
    assert body["model_loaded"] is False
    assert body["thresholds"]["decision"] == 0.5
    assert "overview" in body["artifacts"]


def test_root_is_a_service_descriptor(client):
    body = client.get("/").json()
    assert body["service"] == "fraudlens"
    assert body["health"] == "/api/health"


def test_overview_served_from_artifacts(client):
    body = client.get("/api/overview").json()
    assert body["dataset"]["nodes"] == 24
    assert body["synthetic"] is True


def test_node_detail(client):
    body = client.get("/api/nodes/0").json()
    assert body["tx_id"] == 100_000
    assert body["label"] == "illicit"
    assert body["neighbors"]


def test_node_outside_the_demo_subset_is_404(client):
    r = client.get("/api/nodes/9999")
    assert r.status_code == 404
    assert "demo subset" in r.json()["detail"]


def test_search_resolves_both_index_and_tx_id(client):
    by_idx = client.get("/api/search", params={"q": "0"}).json()
    assert by_idx["match"]["idx"] == 0

    by_tx = client.get("/api/search", params={"q": "100000"}).json()
    assert by_tx["match"]["idx"] == 0

    miss = client.get("/api/search", params={"q": "424242"}).json()
    assert miss["match"] is None


def test_rings_list_and_detail(client):
    listing = client.get("/api/rings").json()
    assert listing["stats"]["count"] == 1
    assert listing["rings"][0]["community_id"] == 3
    # The list view must not ship the heavy per-ring subgraph payload.
    assert "edges" not in listing["rings"][0]

    detail = client.get("/api/rings/3").json()
    assert detail["edges"] == [[0, 1], [1, 2]]

    assert client.get("/api/rings/404").status_code == 404


def test_rings_limit_is_applied(client):
    assert len(client.get("/api/rings", params={"limit": 1}).json()["rings"]) == 1


def test_explain_candidates_and_detail(client):
    cands = client.get("/api/explain/candidates").json()["candidates"]
    assert cands[0]["idx"] == 0

    body = client.get("/api/explain/0").json()
    assert body["driver"] == "network"
    assert body["top_features"][0]["group"] == "aggregated"


def test_explain_unknown_node_explains_why_in_the_error(client):
    r = client.get("/api/explain/777")
    assert r.status_code == 404
    assert "precomputed" in r.json()["detail"]


def test_graph_sample_filters_edges_to_returned_nodes(client):
    body = client.get("/api/graph/sample", params={"n": 100}).json()
    keep = {n["idx"] for n in body["nodes"]}
    assert all(a in keep and b in keep for a, b in body["edges"])


def test_graph_sample_rejects_out_of_range_sizes(client):
    # n is bounded so a stray request cannot ask the live backend for the whole graph.
    assert client.get("/api/graph/sample", params={"n": 1}).status_code == 422
    assert client.get("/api/graph/sample", params={"n": 99999}).status_code == 422


def test_subgraph_falls_back_to_neighbours(client):
    body = client.get("/api/nodes/0/subgraph").json()
    assert "nodes" in body and "edges" in body
