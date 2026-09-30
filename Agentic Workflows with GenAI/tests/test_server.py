"""Tests for the FastAPI server."""

import pytest
from fastapi.testclient import TestClient
from src.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "agentic-workflow-api"}


def test_static_index_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "GenAI Agentic Workflow System" in response.text
