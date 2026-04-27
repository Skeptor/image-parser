import pytest
from httpx import AsyncClient, ASGITransport

from app.app import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.anyio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_parse_no_input(client):
    resp = await client.post("/parse")
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_parse_unsupported_file_type(client, tmp_path):
    fake_file = tmp_path / "schedule.txt"
    fake_file.write_bytes(b"some text")

    resp = await client.post(
        "/parse",
        files={"image": ("schedule.txt", fake_file.read_bytes(), "text/plain")},
    )
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_parse_invalid_url(client):
    resp = await client.post(
        "/parse",
        data={"url": "http://nonexistent.invalid/test.png"},
    )
    assert resp.status_code == 400
