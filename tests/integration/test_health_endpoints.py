import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_healthz_is_open_and_ok(client: AsyncClient) -> None:
    r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_readyz_returns_ready_after_lifespan(client: AsyncClient) -> None:
    r = await client.get("/readyz")
    assert r.status_code == 200
    assert r.json() == {"status": "ready"}
