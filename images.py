"""Image search helpers.

Uses the Openverse API (https://api.openverse.org/) which is free, open, and
requires no API key. Falls back cleanly when a search returns nothing.
"""

from __future__ import annotations

import logging
import random
from typing import Iterable

import httpx

log = logging.getLogger("ai-discord-member.images")

OPENVERSE_URL = "https://api.openverse.org/v1/images/"
# Pick from the top-N so repeated queries don't spam the same photo.
TOP_N = 8
REQUEST_TIMEOUT = 8.0

_BAD_EXT = (".svg",)


def _looks_sendable(result: dict) -> bool:
    url = result.get("url") or ""
    if not url.startswith(("http://", "https://")):
        return False
    if url.lower().endswith(_BAD_EXT):
        return False
    filetype = (result.get("filetype") or "").lower()
    if filetype and filetype not in {"jpg", "jpeg", "png", "gif", "webp"}:
        return False
    return True


async def search_image(
    query: str, *, client: httpx.AsyncClient | None = None
) -> str | None:
    """Return an image URL that matches ``query``, or ``None`` if nothing works."""
    query = (query or "").strip()
    if not query:
        return None

    params = {
        "q": query,
        "page_size": str(max(TOP_N, 3)),
        "license_type": "all",
    }
    headers = {"Accept": "application/json", "User-Agent": "ai-discord-member/1.0"}

    owns_client = client is None
    try:
        if owns_client:
            client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
        assert client is not None
        resp = await client.get(OPENVERSE_URL, params=params, headers=headers)
    except httpx.HTTPError as exc:
        log.warning("Openverse request failed for %r: %s", query, exc)
        return None
    finally:
        if owns_client and client is not None:
            await client.aclose()

    if resp.status_code >= 400:
        log.warning("Openverse returned %s for %r", resp.status_code, query)
        return None

    try:
        data = resp.json()
    except ValueError:
        log.warning("Openverse gave non-JSON for %r", query)
        return None

    results: Iterable[dict] = data.get("results") or []
    sendable = [r for r in results if _looks_sendable(r)]
    if not sendable:
        log.info("No sendable Openverse results for %r", query)
        return None

    pick = random.choice(sendable[:TOP_N])
    url = pick["url"]
    log.info("Image pick for %r: %s", query, url)
    return url
