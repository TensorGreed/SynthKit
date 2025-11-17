"""Utility helpers for creating HTTP sessions with sane retry defaults."""

from __future__ import annotations

from typing import Iterable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


def create_retry_session(
    *,
    total: int = 3,
    backoff_factor: float = 0.3,
    status_forcelist: Iterable[int] = (429, 500, 502, 503, 504),
) -> requests.Session:
    """
    Return a :class:`requests.Session` configured with retry semantics.

    Keeping this logic centralized ensures provider clients maintain consistent
    behavior (timeouts, retries, connection pooling) across deployments.
    """

    session = requests.Session()
    retry = Retry(
        total=total,
        read=total,
        connect=total,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=("POST", "GET"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session
