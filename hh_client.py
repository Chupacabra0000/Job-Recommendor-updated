import os
import time
import requests
from typing import Any, Dict, List, Optional

BASE_URL = "https://api.hh.ru"

def _headers() -> Dict[str, str]:
    """
    HH can reject placeholder/bot-like User-Agent values.
    Use a REAL, identifying string (app + version + real email).
    """
    default_ua = "JobRecommendorHH/1.0 (rana.shoaib7777@gmail.com)"
    ua = os.getenv("HH_USER_AGENT", default_ua)

    return {
        # HH processes HH-User-Agent (this is the one that matters)
        "HH-User-Agent": ua,
        "Accept": "application/json",
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
    }

def _get(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    r = requests.get(url, params=params or {}, headers=_headers(), timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"HH API error {r.status_code}: {r.text}")
    return r.json()

def search_vacancies(
    text: str,
    area: int,
    page: int = 0,
    per_page: int = 50,
    period_days: Optional[int] = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "text": text,
        "area": int(area),
        "page": int(page),
        "per_page": int(per_page),
    }
    if period_days is not None:
        params["period"] = int(period_days)

    return _get(f"{BASE_URL}/vacancies", params=params)

def fetch_vacancies(
    text: str,
    area: int,
    max_items: int = 500,
    per_page: int = 50,
    period_days: Optional[int] = None,
    sleep_s: float = 0.25,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    page = 0

    while len(items) < max_items:
        payload = search_vacancies(
            text=text,
            area=area,
            page=page,
            per_page=per_page,
            period_days=period_days,
        )
        batch = payload.get("items") or []
        if not batch:
            break

        items.extend(batch)

        # Stop if the last page is shorter
        if len(batch) < per_page:
            break

        page += 1
        time.sleep(max(0.0, sleep_s))

        # If HH returns total pages, stop at the end
        pages = payload.get("pages")
        if pages is not None and page >= int(pages):
            break

    return items[:max_items]

def vacancy_details(vacancy_id: str) -> Dict[str, Any]:
    return _get(f"{BASE_URL}/vacancies/{vacancy_id}", params={})
