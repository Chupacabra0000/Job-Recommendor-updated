import requests
from typing import Dict, List, Tuple, Optional

BASE_URL = "https://api.hh.ru"


def fetch_areas_tree() -> List[dict]:
    """
    Returns the full HH areas tree.
    """
    r = requests.get(f"{BASE_URL}/areas", timeout=30, headers={"Accept": "application/json"})
    r.raise_for_status()
    return r.json()


def _find_country(tree: List[dict], country_name: str = "Россия") -> Optional[dict]:
    for c in tree:
        if c.get("name") == country_name:
            return c
    return None


def list_regions_and_cities(tree: List[dict], country_name: str = "Россия") -> Tuple[List[dict], Dict[str, List[dict]]]:
    """
    Returns:
      regions: list of dicts {id, name}
      cities_by_region_id: dict[region_id] -> list of dicts {id, name}
    Strategy:
      - Regions = direct children of the country node.
      - Cities = flattened descendants of a region (leaf-ish nodes).
        HH tree depth varies, so we collect all descendants that have no further 'areas'
        OR are at the last level.
    """
    country = _find_country(tree, country_name=country_name)
    if not country:
        return [], {}

    regions = country.get("areas", []) or []
    regions_out = [{"id": str(r.get("id")), "name": r.get("name", "")} for r in regions if r.get("id")]

    def collect_cities(node: dict) -> List[dict]:
        out: List[dict] = []
        stack = [node]
        while stack:
            cur = stack.pop()
            kids = cur.get("areas", []) or []
            if not kids:
                if cur.get("id") and cur.get("name"):
                    out.append({"id": str(cur["id"]), "name": cur["name"]})
                continue
            # if node has children, keep traversing
            for k in kids:
                stack.append(k)
        # Deduplicate by id
        seen = set()
        unique = []
        for x in out:
            if x["id"] in seen:
                continue
            seen.add(x["id"])
            unique.append(x)
        # Sort by city name
        unique.sort(key=lambda x: x["name"])
        return unique

    cities_by_region_id: Dict[str, List[dict]] = {}
    for r in regions:
        rid = str(r.get("id"))
        cities_by_region_id[rid] = collect_cities(r)

    return regions_out, cities_by_region_id
