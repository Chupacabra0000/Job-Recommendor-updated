import os
import json
import time
import hashlib
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

CACHE_DIR = os.getenv("HH_CACHE_DIR", os.path.join("artifacts", "hh_cache"))

def _ensure_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)

def make_key(params: Dict[str, Any]) -> str:
    # stable hash of params
    s = json.dumps(params, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def paths(key: str) -> Tuple[str, str, str]:
    df_path = os.path.join(CACHE_DIR, f"{key}.parquet")
    emb_path = os.path.join(CACHE_DIR, f"{key}.npy")
    meta_path = os.path.join(CACHE_DIR, f"{key}.json")
    return df_path, emb_path, meta_path

def save(key: str, df: pd.DataFrame, embeddings: np.ndarray, meta: Dict[str, Any]) -> None:
    _ensure_dir()
    df_path, emb_path, meta_path = paths(key)
    df.to_parquet(df_path, index=False)
    np.save(emb_path, embeddings.astype("float32"))
    meta = dict(meta or {})
    meta["saved_at"] = int(time.time())
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_if_fresh(key: str, ttl_seconds: int) -> Optional[Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]]:
    df_path, emb_path, meta_path = paths(key)
    if not (os.path.exists(df_path) and os.path.exists(emb_path) and os.path.exists(meta_path)):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        saved_at = int(meta.get("saved_at", 0))
        if ttl_seconds > 0 and (int(time.time()) - saved_at) > int(ttl_seconds):
            return None
        df = pd.read_parquet(df_path)
        emb = np.load(emb_path).astype("float32")
        return df, emb, meta
    except Exception:
        return None
