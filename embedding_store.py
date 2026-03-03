import os
import sqlite3
import time
from typing import Optional

import numpy as np

DB_PATH = os.getenv("EMB_DB_PATH", os.path.join("artifacts", "embeddings.sqlite3"))


def _ensure_dir() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def init_store() -> None:
    _ensure_dir()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS vacancy_embeddings (
            vacancy_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            dim INTEGER NOT NULL,
            emb BLOB NOT NULL,
            updated_at INTEGER NOT NULL,
            PRIMARY KEY (vacancy_id, model_name)
        )
        """
    )
    conn.commit()
    conn.close()


def get_embedding(vacancy_id: str, model_name: str) -> Optional[np.ndarray]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT dim, emb FROM vacancy_embeddings WHERE vacancy_id=? AND model_name=?",
        (vacancy_id, model_name),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    dim, blob = row
    vec = np.frombuffer(blob, dtype=np.float32)
    if vec.size != int(dim):
        return None
    return vec


def put_embedding(vacancy_id: str, model_name: str, vec: np.ndarray) -> None:
    vec = np.asarray(vec, dtype=np.float32)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO vacancy_embeddings(vacancy_id, model_name, dim, emb, updated_at)
        VALUES(?,?,?,?,?)
        """,
        (vacancy_id, model_name, int(vec.size), vec.tobytes(), int(time.time())),
    )
    conn.commit()
    conn.close()
