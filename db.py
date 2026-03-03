import sqlite3
import os
import base64
import hashlib
import hmac
import datetime
from typing import Optional, List, Dict, Any, Tuple

DB_PATH = os.getenv("APP_DB_PATH", "app.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            text TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            job_id TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, job_id),
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            expires_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    conn.commit()
    conn.close()


# ---------------- Password hashing (stdlib: PBKDF2) ----------------
# stored format: pbkdf2_sha256$<iterations>$<salt_b64>$<hash_b64>

def _pbkdf2_hash(password: str, salt: bytes, iterations: int = 200_000) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    iterations = 200_000
    dk = _pbkdf2_hash(password, salt, iterations)
    return "pbkdf2_sha256$%d$%s$%s" % (
        iterations,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(dk).decode("ascii"),
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters_s, salt_b64, hash_b64 = stored.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        iterations = int(iters_s)
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected = base64.b64decode(hash_b64.encode("ascii"))
        dk = _pbkdf2_hash(password, salt, iterations)
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


# ---------------- Users ----------------
def create_user(email: str, password: str) -> Tuple[bool, str]:
    email = (email or "").strip().lower()
    if not email or "@" not in email:
        return False, "Некорректный email."
    if not password or len(password) < 6:
        return False, "Пароль должен быть минимум 6 символов."
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users(email, password_hash) VALUES(?, ?)",
            (email, hash_password(password)),
        )
        conn.commit()
        return True, "Аккаунт создан."
    except sqlite3.IntegrityError:
        return False, "Пользователь с таким email уже существует."
    finally:
        conn.close()


def authenticate(email: str, password: str) -> Optional[Dict[str, Any]]:
    email = (email or "").strip().lower()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, email, password_hash FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    if not verify_password(password, row["password_hash"]):
        return None
    return {"id": row["id"], "email": row["email"]}


# ---------------- Resumes ----------------
def list_resumes(user_id: int) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, text, created_at FROM resumes WHERE user_id = ? ORDER BY created_at DESC, id DESC",
        (user_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_resume(user_id: int, name: str, text: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO resumes(user_id, name, text) VALUES(?, ?, ?)",
        (user_id, name, text),
    )
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return int(rid)


def delete_resume(user_id: int, resume_id: int) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM resumes WHERE user_id = ? AND id = ?", (user_id, resume_id))
    conn.commit()
    conn.close()


# ---------------- Favorites ----------------
def list_favorites(user_id: int) -> List[str]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT job_id FROM favorites WHERE user_id = ? ORDER BY created_at DESC, id DESC", (user_id,))
    rows = cur.fetchall()
    conn.close()
    return [r["job_id"] for r in rows]


def add_favorite(user_id: int, job_id: str) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO favorites(user_id, job_id) VALUES(?, ?)", (user_id, job_id))
    conn.commit()
    conn.close()


def remove_favorite(user_id: int, job_id: str) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM favorites WHERE user_id = ? AND job_id = ?", (user_id, job_id))
    conn.commit()
    conn.close()

# ---------------- Sessions (simple persistent login) ----------------
# We store a random token in the URL query params. This is for convenience/demo.
# In production you'd want secure cookies + HTTPS + CSRF protections.

def create_session(user_id: int, days_valid: int = 30) -> str:
    token = base64.urlsafe_b64encode(os.urandom(24)).decode('ascii').rstrip('=')
    expires_at = (datetime.datetime.utcnow() + datetime.timedelta(days=days_valid)).isoformat(timespec='seconds')
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO sessions(token, user_id, expires_at) VALUES(?,?,?)',
        (token, int(user_id), expires_at),
    )
    conn.commit()
    conn.close()
    return token

def get_user_by_token(token: str) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        '''
        SELECT u.id, u.email
        FROM sessions s
        JOIN users u ON u.id = s.user_id
        WHERE s.token = ?
          AND s.expires_at > ?
        '''
        , (token, datetime.datetime.utcnow().isoformat(timespec='seconds'))
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {'id': row['id'], 'email': row['email']}

def delete_session(token: str) -> None:
    if not token:
        return
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('DELETE FROM sessions WHERE token = ?', (token,))
    conn.commit()
    conn.close()
