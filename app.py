import math
import re
import time
import html as _html
from typing import Dict, List

import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from db import (
    init_db, create_user, authenticate,
    list_resumes, create_resume, delete_resume,
    list_favorites, add_favorite, remove_favorite,
)
from hh_client import fetch_vacancies, vacancy_details

from tfidf_terms import extract_terms
from embedding_store import init_store, get_embedding, put_embedding

# ---------------------------- Constants ----------------------------
COL_JOB_ID = "Job Id"
COL_WORKPLACE = "workplace"
COL_MODE = "working_mode"
COL_SALARY = "salary"
COL_POSITION = "position"
COL_DUTIES = "job_role_and_duties"
COL_SKILLS = "requisite_skill"
COL_DESC = "offer_details"

DEFAULT_AREA_NOVOSIBIRSK = 4
DEFAULT_QUERY = "Python"

DEFAULT_STARTUP_LIMIT = 500  # show list without embeddings
CACHE_TTL_SECONDS = 60 * 60  # 60 min

# multi-query settings
PER_TERM = 50
TERMS_MIN = 6
TERMS_MAX = 10

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_DESC_CHARS = 2500  # truncate descriptions for embedding speed

# ---------------------------- Page config ----------------------------
st.set_page_config(page_title="HH Job Recommender", page_icon="💼", layout="wide")

# ---------------------------- Init DB + embedding store ----------------------------
init_db()
init_store()

# ---------------------------- UI Styles ----------------------------
st.markdown(
    """
    <style>
      .small-muted { color: rgba(49,51,63,.6); font-size: 0.9rem; }
      .card { border: 1px solid rgba(49,51,63,.15); border-radius: 14px; padding: 16px 18px; margin-bottom: 14px; }
      .pill { display: inline-block; padding: 2px 10px; border-radius: 999px;
              border: 1px solid rgba(49,51,63,.2); margin-right: 6px; margin-top: 6px; font-size: 0.85rem; }
      .pill-strong { font-weight: 600; }
      .snippet { color: rgba(49,51,63,.82); font-size: 0.95rem; margin-top: 8px; }
      .skill-chip { display: inline-block; margin: 4px 6px 0 0; padding: 2px 8px; border-radius: 999px;
                    background: rgba(49,51,63,.06); font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------- Session State ----------------------------
if "user" not in st.session_state:
    st.session_state.user = None
if "page" not in st.session_state:
    st.session_state.page = 1
if "page_size" not in st.session_state:
    st.session_state.page_size = 20
if "resume_source" not in st.session_state:
    st.session_state.resume_source = "None"
if "selected_resume_id" not in st.session_state:
    st.session_state.selected_resume_id = None
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "details_cache" not in st.session_state:
    st.session_state.details_cache = {}  # vacancy_id -> full_desc_text


# ---------------------------- Helpers ----------------------------
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\n".join([page.get_text("text") for page in doc]).strip()


def _strip_html(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = _html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _truncate(s: str, n: int = MAX_DESC_CHARS) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[:n].rstrip() + "…"


def _job_text(row: pd.Series) -> str:
    parts = [
        str(row.get(COL_POSITION, "") or ""),
        str(row.get(COL_WORKPLACE, "") or ""),
        str(row.get(COL_MODE, "") or ""),
        str(row.get(COL_SALARY, "") or ""),
        str(row.get(COL_SKILLS, "") or ""),
        str(row.get(COL_DUTIES, "") or ""),
        _truncate(str(row.get(COL_DESC, "") or "")),
    ]
    s = " ".join(p for p in parts if p)
    return re.sub(r"\s+", " ", s).strip()


def _items_to_df(items: List[dict]) -> pd.DataFrame:
    rows = []
    for it in items:
        vid = str(it.get("id", "")).strip()
        title = (it.get("name") or "").strip()
        employer = (it.get("employer") or {}).get("name") or ""
        schedule = it.get("schedule") or {}
        working_mode = schedule.get("name", "")

        sal = it.get("salary")
        salary_text = ""
        if isinstance(sal, dict):
            s_from, s_to, cur = sal.get("from"), sal.get("to"), sal.get("currency")
            if s_from is not None and s_to is not None:
                salary_text = f"{s_from}–{s_to} {cur}"
            elif s_from is not None:
                salary_text = f"от {s_from} {cur}"
            elif s_to is not None:
                salary_text = f"до {s_to} {cur}"

        snippet = it.get("snippet") or {}
        duties = (snippet.get("responsibility") or "").strip() if isinstance(snippet, dict) else ""
        skills = (snippet.get("requirement") or "").strip() if isinstance(snippet, dict) else ""

        rows.append(
            {
                COL_JOB_ID: vid,
                COL_WORKPLACE: employer,
                COL_MODE: working_mode,
                COL_SALARY: salary_text,
                COL_POSITION: title,
                COL_DUTIES: duties,
                COL_SKILLS: skills,
                COL_DESC: "",
                "alternate_url": it.get("alternate_url", ""),
                "published_at": it.get("published_at", ""),
            }
        )
    df = pd.DataFrame(rows)
    if len(df):
        df["job_text"] = df.apply(_job_text, axis=1)
    else:
        df["job_text"] = ""
    return df


def _dedupe_merge(list_of_items: List[List[dict]]) -> List[dict]:
    seen = set()
    out = []
    for items in list_of_items:
        for it in items:
            vid = str(it.get("id", "")).strip()
            if not vid or vid in seen:
                continue
            seen.add(vid)
            out.append(it)
    return out


def _chips(skills_text: str, limit: int = 10) -> List[str]:
    if not skills_text:
        return []
    parts = re.split(r"[;,]\s*|\n+", skills_text)
    out = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
        if len(out) >= limit:
            break
    return out


def _snippet(s: str, n: int = 230) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


# ---------------------------- Cached fetchers ----------------------------
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _fetch_default_startup(area: int) -> List[dict]:
    return fetch_vacancies(text=DEFAULT_QUERY, area=area, max_items=DEFAULT_STARTUP_LIMIT, per_page=50, period_days=None)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _fetch_term(area: int, term: str, per_term: int) -> List[dict]:
    # period_days narrows noise; adjust if you want more recall
    return fetch_vacancies(text=term, area=area, max_items=per_term, per_page=per_term, period_days=14)


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _fetch_details(vacancy_id: str) -> str:
    full = vacancy_details(vacancy_id)
    return _strip_html(full.get("description") or "")


# ---------------------------- Embeddings + FAISS ranking ----------------------------
def _build_embeddings_for_df(df: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    dim = model.get_sentence_embedding_dimension()
    embs = np.zeros((len(df), dim), dtype=np.float32)

    missing_texts = []
    missing_idx = []

    for i, row in df.iterrows():
        vid = str(row[COL_JOB_ID])
        cached = get_embedding(vid, MODEL_NAME)
        if cached is not None and cached.size == dim:
            embs[i] = cached
        else:
            missing_idx.append(i)
            missing_texts.append(str(row["job_text"] or ""))

    if missing_texts:
        new_emb = model.encode(missing_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        new_emb = np.asarray(new_emb, dtype=np.float32)
        for j, i in enumerate(missing_idx):
            embs[i] = new_emb[j]
            put_embedding(str(df.loc[i, COL_JOB_ID]), MODEL_NAME, new_emb[j])

    # normalize defensively
    denom = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    return embs / denom


def _rank_with_faiss(embs: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    import faiss  # faiss-cpu
    d = int(embs.shape[1])
    index = faiss.IndexFlatIP(d)  # cosine == inner product for normalized vectors
    index.add(embs.astype(np.float32))
    scores, idx = index.search(query_vec.astype(np.float32), embs.shape[0])

    score_arr = np.zeros((embs.shape[0],), dtype=np.float32)
    score_arr[idx[0]] = scores[0]
    return score_arr


# ---------------------------- Auth UI ----------------------------
def auth_screen():
    st.markdown("## 🔐 Вход / Регистрация")
    t1, t2 = st.tabs(["Вход", "Регистрация"])
    with t1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Пароль", type="password", key="login_password")
        if st.button("Войти", use_container_width=True):
            user = authenticate(email, password)
            if not user:
                st.error("Неверный email или пароль.")
            else:
                st.session_state.user = user
                st.rerun()
    with t2:
        email_r = st.text_input("Email", key="reg_email")
        p1 = st.text_input("Пароль (мин. 6 символов)", type="password", key="reg_password")
        p2 = st.text_input("Повторите пароль", type="password", key="reg_password2")
        if st.button("Создать аккаунт", use_container_width=True):
            if p1 != p2:
                st.error("Пароли не совпадают.")
            else:
                ok, msg = create_user(email_r, p1)
                st.success(msg) if ok else st.error(msg)


if st.session_state.user is None:
    auth_screen()
    st.stop()

user_id = int(st.session_state.user["id"])

# ---------------------------- Sidebar ----------------------------
st.sidebar.title("⚙️ Настройки")
if st.sidebar.button("🚪 Выйти", use_container_width=True):
    st.session_state.user = None
    st.rerun()

st.sidebar.subheader("HH.ru")
area = st.sidebar.number_input("Регион (area id)", min_value=1, value=int(DEFAULT_AREA_NOVOSIBIRSK), step=1)

st.sidebar.subheader("Запросы для HH")
mode = st.sidebar.radio("Как выбирать термины?", ["TF-IDF из резюме (рекомендовано)", "Ввести вручную"], index=0)
manual_terms = ""
if mode == "Ввести вручную":
    manual_terms = st.sidebar.text_area(
        "Термины (по одному в строке)",
        placeholder="python\nsql\nairflow\netl\npandas\ndocker",
        height=140,
    )

st.sidebar.subheader("Резюме")
resume_source = st.sidebar.radio("Источник резюме", ["None", "PDF resume", "Created resume"], index=0)
st.session_state.resume_source = resume_source

pdf_text = ""
if resume_source == "PDF resume":
    pdf = st.sidebar.file_uploader("Загрузите PDF", type=["pdf"])
    if pdf is not None:
        with st.spinner("Читаем PDF..."):
            pdf_text = extract_text_from_pdf(pdf.read())
            st.session_state.pdf_text = pdf_text

resumes = list_resumes(user_id)
selected_resume_text = ""
if resume_source == "Created resume" and resumes:
    opts = {f'{r["name"]} (#{r["id"]})': r["id"] for r in resumes}
    label = st.sidebar.selectbox("Выберите резюме", list(opts.keys()))
    rid = opts[label]
    sel = next((x for x in resumes if x["id"] == rid), None)
    selected_resume_text = sel["text"] if sel else ""

resume_text = ""
if resume_source == "PDF resume":
    resume_text = st.session_state.pdf_text
elif resume_source == "Created resume":
    resume_text = selected_resume_text

has_resume = bool((resume_text or "").strip())

st.sidebar.subheader("Показ")
search_in_results = st.sidebar.text_input("Поиск по выдаче", value="")
page_size = st.sidebar.selectbox("На странице", [10, 20, 50, 100], index=1)
st.session_state.page_size = int(page_size)

# ---------------------------- Header ----------------------------
st.title("💼 HH.ru Job Recommender — Новосибирск")
st.caption(
    "На старте показываем 500 дефолтных вакансий без эмбеддингов. "
    "После выбора/загрузки резюме включается TF‑IDF → multi-query → FAISS ранжирование."
)

favorites = set(list_favorites(user_id))

# ---------------------------- Main logic ----------------------------
if not has_resume:
    with st.spinner("Загружаем дефолтные вакансии (без эмбеддингов)..."):
        items = _fetch_default_startup(int(area))
        df = _items_to_df(items)
    df["similarity_score"] = pd.NA
else:
    # terms
    if mode == "Ввести вручную":
        terms = [t.strip() for t in manual_terms.splitlines() if t.strip()]
    else:
        terms = extract_terms(resume_text, top_k=TERMS_MAX)

    terms = terms[:TERMS_MAX]
    if len(terms) < TERMS_MIN:
        terms = list(dict.fromkeys(terms + ["python", "sql"]))[:TERMS_MIN]

    st.write("**Термины для HH:**", ", ".join(terms))

    with st.spinner("Fetching вакансий по терминам (multi-query)..."):
        batches = [_fetch_term(int(area), term, PER_TERM) for term in terms]
        merged = _dedupe_merge(batches)
        df = _items_to_df(merged)

    model = SentenceTransformer(MODEL_NAME)

    with st.spinner("Эмбеддинги вакансий (reuse by vacancy_id) + FAISS ранжирование..."):
        job_embs = _build_embeddings_for_df(df, model)
        q = model.encode([resume_text], normalize_embeddings=True)
        q = np.asarray(q, dtype=np.float32)
        scores = _rank_with_faiss(job_embs, q)
        df["similarity_score"] = scores
        df = df.sort_values("similarity_score", ascending=False).reset_index(drop=True)

    # preload top-10 descriptions
    top_n = min(10, len(df))
    with st.spinner("Подгружаем полные описания только для TOP-10..."):
        for i in range(top_n):
            vid = str(df.loc[i, COL_JOB_ID])
            if vid and vid not in st.session_state.details_cache:
                try:
                    st.session_state.details_cache[vid] = _fetch_details(vid)
                except Exception:
                    st.session_state.details_cache[vid] = ""

# Apply in-results search
if search_in_results.strip():
    q = search_in_results.strip().lower()
    mask = (
        df[COL_POSITION].fillna("").str.lower().str.contains(re.escape(q), na=False)
        | df[COL_WORKPLACE].fillna("").str.lower().str.contains(re.escape(q), na=False)
        | df[COL_SKILLS].fillna("").str.lower().str.contains(re.escape(q), na=False)
    )
    df = df[mask].reset_index(drop=True)

total = len(df)
st.caption(f"Вакансий: {total}")

# Pagination
total_pages = max(1, math.ceil(total / st.session_state.page_size))
st.session_state.page = max(1, min(st.session_state.page, total_pages))
start = (st.session_state.page - 1) * st.session_state.page_size
end = start + st.session_state.page_size
page_df = df.iloc[start:end]

# Render cards
def render_job(row: Dict, idx: int):
    vid = str(row.get(COL_JOB_ID, "") or "")
    title = str(row.get(COL_POSITION, "") or "Untitled")
    company = str(row.get(COL_WORKPLACE, "") or "")
    mode_ = str(row.get(COL_MODE, "") or "")
    sal = str(row.get(COL_SALARY, "") or "")
    url = str(row.get("alternate_url", "") or "")
    pub = str(row.get("published_at", "") or "")
    score = row.get("similarity_score", None)

    pct = None
    if score is not None and pd.notna(score):
        pct = max(0, min(100, int(round(float(score) * 100))))

    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([4, 1.2, 1.2])

    with c1:
        st.markdown(f"### {idx}. {title}", unsafe_allow_html=True)
        pills = []
        if company:
            pills.append(f'<span class="pill pill-strong">{company}</span>')
        if mode_:
            pills.append(f'<span class="pill">{mode_}</span>')
        if sal:
            pills.append(f'<span class="pill">{sal}</span>')
        if pub:
            pills.append(f'<span class="pill">{pub[:10]}</span>')
        if url:
            pills.append(f'<span class="pill"><a href="{url}" target="_blank">hh.ru</a></span>')
        if pills:
            st.markdown(" ".join(pills), unsafe_allow_html=True)

        st.markdown(f"<div class='snippet'>{_snippet(row.get('job_text',''))}</div>", unsafe_allow_html=True)

        chips = _chips(str(row.get(COL_SKILLS, "") or ""), limit=10)
        if chips:
            st.markdown("".join([f"<span class='skill-chip'>{c}</span>" for c in chips]), unsafe_allow_html=True)

    with c2:
        if pct is not None:
            st.metric("Сходство", f"{pct}%")
            st.progress(pct)
        else:
            st.metric("Сходство", "—")

    with c3:
        if vid:
            is_fav = vid in favorites
            label = "⭐ Удалить" if is_fav else "☆ В избранное"
            if st.button(label, use_container_width=True, key=f"fav_{vid}_{idx}"):
                if is_fav:
                    remove_favorite(user_id, vid)
                else:
                    add_favorite(user_id, vid)
                st.rerun()

    with st.expander("Подробнее"):
        full_desc = ""
        if vid and vid in st.session_state.details_cache:
            full_desc = st.session_state.details_cache[vid]
        elif vid and has_resume and idx > 10:
            with st.spinner("Подгружаем полное описание..."):
                try:
                    full_desc = _fetch_details(vid)
                except Exception:
                    full_desc = ""
                st.session_state.details_cache[vid] = full_desc

        if full_desc:
            st.write(full_desc)
        else:
            st.caption("Полное описание не загружено (или отсутствует).")

    st.markdown("</div>", unsafe_allow_html=True)

for i, row in enumerate(page_df.to_dict(orient="records"), start=1):
    render_job(row, start + i)

# Pagination controls
c1, c2, c3 = st.columns([1, 2, 1])
with c1:
    if st.button("⬅️", disabled=st.session_state.page <= 1):
        st.session_state.page -= 1
        st.rerun()
with c2:
    st.write(f"Страница {st.session_state.page} / {total_pages}")
with c3:
    if st.button("➡️", disabled=st.session_state.page >= total_pages):
        st.session_state.page += 1
        st.rerun()
