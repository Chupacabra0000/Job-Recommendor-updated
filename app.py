import math
import re
import html as _html
import hashlib
from typing import Dict, List

import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import pandas as pd

from db import (
    init_db, create_user, authenticate,
    list_resumes, list_favorites, add_favorite, remove_favorite,
    create_session, get_user_by_token, delete_session
)

from hh_client import fetch_vacancies, vacancy_details
from sentence_transformers import SentenceTransformer

from tfidf_terms import extract_terms
from embedding_store import init_store, get_embedding, put_embedding
from hh_areas import fetch_areas_tree, list_regions_and_cities

# ---------- constants ----------
COL_JOB_ID = "Job Id"
COL_WORKPLACE = "workplace"
COL_MODE = "working_mode"
COL_SALARY = "salary"
COL_POSITION = "position"
COL_DUTIES = "job_role_and_duties"
COL_SKILLS = "requisite_skill"
COL_DESC = "offer_details"

DEFAULT_QUERY = "Python"
DEFAULT_STARTUP_LIMIT = 500

PER_TERM = 50
TERMS_MIN = 6
TERMS_MAX = 10

CACHE_TTL_SECONDS = 60 * 60  # 60 min
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_DESC_CHARS = 2500

# ---------- page ----------
st.set_page_config(page_title="HH Job Recommender", page_icon="💼", layout="wide")
init_db()
init_store()

# ---------- styles ----------
st.markdown(
    """
    <style>
      .center-wrap { max-width: 520px; margin: 0 auto; }
      .card { border: 1px solid rgba(49,51,63,.15); border-radius: 14px; padding: 16px 18px; margin-bottom: 14px; }
      .pill { display: inline-block; padding: 2px 10px; border-radius: 999px;
              border: 1px solid rgba(49,51,63,.2); margin-right: 6px; margin-top: 6px; font-size: 0.85rem; }
      .pill-strong { font-weight: 600; }
      .snippet { color: rgba(49,51,63,.82); font-size: 0.95rem; margin-top: 8px; }
      .skill-chip { display: inline-block; margin: 4px 6px 0 0; padding: 2px 8px; border-radius: 999px;
                    background: rgba(49,51,63,.06); font-size: 0.85rem; }
      .muted { color: rgba(49,51,63,.65); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- state ----------
if "user" not in st.session_state:
    st.session_state.user = None

if "page" not in st.session_state:
    st.session_state.page = 1
if "page_size" not in st.session_state:
    st.session_state.page_size = 20

if "resume_source" not in st.session_state:
    st.session_state.resume_source = "None"
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

if "details_cache" not in st.session_state:
    st.session_state.details_cache = {}  # vacancy_id -> full_desc_text

# terms state
if "terms_text" not in st.session_state:
    st.session_state.terms_text = ""
if "resume_hash_for_terms" not in st.session_state:
    st.session_state.resume_hash_for_terms = ""

# manual fetch results state
if "last_results_df" not in st.session_state:
    st.session_state.last_results_df = None
if "last_results_meta" not in st.session_state:
    st.session_state.last_results_meta = {}

# bootstrap default vacancies once per session
if "did_bootstrap_default" not in st.session_state:
    st.session_state.did_bootstrap_default = False

# ---------- persistent login (URL token) ----------
# If user refreshes the page, Streamlit starts a new session.
# We restore user from token in query params.
token = st.query_params.get("token", "")
if st.session_state.user is None and token:
    u = get_user_by_token(token)
    if u:
        st.session_state.user = u

# ---------- helpers ----------
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
                COL_DESC: "",  # lazy
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

@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)  # 24h
def _areas_cached():
    tree = fetch_areas_tree()
    regions, cities_by_region_id = list_regions_and_cities(tree, country_name="Россия")
    return regions, cities_by_region_id

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _fetch_default_startup(area_id: int) -> List[dict]:
    return fetch_vacancies(
        text=DEFAULT_QUERY,
        area=area_id,
        max_items=DEFAULT_STARTUP_LIMIT,
        per_page=50,
        period_days=None
    )

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _fetch_term(area_id: int, term: str, per_term: int) -> List[dict]:
    return fetch_vacancies(
        text=term,
        area=area_id,
        max_items=per_term,
        per_page=per_term,
        period_days=14
    )

@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _fetch_details(vacancy_id: str) -> str:
    full = vacancy_details(vacancy_id)
    return _strip_html(full.get("description") or "")

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

def _build_embeddings_for_df(df: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    dim = model.get_sentence_embedding_dimension()
    embs = np.zeros((len(df), dim), dtype=np.float32)

    missing_texts = []
    missing_idx = []
    for i, row in df.iterrows():
        vid = str(row[COL_JOB_ID])
        cached = get_embedding(vid, MODEL_NAME)
        if cached is not None:
            embs[i] = cached
        else:
            missing_idx.append(i)
            missing_texts.append(str(row["job_text"] or ""))

    if missing_texts:
        new_emb = model.encode(
            missing_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
        )
        new_emb = np.asarray(new_emb, dtype=np.float32)
        for j, i in enumerate(missing_idx):
            embs[i] = new_emb[j]
            put_embedding(str(df.loc[i, COL_JOB_ID]), MODEL_NAME, new_emb[j])

    denom = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    return embs / denom

def _rank_with_faiss(embs: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    import faiss
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs.astype(np.float32))
    scores, idx = index.search(query_vec.astype(np.float32), embs.shape[0])
    score_arr = np.zeros((embs.shape[0],), dtype=np.float32)
    score_arr[idx[0]] = scores[0]
    return score_arr

def _snippet(s: str, n: int = 230) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"

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

# ---------- auth UI (centered) ----------
def auth_screen():
    st.markdown("<div class='center-wrap'>", unsafe_allow_html=True)
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
                # Create persistent session token and store in URL
                tok = create_session(int(user["id"]), days_valid=30)
                st.query_params["token"] = tok
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

    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.user is None:
    auth_screen()
    st.stop()

user_id = int(st.session_state.user["id"])

# ---------- sidebar ----------
st.sidebar.title("⚙️ Настройки")

# Logout: delete session token and clear query param
if st.sidebar.button("🚪 Выйти", use_container_width=True):
    tok = st.query_params.get("token", "")
    if tok:
        delete_session(tok)
    st.query_params.clear()
    st.session_state.user = None
    st.session_state.last_results_df = None
    st.session_state.did_bootstrap_default = False
    st.rerun()

st.sidebar.subheader("Локация (HH Areas)")
regions, cities_by_region_id = _areas_cached()

region_names = [r["name"] for r in regions]
region_name = st.sidebar.selectbox("Регион", region_names, index=0 if region_names else 0)
region_obj = next((r for r in regions if r["name"] == region_name), None)
region_id = region_obj["id"] if region_obj else None

cities = cities_by_region_id.get(str(region_id), []) if region_id else []
city_names = [c["name"] for c in cities]
default_city_idx = 0
if "Новосибирск" in city_names:
    default_city_idx = city_names.index("Новосибирск")
city_name = st.sidebar.selectbox("Город", city_names, index=default_city_idx if city_names else 0)
city_obj = next((c for c in cities if c["name"] == city_name), None)
area_id = int(city_obj["id"]) if city_obj else 1

st.sidebar.subheader("Резюме")
resume_source = st.sidebar.radio("Источник резюме", ["None", "PDF resume", "Created resume"], index=0)
st.session_state.resume_source = resume_source

if resume_source == "PDF resume":
    pdf = st.sidebar.file_uploader("Загрузите PDF", type=["pdf"])
    if pdf is not None:
        with st.spinner("Читаем PDF..."):
            st.session_state.pdf_text = extract_text_from_pdf(pdf.read())

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

st.sidebar.subheader("Термины (TF-IDF)")
# auto-generate TF-IDF terms when resume changes (no HH calls)
if has_resume:
    rh = hashlib.sha256(resume_text.encode("utf-8", errors="ignore")).hexdigest()
    if rh != st.session_state.resume_hash_for_terms:
        auto_terms = extract_terms(resume_text, top_k=TERMS_MAX)
        auto_terms = auto_terms[:TERMS_MAX]
        if len(auto_terms) < TERMS_MIN:
            auto_terms = list(dict.fromkeys(auto_terms + ["python", "sql"]))[:TERMS_MIN]
        st.session_state.terms_text = "\n".join(auto_terms)
        st.session_state.resume_hash_for_terms = rh

    st.sidebar.caption("Термины выбраны автоматически. Отредактируйте и нажмите **Поиск**.")
    st.session_state.terms_text = st.sidebar.text_area(
        "Термины (по одному в строке)",
        value=st.session_state.terms_text,
        height=160,
    )

    add_term = st.sidebar.text_input("Добавить термин", value="")
    if st.sidebar.button("➕ Добавить", use_container_width=True):
        t = add_term.strip()
        if t:
            current = [x.strip() for x in st.session_state.terms_text.splitlines() if x.strip()]
            current.append(t)
            seen = set()
            deduped = []
            for x in current:
                k = x.lower()
                if k in seen:
                    continue
                seen.add(k)
                deduped.append(x)
            st.session_state.terms_text = "\n".join(deduped[:TERMS_MAX])
            st.rerun()
else:
    st.sidebar.info("Загрузите/выберите резюме — термины TF-IDF появятся автоматически.")

st.sidebar.subheader("Показ")
page_size = st.sidebar.selectbox("На странице", [10, 20, 50, 100], index=1)
st.session_state.page_size = int(page_size)

# ✅ Manual trigger for HH fetch (after bootstrap)
do_search = st.sidebar.button("Поиск", use_container_width=True)

# ---------- header ----------
st.title("💼 HH.ru Job Recommender")
st.caption("По умолчанию показываем 500 вакансий. Все последующие запросы к HH — только по кнопке **Поиск**.")

favorites = set(list_favorites(user_id))

# ---------- ACTION: manual search ----------
if do_search:
    # clear lazy details on new search
    st.session_state.details_cache = {}

    if not has_resume:
        with st.spinner("Загружаем дефолтные вакансии (без эмбеддингов)..."):
            items = _fetch_default_startup(int(area_id))
            df = _items_to_df(items)
        df["similarity_score"] = pd.NA
        st.session_state.last_results_df = df
        st.session_state.last_results_meta = {"mode": "default_manual", "area_id": int(area_id)}
    else:
        terms = [t.strip() for t in st.session_state.terms_text.splitlines() if t.strip()]
        terms = terms[:TERMS_MAX]
        if len(terms) < TERMS_MIN:
            terms = list(dict.fromkeys(terms + ["python", "sql"]))[:TERMS_MIN]

        st.write("**Термины для HH:**", ", ".join(terms))

        with st.spinner("Fetching вакансий по терминам (multi-query)..."):
            batches = [_fetch_term(int(area_id), term, PER_TERM) for term in terms]
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

        top_n = min(10, len(df))
        with st.spinner("Подгружаем полные описания только для TOP-10..."):
            for i in range(top_n):
                vid = str(df.loc[i, COL_JOB_ID])
                if vid and vid not in st.session_state.details_cache:
                    try:
                        st.session_state.details_cache[vid] = _fetch_details(vid)
                    except Exception:
                        st.session_state.details_cache[vid] = ""

        st.session_state.last_results_df = df
        st.session_state.last_results_meta = {"mode": "ranked_manual", "area_id": int(area_id), "terms": terms}

    st.session_state.page = 1

# ---------- BOOTSTRAP: show default vacancies on startup automatically ----------
if st.session_state.last_results_df is None:
    if not st.session_state.did_bootstrap_default:
        with st.spinner("Загружаем дефолтные вакансии (без эмбеддингов)..."):
            items = _fetch_default_startup(int(area_id))
            df0 = _items_to_df(items)
        df0["similarity_score"] = pd.NA
        st.session_state.last_results_df = df0
        st.session_state.last_results_meta = {"mode": "default_bootstrap", "area_id": int(area_id)}
        st.session_state.did_bootstrap_default = True
    else:
        st.info("Изменили параметры? Нажмите **Поиск**, чтобы обновить вакансии.")
        st.stop()

df = st.session_state.last_results_df.copy()

# ---------- render ----------
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
        elif vid and st.session_state.last_results_meta.get("mode", "").startswith("ranked") and idx > 10:
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

total = len(df)
st.caption(f"Вакансий: {total}")

total_pages = max(1, math.ceil(total / st.session_state.page_size))
st.session_state.page = max(1, min(st.session_state.page, total_pages))

start = (st.session_state.page - 1) * st.session_state.page_size
end = start + st.session_state.page_size
page_df = df.iloc[start:end]

for i, row in enumerate(page_df.to_dict(orient="records"), start=1):
    render_job(row, start + i)

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
