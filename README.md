# HH.ru Job Recommender (Streamlit)

This version of the project is **HH.ru-only**: vacancies are fetched live from **api.hh.ru** and cached locally.

## Features
- Login / registration (SQLite)
- Save multiple resumes (SQLite)
- Upload PDF resume (text extracted via PyMuPDF)
- Fetch vacancies from HH.ru (default: **Russia / Novosibirsk**, area id = 4)
- Semantic ranking (Sentence-Transformers, multilingual model)
- Favorites

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- HH.ru search has a hard paging depth limit (about **2000** items per query). See HH API discussions for details. 
- Vacancies + embeddings are cached under `artifacts/hh_cache/` and also cached in Streamlit (TTL configurable in the sidebar).


## Performance upgrades
- Startup shows 500 default vacancies without embeddings.
- After selecting/uploading a resume: TF-IDF terms → multi-query fetch → FAISS ranking.
- Full descriptions are loaded only for TOP-10, and then lazily on click.
- Embeddings are reused per vacancy_id via SQLite cache (`artifacts/embeddings.sqlite3`).
