import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


class JobRecommendationSystem:
    """Semantic matcher for jobs vs resume.

    - Primary scoring: SentenceTransformer embeddings (cosine similarity).
    - Optional explanations: per-pair TF-IDF keyword overlap.

    Can run in two modes:
    1) CSV + artifacts (static dataset)
    2) DataFrame + in-memory (dynamic jobs from HH API)
    """

    def __init__(
        self,
        jobs_source: Union[str, pd.DataFrame],
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        batch_size: int = 64,
        enable_explanations: bool = True,
        stop_words: str = "english",
        use_artifacts: bool = True,
        precomputed_embeddings: Optional[np.ndarray] = None,
    ):
        self.jobs_source = jobs_source
        self.model_name = model_name
        self.batch_size = batch_size
        self.enable_explanations = enable_explanations
        self.stop_words = stop_words
        self.use_artifacts = bool(use_artifacts and isinstance(jobs_source, str))

        if self.use_artifacts:
            _ensure_dir(ARTIFACT_DIR)
            self.jobs_path = os.path.join(ARTIFACT_DIR, "jobs_clean.parquet")
            self.emb_path = os.path.join(ARTIFACT_DIR, "job_embeddings.npy")
        else:
            self.jobs_path = ""
            self.emb_path = ""

        self.model = SentenceTransformer(self.model_name)

        self.jobs_df = self._load_or_prepare_jobs()
        self.embeddings = self._load_or_build_embeddings(precomputed_embeddings)

        self._tfidf_vectorizer: Optional[TfidfVectorizer] = None

    def _build_job_text(self, df: pd.DataFrame) -> pd.Series:
        cols = [
            "workplace",
            "working_mode",
            "position",
            "job_role_and_duties",
            "requisite_skill",
            "offer_details",
            "salary",
        ]
        present = [c for c in cols if c in df.columns]
        text = df[present].fillna("").astype(str).agg(" ".join, axis=1)
        return text.str.replace(r"\s+", " ", regex=True).str.strip()

    def _load_or_prepare_jobs(self) -> pd.DataFrame:
        if isinstance(self.jobs_source, pd.DataFrame):
            df = self.jobs_source.copy()
            if "job_text" not in df.columns:
                df["job_text"] = self._build_job_text(df)
            return df

        jobs_csv = self.jobs_source
        if self.use_artifacts and os.path.exists(self.jobs_path):
            df = pd.read_parquet(self.jobs_path)
            if "job_text" not in df.columns:
                df["job_text"] = self._build_job_text(df)
                df.to_parquet(self.jobs_path, index=False)
            return df

        df = pd.read_csv(jobs_csv)
        df["job_text"] = self._build_job_text(df)
        if self.use_artifacts:
            df.to_parquet(self.jobs_path, index=False)
        return df

    def _load_or_build_embeddings(self, precomputed: Optional[np.ndarray]) -> np.ndarray:
        if precomputed is not None:
            emb = np.asarray(precomputed, dtype="float32")
            return _normalize_rows(emb)

        if self.use_artifacts and self.emb_path and os.path.exists(self.emb_path):
            emb = np.load(self.emb_path).astype("float32")
            return _normalize_rows(emb)

        texts = self.jobs_df["job_text"].fillna("").astype(str).tolist()
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        emb = np.asarray(emb, dtype="float32")

        if self.use_artifacts and self.emb_path:
            np.save(self.emb_path, emb)
        return emb

    def score_all_jobs(self, resume_text: str) -> pd.DataFrame:
        """Return jobs_df copy with similarity_score (0..1-ish)."""
        resume_text = (resume_text or "").strip()
        df = self.jobs_df.copy()
        if not resume_text:
            df["similarity_score"] = np.nan
            return df

        q = self.model.encode([resume_text], normalize_embeddings=True)
        q = np.asarray(q, dtype="float32")
        scores = (self.embeddings @ q.T).reshape(-1)
        df["similarity_score"] = scores
        return df

    def explain_match(self, resume_text: str, job_text: str, top_k: int = 10) -> Dict[str, List[str]]:
        resume_text = (resume_text or "").strip()
        job_text = (job_text or "").strip()
        if not resume_text or not job_text:
            return {"resume_keywords": [], "job_keywords": [], "matched_keywords": []}

        vec = TfidfVectorizer(stop_words=self.stop_words, ngram_range=(1, 2), max_features=3000)
        X = vec.fit_transform([resume_text, job_text])
        terms = np.array(vec.get_feature_names_out())

        def top_terms_sparse(row_idx: int) -> List[str]:
            row = X[row_idx]
            if row.nnz == 0:
                return []
            data = row.data
            idx = row.indices
            order = np.argsort(data)[-top_k:][::-1]
            return [terms[idx[i]] for i in order]

        rk = top_terms_sparse(0)
        jk = top_terms_sparse(1)
        matched = [t for t in rk if t in set(jk)]
        return {"resume_keywords": rk, "job_keywords": jk, "matched_keywords": matched[:top_k]}
