import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")


def ensure_dir():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)


def build_job_text(df: pd.DataFrame) -> pd.Series:
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


def main():
    ensure_dir()
    jobs_csv = "JobsFE.csv"
    jobs_path = os.path.join(ARTIFACT_DIR, "jobs_clean.parquet")
    emb_path = os.path.join(ARTIFACT_DIR, "job_embeddings.npy")

    df = pd.read_csv(jobs_csv)
    df["job_text"] = build_job_text(df)
    df.to_parquet(jobs_path, index=False)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = df["job_text"].fillna("").astype(str).tolist()
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype="float32")
    np.save(emb_path, emb)

    print("Saved:", jobs_path)
    print("Saved:", emb_path)
    print("Done.")


if __name__ == "__main__":
    main()
