import re
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer

RU_STOP = set("""
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два об другой хоть после над больше тот через эти нас про всего них какая много разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между
""".split())

EN_STOP = set("""
the a an and or but if then else for to of in on at by with from into up down over under
is are was were be been being this that these those it its as not no yes
i you he she we they my your our their me him her us them
""".split())


def _clean(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[\w\.-]+@[\w\.-]+", " ", text)
    text = re.sub(r"\+?\d[\d\s\-\(\)]{7,}\d", " ", text)
    text = re.sub(r"[^\w\s\+\#]", " ", text)  # keep + and # (c++, c#)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_docs(text: str) -> List[str]:
    parts = re.split(r"[\n•\-\*]+", text)
    parts = [p.strip() for p in parts if len(p.strip()) > 20]
    return parts if parts else [text]


def extract_terms(resume_text: str, top_k: int = 10) -> List[str]:
    clean = _clean(resume_text)
    docs = _split_docs(clean)

    vec = TfidfVectorizer(
        token_pattern=r"(?u)\b[\w\+\#]{3,}\b",
        ngram_range=(1, 2),
        max_features=6000,
    )
    X = vec.fit_transform(docs)
    terms = vec.get_feature_names_out()
    scores = X.sum(axis=0).A1

    ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
    stop = RU_STOP | EN_STOP

    out: List[str] = []
    for t, _ in ranked:
        if t in stop:
            continue
        if len(t) < 3:
            continue
        out.append(t)
        if len(out) >= top_k:
            break
    return out
