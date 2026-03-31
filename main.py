from __future__ import annotations

import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


import re
from typing import Optional

EMOJI_ONLY_RE = re.compile(
    r"^[\s"
    r"\U0001F300-\U0001FAFF"  # emoji blocks
    r"\U00002600-\U000027BF"  # misc symbols
    r"\U0001F1E6-\U0001F1FF"  # flags
    r"]+$",
    flags=re.UNICODE,
)



# =========================
# 1. НАСТРОЙКИ
# =========================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

QUESTIONS_FILE = os.path.join(DATA_DIR, "user_questions.xlsx")
ABUSIVE_WORDS_FILE = os.path.join(DATA_DIR, "ru_abusive_words.txt")
CURSE_WORDS_FILE = os.path.join(DATA_DIR, "ru_curse_words.txt")
FAQ_FILE = os.path.join(DATA_DIR, "Database.xlsx")
GENERAL_FILE = os.path.join(DATA_DIR, "Database-2.xlsx")
PROGRAM_FILE = os.path.join(DATA_DIR, "all_program.xlsx")

INTENT_MODEL_NAME = "cointegrated/rubert-tiny2"
TOXICITY_MODEL_NAME = "fasherr/toxicity_rubert"

# При желании можно отключить модель токсичности через ENV,
# если на первом запуске хотите уменьшить количество скачиваний.
ENABLE_TOXICITY_MODEL = os.getenv("ENABLE_TOXICITY_MODEL", "1") == "1"
TOXICITY_THRESHOLD = float(os.getenv("TOXICITY_THRESHOLD", "0.75"))


# =========================
# 2. НОРМАЛИЗАЦИЯ
# =========================
RUSSIAN_LIGHT_SUFFIXES = [
    "иями", "ями", "ами", "иях", "ях", "ого", "ему", "ому", "ими", "ыми",
    "ее", "ие", "ые", "ое", "ей", "ий", "ый", "ой", "ем", "им", "ым", "ом",
    "его", "их", "ых", "ую", "юю", "ая", "яя", "ою", "ею", "ах", "ях", "ам",
    "ям", "ов", "ев", "ия", "ья", "ию", "ью", "ие", "ье", "ии", "ьи",
    "а", "я", "ы", "и", "е", "у", "ю", "о"
]


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip().replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_match_text(text: str) -> str:
    text = normalize_text(str(text))
    text = re.sub(r"[«»\"“”]", " ", text)
    text = re.sub(r"[-–—/]", " ", text)
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def light_stem(word: str) -> str:
    word = normalize_match_text(word)
    if len(word) <= 3:
        return word
    for suffix in sorted(RUSSIAN_LIGHT_SUFFIXES, key=len, reverse=True):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[а-яa-z0-9]+", normalize_match_text(text))


def stem_tokenize(text: str) -> List[str]:
    return [light_stem(tok) for tok in simple_tokenize(text) if tok]


def normalize_for_match(text: str) -> str:
    text = str(text).lower().replace("ё", "е")
    text = re.sub(r"[-–—/]", " ", text)
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================
# 3. МОДЕРАЦИЯ
# =========================
TOXIC_STEM_HINTS = {
    "бляд", "бля", "пизд", "ху", "еб", "сук", "жоп", "хрен", "нахрен",
    "долбан", "черт", "чертвозьм", "мудак", "гандон"
}

TOXIC_REGEX_PATTERNS = [
    r"\bнахрен\b",
    r"\bкакого\s+хрена\b",
    r"\bчерт\s+возьми\b",
    r"\bоху[еияй]\w*",
    r"\bдолбан\w*",
]


def load_words_from_txt(file_path: str) -> set[str]:
    words: set[str] = set()
    if not os.path.exists(file_path):
        return words
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            word = normalize_text(line.strip())
            if word:
                words.add(word)
    return words


def check_empty_or_too_short(text: str) -> bool:
    return len(normalize_text(text)) == 0


def check_too_long(text: str, max_len: int = 2000) -> bool:
    return len(text) > max_len


def contains_bad_words(text: str, bad_words: set[str]) -> tuple[bool, List[str]]:
    normalized = normalize_text(text)
    normalized_match = normalize_match_text(text)
    tokens = simple_tokenize(normalized_match)
    stem_tokens_q = stem_tokenize(normalized_match)

    found_words = set()

    for token in tokens:
        if token in bad_words:
            found_words.add(token)

    for bad_word in bad_words:
        bad_word = normalize_text(bad_word)
        if len(bad_word) >= 4 and bad_word in normalized:
            found_words.add(bad_word)

    bad_stems = {light_stem(w) for w in bad_words if light_stem(w)}
    for tok in stem_tokens_q:
        if tok in bad_stems or tok in TOXIC_STEM_HINTS:
            found_words.add(tok)

    for pattern in TOXIC_REGEX_PATTERNS:
        if re.search(pattern, normalized_match):
            found_words.add(pattern)

    found = sorted(found_words)
    return len(found) > 0, found


def moderate_question(question: str, bad_words: set[str]) -> dict[str, Any]:
    if check_empty_or_too_short(question):
        return {"status": "empty", "allow_to_continue": False, "comment": "Пустой вопрос", "found_bad_words": []}
    if check_too_long(question):
        return {"status": "too_long", "allow_to_continue": False, "comment": "Слишком длинный вопрос", "found_bad_words": []}
    is_toxic, found_words = contains_bad_words(question, bad_words)
    if is_toxic:
        return {
            "status": "toxic",
            "allow_to_continue": False,
            "comment": "Обнаружены нежелательные слова",
            "found_bad_words": found_words,
        }
    return {"status": "ok", "allow_to_continue": True, "comment": "Вопрос прошёл модерацию", "found_bad_words": []}


# =========================
# 4. ПРИМЕРЫ И ПРАВИЛА
# =========================
DOMAIN_BASE_EXAMPLES = [
    "Какие ЕГЭ нужны для поступления на программу?",
    "Сколько стоит обучение на программе?",
    "Какой проходной балл на программу?",
    "Сколько бюджетных мест на программе?",
    "Сколько платных мест на программе?",
    "Что такое мегакластер?",
    "Чем отличаются мегакластеры?",
    "Какие есть образовательные программы?",
    "В каком институте реализуется программа?",
    "Какая длительность обучения на программе?",
    "Какие треки есть на программе?",
    "Чему учат на программе?",
    "Какие навыки получает выпускник?",
    "Какие секции есть в Академии?",
    "Есть ли у студентов лаборатории?",
    "Где находится кампус Академии?",
    "Что такое ЯДРО в образовательных программах?",
    "Есть ли военный учебный центр?",
    "Сколько лет действует олимпиада для БВИ?",
    "Академия и филиалы это один вуз?",
    "Можно ли изучать второй иностранный язык?",
]

HARD_NEGATIVE_EXAMPLES = [
    "Какая сегодня погода?",
    "Напиши код на Python",
    "Что посмотреть вечером?",
    "Как приготовить пасту?",
    "Что такое институт брака?",
    "Сколько стоит айфон?",
    "Кто выиграл Лигу чемпионов?",
]

DOMAIN_STRONG_KEYWORDS = [
    "поступ", "абитури", "егэ", "экзамен", "проходн", "балл", "бюдж", "платн",
    "контракт", "мест", "стоимост", "мегакластер", "направлен", "прием",
    "приемн", "вступит", "олимпиад", "бви", "филиал", "кампус", "лаборатор",
    "секци", "военн", "ядро", "общежити", "стипенд", "иностранн", "академ"
]

DOMAIN_WEAK_KEYWORDS = [
    "обучен", "учеб", "программ", "институт", "трек", "навык",
    "выпускник", "специальност", "факультет", "партнер", "практик", "карьер"
]

EXACT_FIELD_RULES = [
    {"field_name": "eges_budget", "patterns": ["какие егэ нужны на бюджет", "егэ на бюджет", "экзамены на бюджет"]},
    {"field_name": "eges_contract", "patterns": ["какие егэ нужны на платное", "егэ на платное", "егэ на контракт", "экзамены на платное"]},
    {"field_name": "budget_2025", "patterns": ["бюджетные места", "места на бюджете", "есть ли бюджет", "есть бюджет", "бюджет"]},
    {"field_name": "contract_2025", "patterns": ["платные места", "контрактные места", "места на платном обучении", "мест на платное"]},
    {"field_name": "cost", "patterns": ["сколько стоит", "стоимость обучения", "стоимость", "цена программы", "цена обучения"]},
    {"field_name": "pass_2024", "patterns": ["проходной балл", "какой проходной балл", "баллы для поступления", "какой балл"]},
    {"field_name": "edu_years", "patterns": ["сколько лет учиться", "длительность обучения", "срок обучения"]},
    {"field_name": "edu_form", "patterns": ["форма обучения", "очная", "заочная", "дистанционная"]},
    {"field_name": "megacluster", "patterns": ["какой мегакластер", "к какому мегакластеру относится", "мегакластер программы"]},
    {"field_name": "tracks", "patterns": ["какие есть треки", "профили программы", "специализации программы"]},
    {"field_name": "institute", "patterns": ["какой институт", "какая школа", "где реализуется программа"]},
    {"field_name": "major", "patterns": ["направление подготовки", "код направления"]},
    {"field_name": "desc", "patterns": ["о чем программа", "описание программы", "что изучают на программе"]},
    {"field_name": "skills", "patterns": ["чему научусь", "какие навыки получу", "компетенции выпускника"]},
]

UNSUPPORTED_PROGRAM_FIELD_RULES = {
    "salary": ["зарплат", "средняя зарплата"],
    "competition": ["конкурс на место", "конкурс"],
    "deadline": ["до какого числа", "срок подачи", "когда подать", "дедлайн"],
    "dorm": ["общежити", "проживан"],
    "avg_ege": ["средний балл егэ", "средний балл"],
    "employment": ["трудоустрой", "процент трудоустройства"],
    "stipend": ["стипенд"],
    "ranking": ["рейтинг"],
}

FIELD_LABELS = {
    "cost": "стоимость обучения",
    "pass_2024": "проходной балл",
    "budget_2025": "бюджетные места",
    "contract_2025": "платные места",
    "eges_budget": "ЕГЭ для бюджета",
    "eges_contract": "ЕГЭ для платного обучения",
    "edu_form": "форма обучения",
    "edu_years": "длительность обучения",
    "megacluster": "мегакластер",
    "tracks": "треки",
    "institute": "институт",
    "major": "направление подготовки",
    "desc": "описание программы",
    "skills": "навыки выпускника",
}

FIELD_GROUPS = {
    "cost": ["cost"],
    "pass": ["pass_2024"],
    "places": ["budget_2025", "contract_2025"],
    "eges": ["eges_budget", "eges_contract"],
    "edu_form": ["edu_form"],
    "edu_years": ["edu_years"],
    "tracks": ["tracks"],
    "institute": ["institute"],
    "megacluster": ["megacluster"],
    "major": ["major"],
    "desc": ["desc"],
    "skills": ["skills"],
}


@dataclass
class AssistantResources:
    bad_words: set[str]
    toxicity_classifier: Any
    intent_model: SentenceTransformer
    faq_df: pd.DataFrame
    general_df: pd.DataFrame
    program_df: pd.DataFrame
    faq_vectorizer: TfidfVectorizer
    faq_tfidf: Any
    general_vectorizer: TfidfVectorizer
    general_tfidf: Any
    faq_question_embeddings: Any
    general_text_embeddings: Any
    program_embeddings: Any
    domain_example_embeddings: Any
    hard_negative_embeddings: Any
    intent_example_embeddings: Any
    intent_label_names: List[str]
    intent_label_texts: List[str]
    entity_catalog: Dict[str, List[str]]
    unique_programs: List[str]
    program_match_texts: List[str]


# =========================
# 5. ЗАГРУЗКА РЕСУРСОВ
# =========================
def _clean_entity_value(text: str) -> str:
    text = normalize_text(str(text))
    text = re.sub(r"[«»\"“”]", " ", text)
    text = re.sub(r"[-–—/]", " ", text)
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_multivalue_text(text: str) -> List[str]:
    parts = re.split(r"[\n;,]+", str(text))
    parts = [_clean_entity_value(p) for p in parts]
    return [p for p in parts if p and len(p) >= 3]


def _build_entity_catalog(df: pd.DataFrame) -> Dict[str, List[str]]:
    catalog = {"program": set(), "megacluster": set(), "institute": set(), "major": set(), "track": set()}
    for col, key, min_len in [
        ("program", "program", 4),
        ("megacluster", "megacluster", 4),
        ("institute", "institute", 3),
        ("major", "major", 5),
    ]:
        if col in df.columns:
            for value in df[col].tolist():
                val = _clean_entity_value(value)
                if len(val) >= min_len:
                    catalog[key].add(val)
    if "tracks" in df.columns:
        for value in df["tracks"].tolist():
            for part in _split_multivalue_text(value):
                if len(part) >= 4:
                    catalog["track"].add(part)
    return {k: sorted(v, key=len, reverse=True) for k, v in catalog.items()}


def _load_faq_examples(faq_df: pd.DataFrame) -> List[str]:
    if "Question" not in faq_df.columns:
        return []
    qs = faq_df["Question"].dropna().astype(str).str.strip().tolist()
    return [q for q in qs if q]


def _load_general_examples(general_df: pd.DataFrame) -> List[str]:
    if "header" not in general_df.columns:
        return []
    headers = general_df["header"].dropna().astype(str).str.strip().tolist()
    examples: List[str] = []
    for h in headers:
        if not h:
            continue
        examples.extend([h, f"Расскажи про {h}", f"Что значит {h}?"])
    return list(dict.fromkeys(examples))


def _load_table_examples(program_df: pd.DataFrame, max_programs: int = 30) -> List[str]:
    if "program" not in program_df.columns:
        return []
    programs = program_df["program"].dropna().astype(str).str.strip().tolist()
    programs = list(dict.fromkeys([p for p in programs if p]))[:max_programs]
    examples: List[str] = []
    for program in programs:
        examples.extend([
            f"Сколько стоит обучение на программе {program}?",
            f"Какие ЕГЭ нужны для поступления на программу {program}?",
            f"Сколько бюджетных мест на программе {program}?",
            f"Сколько платных мест на программе {program}?",
            f"Какой проходной балл на программе {program}?",
            f"Какая форма обучения у программы {program}?",
            f"Сколько лет длится обучение на программе {program}?",
            f"Какие треки есть на программе {program}?",
            f"Какой институт отвечает за программу {program}?",
        ])
    return examples


@lru_cache(maxsize=1)
def get_resources() -> AssistantResources:
    required = [FAQ_FILE, GENERAL_FILE, PROGRAM_FILE]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Не найдены файлы данных: " + ", ".join(missing) + ". "
            "Положите Excel/TXT файлы из ноутбука в папку data рядом с main.py."
        )

    faq_df = pd.read_excel(FAQ_FILE).fillna("")
    general_df = pd.read_excel(GENERAL_FILE).fillna("")
    program_df = pd.read_excel(PROGRAM_FILE).fillna("")

    if "Question" in faq_df.columns:
        faq_df["search_text"] = faq_df["Question"].astype(str).map(normalize_for_match)
    else:
        faq_df["Question"] = ""
        faq_df["Answer"] = ""
        faq_df["search_text"] = ""

    if "header" not in general_df.columns:
        general_df["header"] = ""
    if "text" not in general_df.columns:
        # в некоторых выгрузках колонка может называться иначе
        for alt in ["Text", "body", "content"]:
            if alt in general_df.columns:
                general_df["text"] = general_df[alt]
                break
        else:
            general_df["text"] = ""
    general_df["search_text"] = (
        general_df["header"].astype(str).map(normalize_for_match) + " " +
        general_df["text"].astype(str).map(normalize_for_match)
    ).str.strip()

    if "program" not in program_df.columns:
        raise ValueError("В all_program.xlsx должна быть колонка 'program'.")
    for col in ["program", "megacluster", "institute", "major", "tracks", "desc", "skills"]:
        if col not in program_df.columns:
            program_df[col] = ""
    program_df["program"] = program_df["program"].fillna("").astype(str).str.strip()
    program_df = program_df[program_df["program"] != ""].copy()

    bad_words = load_words_from_txt(ABUSIVE_WORDS_FILE).union(load_words_from_txt(CURSE_WORDS_FILE))

    toxicity_classifier = None
    if ENABLE_TOXICITY_MODEL:
        toxicity_classifier = pipeline(
            task="text-classification",
            model=TOXICITY_MODEL_NAME,
            tokenizer=TOXICITY_MODEL_NAME,
            truncation=True,
            max_length=512,
        )

    intent_model = SentenceTransformer(INTENT_MODEL_NAME)

    faq_examples = _load_faq_examples(faq_df)
    general_examples = _load_general_examples(general_df)
    table_examples = _load_table_examples(program_df)

    domain_examples = list(dict.fromkeys(DOMAIN_BASE_EXAMPLES + faq_examples[:250] + general_examples[:250] + table_examples[:180]))

    intent_examples_map = {
        "faq": faq_examples[:400],
        "general": general_examples[:400],
        "table": table_examples[:300],
    }
    intent_label_names: List[str] = []
    intent_label_texts: List[str] = []
    for label, examples in intent_examples_map.items():
        for text in examples:
            intent_label_names.append(label)
            intent_label_texts.append(text)
    if not intent_label_texts:
        intent_label_names = ["general"]
        intent_label_texts = ["Что такое программа обучения?"]

    domain_example_embeddings = intent_model.encode(domain_examples, convert_to_tensor=True, normalize_embeddings=True)
    hard_negative_embeddings = intent_model.encode(HARD_NEGATIVE_EXAMPLES, convert_to_tensor=True, normalize_embeddings=True)
    intent_example_embeddings = intent_model.encode(intent_label_texts, convert_to_tensor=True, normalize_embeddings=True)

    faq_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
    faq_tfidf = faq_vectorizer.fit_transform(faq_df["search_text"].tolist())
    faq_question_embeddings = intent_model.encode(faq_df["search_text"].tolist(), convert_to_tensor=True, normalize_embeddings=True)

    general_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
    general_tfidf = general_vectorizer.fit_transform(general_df["search_text"].tolist())
    general_text_embeddings = intent_model.encode(general_df["search_text"].tolist(), convert_to_tensor=True, normalize_embeddings=True)

    unique_programs = program_df["program"].drop_duplicates().tolist()
    program_match_texts = [normalize_for_match(p) for p in unique_programs]
    program_embeddings = intent_model.encode(program_match_texts, convert_to_tensor=True, normalize_embeddings=True)

    entity_catalog = _build_entity_catalog(program_df)

    return AssistantResources(
        bad_words=bad_words,
        toxicity_classifier=toxicity_classifier,
        intent_model=intent_model,
        faq_df=faq_df,
        general_df=general_df,
        program_df=program_df,
        faq_vectorizer=faq_vectorizer,
        faq_tfidf=faq_tfidf,
        general_vectorizer=general_vectorizer,
        general_tfidf=general_tfidf,
        faq_question_embeddings=faq_question_embeddings,
        general_text_embeddings=general_text_embeddings,
        program_embeddings=program_embeddings,
        domain_example_embeddings=domain_example_embeddings,
        hard_negative_embeddings=hard_negative_embeddings,
        intent_example_embeddings=intent_example_embeddings,
        intent_label_names=intent_label_names,
        intent_label_texts=intent_label_texts,
        entity_catalog=entity_catalog,
        unique_programs=unique_programs,
        program_match_texts=program_match_texts,
    )


# =========================
# 6. КЛАССИФИКАЦИЯ
# =========================
def count_keyword_hits(question: str, keywords: List[str]) -> int:
    q = normalize_text(question)
    return sum(1 for kw in keywords if kw in q)


def _entity_match_score(question: str, entity_value: str) -> float:
    q_stems = set(stem_tokenize(question))
    e_stems = set(stem_tokenize(entity_value))
    if not q_stems or not e_stems:
        return 0.0
    overlap = len(q_stems & e_stems) / len(e_stems)
    char_score = SequenceMatcher(None, normalize_match_text(question), normalize_match_text(entity_value)).ratio()
    return 0.7 * overlap + 0.3 * char_score


def find_entity_hits(question: str, catalog: Optional[Dict[str, List[str]]] = None) -> List[Dict[str, Any]]:
    resources = get_resources()
    catalog = catalog or resources.entity_catalog
    q_norm = _clean_entity_value(question)
    hits: List[Dict[str, Any]] = []
    for entity_type, values in catalog.items():
        for value in values:
            if len(value) < 3:
                continue
            if value in q_norm:
                hits.append({"entity_type": entity_type, "entity_value": value, "match_type": "exact", "score": 1.0})
                continue
            score = _entity_match_score(q_norm, value)
            if score >= 0.62:
                hits.append({"entity_type": entity_type, "entity_value": value, "match_type": "soft", "score": round(float(score), 4)})
    dedup: List[Dict[str, Any]] = []
    seen = set()
    for item in sorted(hits, key=lambda x: x["score"], reverse=True):
        key = (item["entity_type"], item["entity_value"])
        if key not in seen:
            seen.add(key)
            dedup.append(item)
    return dedup


def get_best_program_hint_score(question: str) -> float:
    resources = get_resources()
    q_norm = _clean_entity_value(question)
    best = 0.0
    for value in resources.entity_catalog.get("program", []):
        score = _entity_match_score(q_norm, value)
        if score > best:
            best = score
    return round(float(best), 4)


def detect_table_rule(question: str) -> Dict[str, Any]:
    q_norm = _clean_entity_value(question)
    for label, patterns in UNSUPPORTED_PROGRAM_FIELD_RULES.items():
        for pat in patterns:
            if pat in q_norm:
                return {"is_table": True, "is_supported": False, "field_name": None, "matched_keyword": pat, "unsupported_label": label}
    matches = []
    for rule in EXACT_FIELD_RULES:
        for pat in rule["patterns"]:
            if _clean_entity_value(pat) in q_norm:
                matches.append({"field_name": rule["field_name"], "matched_keyword": pat, "priority": len(pat)})
    if ("егэ" in q_norm or "экзамен" in q_norm or "предмет" in q_norm or "вступительн" in q_norm):
        if "платн" in q_norm or "контракт" in q_norm:
            matches.append({"field_name": "eges_contract", "matched_keyword": "eges_contract", "priority": 100})
        elif "бюдж" in q_norm:
            matches.append({"field_name": "eges_budget", "matched_keyword": "eges_budget", "priority": 100})
    if not matches:
        return {"is_table": False, "is_supported": None, "field_name": None, "matched_keyword": None, "unsupported_label": None}
    best = sorted(matches, key=lambda x: x["priority"], reverse=True)[0]
    return {"is_table": True, "is_supported": True, "field_name": best["field_name"], "matched_keyword": best["matched_keyword"], "unsupported_label": None}


def check_toxicity(question: str, toxicity_threshold: float = TOXICITY_THRESHOLD) -> Dict[str, Any]:
    resources = get_resources()
    if resources.toxicity_classifier is None:
        return {"toxicity_label": "skipped", "toxicity_score": None, "allow_after_toxicity": True}
    pred = resources.toxicity_classifier(question)[0]
    raw_label = str(pred["label"]).lower()
    raw_score = float(pred["score"])
    is_toxic = any(marker in raw_label for marker in ["toxic", "1", "label_1"])
    if is_toxic and raw_score >= toxicity_threshold:
        return {"toxicity_label": "toxic", "toxicity_score": round(raw_score, 4), "allow_after_toxicity": False}
    return {"toxicity_label": "non_toxic", "toxicity_score": round(raw_score, 4), "allow_after_toxicity": True}


def is_in_admission_domain(question: str, semantic_threshold: float = 0.50, margin_threshold: float = 0.02) -> Dict[str, Any]:
    resources = get_resources()
    q_norm = normalize_text(question)
    q_emb = resources.intent_model.encode(q_norm, convert_to_tensor=True, normalize_embeddings=True)
    domain_sims = util.cos_sim(q_emb, resources.domain_example_embeddings)[0].cpu().numpy()
    domain_best_idx = int(np.argmax(domain_sims))
    domain_score = float(np.max(domain_sims))
    negative_sims = util.cos_sim(q_emb, resources.hard_negative_embeddings)[0].cpu().numpy()
    negative_best_idx = int(np.argmax(negative_sims))
    negative_score = float(np.max(negative_sims))
    strong_hits = count_keyword_hits(q_norm, DOMAIN_STRONG_KEYWORDS)
    weak_hits = count_keyword_hits(q_norm, DOMAIN_WEAK_KEYWORDS)
    entity_hits = find_entity_hits(q_norm)
    entity_hit_count = len(entity_hits)
    table_rule = detect_table_rule(q_norm)
    program_hint_score = get_best_program_hint_score(q_norm)
    margin = domain_score - negative_score

    if entity_hit_count >= 1:
        in_domain, reason = True, "entity_hit"
    elif program_hint_score >= 0.68:
        in_domain, reason = True, "program_hint"
    elif table_rule["is_table"]:
        in_domain, reason = True, "table_like_question"
    elif strong_hits >= 1 and domain_score >= semantic_threshold and margin >= margin_threshold:
        in_domain, reason = True, "strong_keyword_plus_semantics"
    elif weak_hits >= 2 and domain_score >= 0.53 and margin >= 0.01:
        in_domain, reason = True, "weak_keywords_plus_semantics"
    elif domain_score >= 0.64 and margin >= 0.04:
        in_domain, reason = True, "high_semantic_match"
    else:
        in_domain, reason = False, "outside_domain"

    return {
        "in_domain": in_domain,
        "domain_score": round(domain_score, 4),
        "negative_score": round(negative_score, 4),
        "domain_margin": round(margin, 4),
        "domain_strong_hits": strong_hits,
        "domain_weak_hits": weak_hits,
        "entity_hit_count": entity_hit_count,
        "entity_hits": entity_hits,
        "program_hint_score": program_hint_score,
        "matched_domain_example": DOMAIN_BASE_EXAMPLES[0] if len(domain_sims) == 0 else None,
        "matched_negative_example": HARD_NEGATIVE_EXAMPLES[negative_best_idx],
        "table_rule": table_rule,
        "domain_reason": reason,
    }


def classify_question_type_model(question: str, threshold: float = 0.44) -> Dict[str, Any]:
    resources = get_resources()
    table_rule = detect_table_rule(question)
    entity_hits = find_entity_hits(question)
    program_hint_score = get_best_program_hint_score(question)
    q_norm = normalize_text(question)

    if table_rule["is_table"] and (len(entity_hits) >= 1 or program_hint_score >= 0.55 or "программа" in q_norm):
        return {"question_type": "table", "best_score": 1.0, "matched_example": f'rule_table::{table_rule.get("field_name")}::{table_rule.get("matched_keyword")}' }

    question_embedding = resources.intent_model.encode(q_norm, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(question_embedding, resources.intent_example_embeddings)[0].cpu().numpy()
    label_bucket: Dict[str, List[tuple[float, str]]] = {}
    for score, label, example in zip(scores, resources.intent_label_names, resources.intent_label_texts):
        label_bucket.setdefault(label, []).append((float(score), example))
    label_scores = []
    for label, pairs in label_bucket.items():
        pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
        top_scores = [p[0] for p in pairs[:5]]
        agg_score = 0.7 * top_scores[0] + 0.3 * float(np.mean(top_scores))
        label_scores.append({"label": label, "score": agg_score, "matched_example": pairs[0][1]})
    best = sorted(label_scores, key=lambda x: x["score"], reverse=True)[0]
    predicted_label = best["label"]
    best_score = float(best["score"])
    matched_example = best["matched_example"]
    if best_score < threshold:
        predicted_label = "general"
    if table_rule["is_table"] and predicted_label != "table" and program_hint_score >= 0.55:
        predicted_label = "table"
        matched_example = f'rule_table::{table_rule.get("field_name")}::{table_rule.get("matched_keyword")}'
        best_score = max(best_score, 0.75)
    return {"question_type": predicted_label, "best_score": round(best_score, 4), "matched_example": matched_example}


def classify_question_pipeline(question: str) -> Dict[str, Any]:
    resources = get_resources()
    moderation = moderate_question(question, resources.bad_words)
    if not moderation["allow_to_continue"]:
        return {
            "question_type": "blocked",
            "best_score": 1.0,
            "matched_example": moderation["comment"],
            "found_bad_words": moderation["found_bad_words"],
            "domain_score": None,
            "negative_score": None,
            "domain_margin": None,
            "domain_strong_hits": 0,
            "domain_weak_hits": 0,
            "entity_hit_count": 0,
            "matched_domain_example": None,
            "matched_negative_example": None,
            "domain_reason": "moderation_block",
        }

    toxicity = check_toxicity(question)
    if not toxicity["allow_after_toxicity"]:
        return {
            "question_type": "blocked",
            "best_score": toxicity["toxicity_score"] or 1.0,
            "matched_example": "toxicity_model_block",
            "found_bad_words": [],
            "domain_score": None,
            "negative_score": None,
            "domain_margin": None,
            "domain_strong_hits": 0,
            "domain_weak_hits": 0,
            "entity_hit_count": 0,
            "matched_domain_example": None,
            "matched_negative_example": None,
            "domain_reason": "toxicity_block",
        }

    domain = is_in_admission_domain(question)
    if not domain["in_domain"]:
        return {
            "question_type": "out_of_scope",
            "best_score": domain["domain_score"],
            "matched_example": domain.get("matched_negative_example"),
            "found_bad_words": [],
            **domain,
        }

    qtype = classify_question_type_model(question)
    return {**qtype, "found_bad_words": [], **domain}


# =========================
# 7. ROUTING + RETRIEVAL
# =========================
def route_question(question_type: str) -> Dict[str, Any]:
    if question_type == "table":
        return {"route_target": "program_table", "source_name": "all_program.xlsx", "ready_for_retrieval": True}
    if question_type == "faq":
        return {"route_target": "faq_base", "source_name": "Database.xlsx", "ready_for_retrieval": True}
    if question_type == "general":
        return {"route_target": "general_base", "source_name": "Database-2.xlsx", "ready_for_retrieval": True}
    if question_type == "blocked":
        return {"route_target": "blocked", "source_name": None, "ready_for_retrieval": False}
    if question_type == "out_of_scope":
        return {"route_target": "out_of_scope", "source_name": None, "ready_for_retrieval": False}
    return {"route_target": "unknown", "source_name": None, "ready_for_retrieval": False}


def lexical_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_for_match(a), normalize_for_match(b)).ratio()


def token_overlap_ratio(a: str, b: str) -> float:
    a_set = set(stem_tokenize(a))
    b_set = set(stem_tokenize(b))
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def embed_text(text: str):
    resources = get_resources()
    return resources.intent_model.encode(text, convert_to_tensor=True, normalize_embeddings=True)


def get_top_program_candidates(question: str, top_k: int = 3) -> List[Dict[str, Any]]:
    resources = get_resources()
    q_norm = normalize_for_match(question)
    q_emb = embed_text(q_norm)
    sem_sims = util.cos_sim(q_emb, resources.program_embeddings)[0].cpu().numpy()
    results = []
    for idx, program in enumerate(resources.unique_programs):
        lex = lexical_similarity(q_norm, resources.program_match_texts[idx])
        overlap = token_overlap_ratio(q_norm, resources.program_match_texts[idx])
        score = 0.60 * float(sem_sims[idx]) + 0.25 * lex + 0.15 * overlap
        results.append({"program": program, "score": round(float(score), 4)})
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def detect_program_candidate(question: str) -> Dict[str, Any]:
    top3 = get_top_program_candidates(question, top_k=3)
    if not top3:
        return {"program_candidate": None, "program_candidates_top3": [], "clarification_needed": True}
    best = top3[0]
    clarification_needed = len(top3) > 1 and (best["score"] - top3[1]["score"]) < 0.03
    return {"program_candidate": best["program"], "program_candidates_top3": top3, "clarification_needed": clarification_needed}


def detect_best_field_group(question: str) -> Dict[str, Any]:
    table_rule = detect_table_rule(question)
    if table_rule["field_name"]:
        for group_name, field_names in FIELD_GROUPS.items():
            if table_rule["field_name"] in field_names:
                return {
                    "field_group": group_name,
                    "requested_field_best": table_rule["field_name"],
                    "field_candidates_topk": [{"key": table_rule["field_name"], "score": 1.0}],
                    "unsupported_field_label": table_rule["unsupported_label"],
                }
    q = normalize_text(question)
    scored = []
    for group_name, field_names in FIELD_GROUPS.items():
        score = 0.0
        for field_name in field_names:
            label = FIELD_LABELS.get(field_name, field_name)
            score = max(score, lexical_similarity(q, label), token_overlap_ratio(q, label))
        scored.append((group_name, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    best_group, best_score = scored[0]
    requested_field = FIELD_GROUPS[best_group][0] if best_score >= 0.15 else None
    field_candidates_topk = [{"key": f, "score": round(best_score, 4)} for f in FIELD_GROUPS[best_group]]
    return {
        "field_group": best_group,
        "requested_field_best": requested_field,
        "field_candidates_topk": field_candidates_topk,
        "unsupported_field_label": table_rule["unsupported_label"],
    }


def prepare_single_query(question: str, route_target: str, ready_for_retrieval: bool) -> Dict[str, Any]:
    if not ready_for_retrieval:
        return {"query_ready": False}
    if route_target == "program_table":
        program_info = detect_program_candidate(question)
        field_info = detect_best_field_group(question)
        return {
            "query_ready": True,
            **program_info,
            **field_info,
        }
    return {"query_ready": True}


def hybrid_row_score(query: str, candidate_text: str, semantic_score: Optional[float] = None) -> float:
    lex = lexical_similarity(query, candidate_text)
    overlap = token_overlap_ratio(query, candidate_text)
    if semantic_score is None:
        return 0.55 * lex + 0.45 * overlap
    return 0.55 * float(semantic_score) + 0.25 * lex + 0.20 * overlap


def rank_program_rows(question: str, program_candidates_top3: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    resources = get_resources()
    ranked_rows = []
    if not program_candidates_top3:
        return ranked_rows
    q_norm = normalize_for_match(question)
    for cand in program_candidates_top3:
        program_name = cand.get("program")
        program_score = float(cand.get("score", 0.0))
        subset = resources.program_df[resources.program_df["program"] == program_name].copy()
        if subset.empty:
            continue
        for idx, row in subset.iterrows():
            row_text = " ".join([str(row.get(c, "")) for c in ["program", "megacluster", "institute", "major", "tracks", "desc", "skills"]])
            final_score = hybrid_row_score(q_norm, row_text, semantic_score=program_score)
            ranked_rows.append({
                "row_idx": idx,
                "program": row.get("program", ""),
                "row_score": round(float(final_score), 4),
                "program_candidate_score": round(float(program_score), 4),
            })
    dedup: Dict[int, Dict[str, Any]] = {}
    for item in ranked_rows:
        idx = item["row_idx"]
        if idx not in dedup or item["row_score"] > dedup[idx]["row_score"]:
            dedup[idx] = item
    return sorted(dedup.values(), key=lambda x: x["row_score"], reverse=True)[:top_k]


def get_candidate_field_names(row: pd.Series, requested_field_best: Optional[str], field_candidates_topk: Optional[List[Dict[str, Any]]]) -> List[str]:
    candidates = []
    if requested_field_best and requested_field_best in row.index:
        candidates.append(requested_field_best)
        return candidates
    if isinstance(field_candidates_topk, list):
        for item in field_candidates_topk:
            field_name = item.get("key")
            if field_name in row.index and field_name not in candidates:
                candidates.append(field_name)
    return candidates


def retrieve_from_program_table(question: str,
                                program_candidates_top3: List[Dict[str, Any]],
                                requested_field_best: Optional[str],
                                field_candidates_topk: Optional[List[Dict[str, Any]]],
                                field_group: Optional[str],
                                unsupported_field_label: Optional[str] = None) -> Dict[str, Any]:
    resources = get_resources()
    ranked_rows = rank_program_rows(question, program_candidates_top3, top_k=3)
    if not ranked_rows:
        return {"retrieval_status": "program_not_found", "retrieval_type": "program_table", "matched_program": None, "matched_value": None, "field_values_found": [], "unsupported_field_label": unsupported_field_label}
    best_row_info = ranked_rows[0]
    row = resources.program_df.loc[best_row_info["row_idx"]]
    if unsupported_field_label:
        return {
            "retrieval_status": "field_not_found",
            "retrieval_type": "program_table",
            "matched_program": row.get("program", ""),
            "matched_field": None,
            "matched_value": None,
            "field_group": field_group,
            "field_values_found": [],
            "retrieval_score": best_row_info["row_score"],
            "unsupported_field_label": unsupported_field_label,
        }
    candidate_fields = get_candidate_field_names(row, requested_field_best, field_candidates_topk)
    field_values_found = []
    for field_name in candidate_fields:
        value = row.get(field_name, "")
        if str(value).strip() != "":
            field_values_found.append({"field_name": field_name, "field_label": FIELD_LABELS.get(field_name, field_name), "value": value})
    matched_field = field_values_found[0]["field_name"] if field_values_found else None
    matched_value = field_values_found[0]["value"] if field_values_found else None
    retrieval_status = "found" if matched_value is not None else "field_not_found"
    return {
        "retrieval_status": retrieval_status,
        "retrieval_type": "program_table",
        "matched_program": row.get("program", ""),
        "matched_field": matched_field,
        "matched_value": matched_value,
        "field_group": field_group,
        "field_values_found": field_values_found,
        "retrieval_score": best_row_info["row_score"],
        "unsupported_field_label": unsupported_field_label,
    }


def retrieve_from_faq(question: str, top_k: int = 3) -> Dict[str, Any]:
    resources = get_resources()
    q_norm = normalize_for_match(question)
    q_emb = embed_text(q_norm)
    sem_sims = util.cos_sim(q_emb, resources.faq_question_embeddings)[0].cpu().numpy()
    tfidf_vec = resources.faq_vectorizer.transform([q_norm])
    tfidf_sims = cosine_similarity(tfidf_vec, resources.faq_tfidf)[0]
    scores = []
    for idx, row in resources.faq_df.iterrows():
        score = 0.60 * float(sem_sims[idx]) + 0.25 * float(tfidf_sims[idx]) + 0.15 * token_overlap_ratio(q_norm, row["search_text"])
        scores.append(score)
    top_idx = np.argsort(scores)[::-1][:top_k]
    faq_hits = [{
        "faq_question": resources.faq_df.iloc[idx]["Question"],
        "faq_answer": resources.faq_df.iloc[idx].get("Answer", ""),
        "score": round(float(scores[idx]), 4),
    } for idx in top_idx]
    if not faq_hits or float(faq_hits[0]["score"]) < 0.33:
        return {"retrieval_status": "not_found", "retrieval_type": "faq_base", "matched_question": None, "matched_answer": None, "retrieval_score": faq_hits[0]["score"] if faq_hits else None, "faq_hits_topk": faq_hits}
    best = faq_hits[0]
    return {"retrieval_status": "found", "retrieval_type": "faq_base", "matched_question": best["faq_question"], "matched_answer": best["faq_answer"], "retrieval_score": best["score"], "faq_hits_topk": faq_hits}


def retrieve_from_general_base(question: str, top_k: int = 3) -> Dict[str, Any]:
    resources = get_resources()
    q_norm = normalize_for_match(question)
    q_emb = embed_text(q_norm)
    sem_sims = util.cos_sim(q_emb, resources.general_text_embeddings)[0].cpu().numpy()
    tfidf_vec = resources.general_vectorizer.transform([q_norm])
    tfidf_sims = cosine_similarity(tfidf_vec, resources.general_tfidf)[0]
    scores = []
    for idx, row in resources.general_df.iterrows():
        score = 0.60 * float(sem_sims[idx]) + 0.25 * float(tfidf_sims[idx]) + 0.15 * token_overlap_ratio(q_norm, row["search_text"])
        scores.append(score)
    top_idx = np.argsort(scores)[::-1][:top_k]
    hits = [{
        "header": resources.general_df.iloc[idx]["header"],
        "text": resources.general_df.iloc[idx]["text"],
        "score": round(float(scores[idx]), 4),
    } for idx in top_idx]
    if not hits or float(hits[0]["score"]) < 0.33:
        return {"retrieval_status": "not_found", "retrieval_type": "general_base", "matched_header": None, "matched_text": None, "retrieval_score": hits[0]["score"] if hits else None, "general_hits_topk": hits}
    best = hits[0]
    return {"retrieval_status": "found", "retrieval_type": "general_base", "matched_header": best["header"], "matched_text": best["text"], "retrieval_score": best["score"], "general_hits_topk": hits}


def retrieve_single_record(state: Dict[str, Any]) -> Dict[str, Any]:
    route_target = state["route_target"]
    question = state["question"]
    if route_target == "blocked":
        return {"retrieval_status": "blocked", "retrieval_type": "blocked"}
    if route_target == "out_of_scope":
        return {"retrieval_status": "out_of_scope", "retrieval_type": "out_of_scope"}
    if route_target == "program_table":
        return retrieve_from_program_table(
            question=question,
            program_candidates_top3=state.get("program_candidates_top3") or [],
            requested_field_best=state.get("requested_field_best"),
            field_candidates_topk=state.get("field_candidates_topk") or [],
            field_group=state.get("field_group"),
            unsupported_field_label=state.get("unsupported_field_label"),
        )
    if route_target == "faq_base":
        return retrieve_from_faq(question, top_k=3)
    if route_target == "general_base":
        return retrieve_from_general_base(question, top_k=3)
    return {"retrieval_status": "unsupported_route", "retrieval_type": None}


# =========================
# 8. ОТВЕТЫ
# =========================
def get_out_of_scope_response() -> str:
    return (
        "Я отвечаю только на вопросы о поступлении: программах, стоимости обучения, "
        "проходных баллах, ЕГЭ, количестве мест и мегакластерах."
    )


def get_blocked_response() -> str:
    return "Пожалуйста, сформулируйте вопрос корректно и без грубых выражений."


def get_clarification_program_response(state: Dict[str, Any]) -> str:
    candidates = state.get("program_candidates_top3") or []
    names = [c.get("program", "") for c in candidates[:3] if c.get("program")]
    if names:
        return "Я не смог однозначно определить программу. Возможно, вы имели в виду: " + "; ".join(names) + "."
    return "Уточните, пожалуйста, название программы."


def fallback_not_found_answer(state: Dict[str, Any]) -> str:
    route_target = state.get("route_target")
    if route_target == "program_table":
        return "Я не нашёл точной информации по этому запросу в таблице программ. Уточните название программы или параметр вопроса."
    if route_target == "faq_base":
        return "Я не нашёл подходящего готового ответа на этот вопрос в FAQ."
    if route_target == "general_base":
        return "Я не нашёл полного ответа на этот вопрос в текстовой базе знаний."
    return "Я не нашёл достаточной информации в базе, чтобы надёжно ответить на этот вопрос."


def fallback_program_table_answer(state: Dict[str, Any]) -> str:
    matched_program = state.get("matched_program")
    field_values_found = state.get("field_values_found") or []
    context_status = state.get("retrieval_status")
    unsupported_field_label = state.get("unsupported_field_label")
    matched_field = state.get("matched_field")
    matched_value = state.get("matched_value")

    if state.get("clarification_needed") and context_status in {"program_not_found", "field_not_found"}:
        return get_clarification_program_response(state)

    if unsupported_field_label:
        if matched_program:
            return f"Я нашёл программу «{matched_program}», но в доступных данных нет показателя по вашему запросу."
        return "В доступных данных нет показателя по вашему запросу."

    if matched_field == "budget_2025":
        try:
            value_num = float(str(matched_value).replace(",", "."))
        except Exception:
            value_num = None
        if value_num == 0:
            return f"На программе «{matched_program}» бюджетных мест нет." if matched_program else "Бюджетных мест нет."

    if not field_values_found:
        if matched_program:
            return f"Я нашёл программу «{matched_program}», но нужное значение в таблице определить не удалось."
        return "Я не смог надёжно извлечь нужное значение из таблицы."

    if len(field_values_found) == 1:
        item = field_values_found[0]
        label = item.get("field_label", item.get("field_name", ""))
        value = str(item.get("value", "")).strip()
        if matched_program:
            return f"Для программы «{matched_program}» {label.lower()} — {value}."
        return f"{label} — {value}."

    parts = [f"Для программы «{matched_program}» найдено несколько связанных значений:"] if matched_program else []
    for item in field_values_found:
        parts.append(f"{item.get('field_label', item.get('field_name', ''))}: {item.get('value', '')}")
    return " ".join(parts)


def fallback_faq_answer(state: Dict[str, Any]) -> str:
    matched_answer = state.get("matched_answer")
    return str(matched_answer).strip() if matched_answer and str(matched_answer).strip() else "Я не смог найти подходящий готовый ответ в FAQ."


def fallback_general_answer(state: Dict[str, Any]) -> str:
    matched_text = state.get("matched_text")
    return str(matched_text).strip() if matched_text and str(matched_text).strip() else "Я не смог найти подходящий фрагмент в базе знаний."


def build_fallback_answer(state: Dict[str, Any]) -> str:
    route_target = state.get("route_target")
    retrieval_status = state.get("retrieval_status")
    if route_target == "blocked" or retrieval_status == "blocked":
        return get_blocked_response()
    if route_target == "out_of_scope" or retrieval_status == "out_of_scope":
        return get_out_of_scope_response()
    if route_target == "program_table":
        return fallback_program_table_answer(state)
    if route_target == "faq_base":
        return fallback_faq_answer(state)
    if route_target == "general_base":
        return fallback_general_answer(state)
    return fallback_not_found_answer(state)


def clean_final_answer(text: str) -> str:
    if not text:
        return text
    cleaned = text.strip()
    for pattern in [
        r"^согласно предоставленной информации[,:\s-]*",
        r"^согласно имеющейся информации[,:\s-]*",
        r"^согласно представленным данным[,:\s-]*",
        r"^по имеющейся информации[,:\s-]*",
        r"^на основе предоставленной информации[,:\s-]*",
        r"^на основе представленной информации[,:\s-]*",
        r"^исходя из представленных данных[,:\s-]*",
        r"^в предоставленных материалах[,:\s-]*",
    ]:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"^[:\-–—\s]+", "", cleaned).strip()
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


# =========================
# 9. ПУБЛИЧНЫЙ ИНТЕРФЕЙС
# =========================
def combine_question_with_docx(user_text: str, docx_text: Optional[str]) -> str:
    user_text = (user_text or "").strip()
    docx_text = (docx_text or "").strip()
    if user_text and docx_text:
        return f"{user_text}\n\nКонтекст из DOCX:\n{docx_text[:4000]}"
    if user_text:
        return user_text
    return docx_text[:4000]


def is_emoji_only(text: str) -> bool:
    text = (text or "").strip()
    return bool(text) and bool(EMOJI_ONLY_RE.fullmatch(text))

def run_model(user_text: str, docx_text: Optional[str] = None) -> str:
    """
    Единая функция для API и Telegram-бота.

    На вход:
    - user_text: текст пользователя
    - docx_text: уже извлечённый текст из .docx (опционально)

    На выход:
    - строка ответа
    """
    question = combine_question_with_docx(user_text, docx_text)
    if not question.strip():
        return "Введите вопрос или приложите .docx файл."

    if is_emoji_only(question):
        return "🖕"

    classification = classify_question_pipeline(question)
    routing = route_question(classification["question_type"])
    prepared = prepare_single_query(question, routing["route_target"], routing["ready_for_retrieval"])

    state: Dict[str, Any] = {
        "question": question,
        **classification,
        **routing,
        **prepared,
    }
    retrieval = retrieve_single_record(state)
    state.update(retrieval)

    answer = build_fallback_answer(state)
    answer = clean_final_answer(answer)
    return answer


def warmup() -> None:
    """Предзагрузка моделей и таблиц. Полезно вызвать при старте API."""
    get_resources()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Локальный запуск пайплайна из main.py")
    parser.add_argument("question", nargs="?", default="Сколько стоит обучение на программе Бизнес-информатика?")
    args = parser.parse_args()

    print("Загружаю ресурсы...")
    warmup()
    print(run_model(args.question))
