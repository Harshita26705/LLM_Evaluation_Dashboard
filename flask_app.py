"""
Flask Application for LLM Evaluation Dashboard
A professional web-based LLM evaluation platform
"""

from flask import Flask, render_template, request, jsonify
import json
import io
import math
import re
import ast
import importlib
import nltk
from difflib import SequenceMatcher
from collections import Counter
from PIL import Image
import numpy as np
import pandas as pd

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from nltk.tokenize import TreebankWordTokenizer
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception as e:
    SentenceTransformer = None
    util = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print(f"   ⚠️ SentenceTransformer import unavailable: {e}")
from detoxify import Detoxify
import base64

# Import enhanced code analyzer
try:
    from code_analyzer import get_analyzer
    CODE_ANALYZER_AVAILABLE = True
    print("   ✅ Loaded enhanced code analyzer (Ollama)")
except Exception as e:
    CODE_ANALYZER_AVAILABLE = False
    print(f"   ⚠️ Enhanced code analyzer not available: {e}")

# Try to load spaCy for enhanced NER
try:
    en_core_web_sm = importlib.import_module("en_core_web_sm")
    nlp = en_core_web_sm.load()
    print("   ✅ Loaded spacy model")
except Exception:
    print("   ⚠️ Spacy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Lazy-loaded models (avoid blocking server startup)
embedder = None
tokenizer = None
toxicity_model = None
models_loaded = False

def ensure_models_loaded():
    """Load models once on first request to avoid startup delays."""
    global embedder, tokenizer, toxicity_model, models_loaded
    if models_loaded:
        return
    print("⏳ Loading models...")
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            print("   ✅ Loaded sentence embedder")
        except Exception as e:
            print(f"   ⚠️ Could not load embedder: {e}")
            embedder = None
    else:
        print("   ⚠️ Sentence embedder disabled; using lexical fallback for semantic similarity")
        embedder = None

    try:
        tokenizer = TreebankWordTokenizer()
        print("   ✅ Loaded tokenizer")
    except Exception as e:
        print(f"   ⚠️ Could not load tokenizer: {e}")
        tokenizer = None

    try:
        toxicity_model = Detoxify("original")
        print("   ✅ Loaded toxicity model")
    except Exception as e:
        print(f"   ⚠️ Could not load toxicity model: {e}")
        toxicity_model = None

    models_loaded = True
    print("✅ Models loaded!\n")

# --- Evaluation Metrics ---
LAMBDA_LEN = 0.5
EXPECTED_LEN_RATIO = 2.0
GAMMA_TOX = 5.0
GAMMA_BIAS = 4.0
ALPHA_COH = 0.6
HALL_MULT = 0.3
SAFETY_THRESHOLD = 0.30
SAFETY_CAP = 0.50

WEIGHTS = {
    "relevance": 0.25,
    "length_fit": 0.15,
    "coherence": 0.20,
    "rouge1_f1": 0.10,
    "toxicity_inv": 0.15,
    "bias_inv": 0.10,
    "hall_inv": 0.05
}

# --- Utility Functions ---
TOKEN_EQUIVALENTS = {
    "hi": "greeting",
    "hello": "greeting",
    "hey": "greeting",
    "hiya": "greeting",
    "yo": "greeting",
    "thanks": "thank",
    "thankyou": "thank",
    "thx": "thank"
}

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "am", "be", "been", "being",
    "to", "for", "of", "in", "on", "at", "with", "by", "from", "as", "it", "this",
    "that", "these", "those", "and", "or", "but", "if", "then", "than", "so", "what",
    "whats", "s", "do", "does", "did", "you", "your", "yours", "i", "me", "my", "we",
    "our", "ours", "they", "them", "their", "theirs", "he", "she", "him", "her", "his",
    "hers", "up"
}

SMALL_TALK_PROMPTS = {
    "greeting", "hi", "hello", "hey", "whats up", "what is up", "sup", "how are you", "how r u"
}

SMALL_TALK_RESPONSES = {
    "greeting", "nothing much", "not much", "all good", "doing well", "i am good", "i'm good",
    "good", "fine", "hello", "hi", "hey"
}

ENTITY_LABEL_MAP = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "NORP": "GROUP",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "LOCATION",
    "EVENT": "EVENT"
}

ENTITY_TYPE_PRIORITY = {
    "LOCATION": 5,
    "PERSON": 4,
    "ORG": 4,
    "GROUP": 3,
    "EVENT": 2,
    "ANIMAL": 1,
    "UNKNOWN": 0
}

LOCATION_TYPE_WORDS = {
    "city", "town", "village", "district", "state", "country", "capital", "province", "region", "location"
}

KNOWN_LOCATIONS = {
    "india", "noida", "delhi", "new delhi", "mumbai", "bangalore", "bengaluru", "kolkata",
    "hyderabad", "pune", "chennai", "gurgaon", "gurugram", "uttar pradesh", "new york",
    "london", "paris", "tokyo", "sydney", "california", "singapore", "dubai"
}

ANIMAL_TERMS = {
    "dog", "cat", "cow", "horse", "lion", "tiger", "elephant", "rabbit", "monkey", "bird", "fish", "snake"
}

GENERIC_ENTITY_STOPWORDS = {
    "the", "this", "that", "these", "those", "there", "here", "today", "tomorrow", "yesterday", "none"
}

PERSON_TITLE_TERMS = {"mr", "mrs", "ms", "dr", "prof", "sir", "madam"}

BIAS_GROUPS = {
    "male": {"he", "him", "his", "man", "male", "father", "boy", "men", "boys"},
    "female": {"she", "her", "hers", "woman", "female", "mother", "girl", "women", "girls"},
    "white": {"white", "caucasian", "european"},
    "black": {"black", "african"},
    "asian": {"asian", "indian", "chinese", "japanese", "korean"},
    "latino": {"latino", "latina", "hispanic"}
}

COLOR_KEYWORDS = {
    "red": np.array([210, 60, 60]),
    "blue": np.array([70, 120, 220]),
    "green": np.array([70, 170, 90]),
    "yellow": np.array([220, 210, 80]),
    "orange": np.array([225, 140, 60]),
    "purple": np.array([140, 90, 190]),
    "pink": np.array([220, 130, 170]),
    "black": np.array([35, 35, 35]),
    "white": np.array([225, 225, 225]),
    "gray": np.array([140, 140, 140]),
    "grey": np.array([140, 140, 140])
}

BRIGHT_WORDS = {"bright", "sunny", "day", "light", "vivid"}
DARK_WORDS = {"dark", "night", "shadow", "moody", "dim"}
GRAYSCALE_WORDS = {"black and white", "monochrome", "grayscale", "greyscale"}
DETAIL_WORDS = {"detailed", "intricate", "high detail", "complex"}


def _normalize_token(token):
    cleaned = re.sub(r"[^a-z0-9]+", "", token.lower())
    if not cleaned:
        return ""
    return TOKEN_EQUIVALENTS.get(cleaned, cleaned)


def unigrams(text, drop_stopwords=False):
    """Extract normalized tokens from text"""
    if tokenizer is None:
        raw_tokens = re.findall(r"[A-Za-z0-9']+", text)
    else:
        raw_tokens = tokenizer.tokenize(text)

    tokens = []
    for raw_token in raw_tokens:
        normalized = _normalize_token(raw_token)
        if not normalized:
            continue
        if drop_stopwords and normalized in STOPWORDS:
            continue
        tokens.append(normalized)
    return tokens


def _safe_cosine_similarity(text_a, text_b):
    """Compute semantic similarity with transformer fallback to lexical similarity."""
    if not text_a.strip() or not text_b.strip():
        return 0.0

    if embedder is not None:
        try:
            emb_a = embedder.encode(text_a, convert_to_tensor=True)
            emb_b = embedder.encode(text_b, convert_to_tensor=True)
            similarity = float(util.pytorch_cos_sim(emb_a, emb_b).item())
            return max(0.0, min(1.0, similarity))
        except Exception:
            pass

    return SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()


def _normalized_phrase(text):
    tokens = unigrams(text)
    return " ".join(tokens)


def _is_small_talk_prompt(text):
    normalized = _normalized_phrase(text)
    return any(prompt in normalized for prompt in SMALL_TALK_PROMPTS)


def _is_small_talk_response(text):
    normalized = _normalized_phrase(text)
    return any(resp in normalized for resp in SMALL_TALK_RESPONSES)


def _is_echo_response(query_text, response_text):
    if _is_small_talk_prompt(query_text) and _is_small_talk_response(response_text):
        return False

    query_tokens = set(unigrams(query_text, drop_stopwords=True))
    response_tokens = unigrams(response_text, drop_stopwords=True)
    if not response_tokens:
        return False

    response_set = set(response_tokens)
    if len(response_tokens) <= 2 and response_set and response_set.issubset(query_tokens):
        return True

    q_phrase = _normalized_phrase(query_text)
    r_phrase = _normalized_phrase(response_text)
    return len(response_tokens) <= 3 and bool(r_phrase) and r_phrase in q_phrase


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))

def rouge1_f1(reference, candidate):
    """Calculate ROUGE-1 F1 score"""
    ref_unigrams = unigrams(reference, drop_stopwords=True)
    cand_unigrams = unigrams(candidate, drop_stopwords=True)
    if not ref_unigrams or not cand_unigrams:
        return 0.0

    ref_counts = Counter(ref_unigrams)
    cand_counts = Counter(cand_unigrams)
    overlap = sum(min(ref_counts[t], cand_counts[t]) for t in set(ref_counts) | set(cand_counts))
    prec = overlap / (sum(cand_counts.values()) + 1e-5)
    rec = overlap / (sum(ref_counts.values()) + 1e-5)
    if prec + rec == 0:
        return 0.0

    score = 2 * prec * rec / (prec + rec)
    if _is_echo_response(reference, candidate):
        score *= 0.2
    return _clamp(score)

def length_fit(query_text, response_text, expected=EXPECTED_LEN_RATIO, lam=LAMBDA_LEN):
    """Evaluate response length fitness"""
    q_len = max(1, len(unigrams(query_text)))
    r_len = max(1, len(unigrams(response_text)))
    ratio = r_len / q_len
    return math.exp(-lam * abs(math.log(ratio / expected)))

UNCERTAINTY_MARKERS = {"maybe","not","unsure","uncertain","perhaps","guess","might","could"}

def coherence_score(query_text, response_text, alpha=ALPHA_COH):
    """Measure coherence of response with uncertainty and relevance blend"""
    toks = unigrams(response_text)
    if not toks:
        return 0.0

    unc_density = sum(1 for t in toks if t in UNCERTAINTY_MARKERS) / len(toks)
    rel = relevance_score(query_text, response_text)
    echo_penalty = 0.25 if _is_echo_response(query_text, response_text) else 0.0
    score = alpha * (1 - unc_density) + (1 - alpha) * rel - echo_penalty
    return _clamp(score)

def toxicity_penalty(response_text, gamma=GAMMA_TOX):
    """Detect toxicity in text with convex penalty"""
    try:
        if toxicity_model is None:
            raise Exception("Toxicity model not loaded")
        tox_prob = float(toxicity_model.predict(response_text)["toxicity"])
        penalty = 1 - math.exp(-gamma * tox_prob)
    except Exception:
        toks = unigrams(response_text)
        toxic_terms = {"stupid", "idiot", "hate", "kill", "trash", "dumb"}
        toxic_count = sum(1 for t in toks if t in toxic_terms)
        ratio = toxic_count / (len(toks) + 1e-5)
        penalty = 1 - math.exp(-gamma * ratio)
    return max(0.0, min(1.0, penalty))

def _extract_bias_counts(tokens):
    return {group: sum(1 for token in tokens if token in words) for group, words in BIAS_GROUPS.items()}


def bias_penalty(response_text, gamma=GAMMA_BIAS, return_breakdown=False):
    """Calculate bias penalty based on imbalance (balanced opposites cancel out)."""
    tokens = unigrams(response_text)
    counts = _extract_bias_counts(tokens)

    male_count = counts.get("male", 0)
    female_count = counts.get("female", 0)
    gender_total = male_count + female_count
    gender_imbalance = abs(male_count - female_count) / (gender_total + 1e-5) if gender_total else 0.0

    race_counts = [counts.get("white", 0), counts.get("black", 0), counts.get("asian", 0), counts.get("latino", 0)]
    race_total = sum(race_counts)
    non_zero_race = [count for count in race_counts if count > 0]

    if race_total == 0:
        race_imbalance = 0.0
    elif len(non_zero_race) <= 1:
        race_imbalance = 1.0
    else:
        race_imbalance = (max(non_zero_race) - min(non_zero_race)) / (race_total + 1e-5)

    overall_imbalance = 0.7 * gender_imbalance + 0.3 * race_imbalance
    penalty = 1 - math.exp(-gamma * overall_imbalance)
    penalty = _clamp(penalty)

    breakdown = {
        "Male Terms": male_count,
        "Female Terms": female_count,
        "Gender Imbalance": round(gender_imbalance, 3),
        "White Mentions": counts.get("white", 0),
        "Black Mentions": counts.get("black", 0),
        "Asian Mentions": counts.get("asian", 0),
        "Latino Mentions": counts.get("latino", 0),
        "Race Imbalance": round(race_imbalance, 3)
    }

    if return_breakdown:
        return penalty, breakdown
    return penalty

def detect_bias(response):
    """Detect bias in response with entity extraction"""
    entities = []
    if nlp is not None:
        try:
            doc = nlp(response)
            entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in {"PERSON", "NORP"}]
        except Exception:
            pass

    pen, breakdown = bias_penalty(response, return_breakdown=True)
    return {
        "Bias Penalty": round(pen, 3),
        "Entity Analysis": breakdown,
        "Named Entities": entities
    }


def _normalize_entity_text(entity_text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]+", "", entity_text.lower())).strip()


def _entity_rank(entity_type, source):
    source_bonus = 100 if source == "nlp" else 0
    return source_bonus + ENTITY_TYPE_PRIORITY.get(entity_type, 0)


def _infer_regex_entity_type(candidate_text, full_text):
    normalized = _normalize_entity_text(candidate_text)
    if not normalized or normalized in GENERIC_ENTITY_STOPWORDS:
        return None

    tokens = normalized.split()
    text_lower = full_text.lower()

    if normalized in ANIMAL_TERMS:
        return "ANIMAL"

    if normalized in KNOWN_LOCATIONS:
        return "LOCATION"

    escaped_candidate = re.escape(candidate_text.strip())

    if re.search(
        rf"\b{escaped_candidate}\s+is\s+(?:a|an)\s+(?:{'|'.join(LOCATION_TYPE_WORDS)})\b",
        full_text,
        flags=re.IGNORECASE
    ):
        return "LOCATION"

    if re.search(
        rf"\b(?:in|at|near|from)\s+{escaped_candidate}\b",
        full_text,
        flags=re.IGNORECASE
    ):
        return "LOCATION"

    if len(tokens) >= 2:
        if normalized in KNOWN_LOCATIONS or any(token in KNOWN_LOCATIONS for token in tokens):
            return "LOCATION"
        return "PERSON"

    token = tokens[0]

    if token in ANIMAL_TERMS:
        return "ANIMAL"

    if re.search(
        rf"\b(?:{'|'.join(PERSON_TITLE_TERMS)})\.?\s+{escaped_candidate}\b",
        text_lower,
        flags=re.IGNORECASE
    ):
        return "PERSON"

    if re.search(
        rf"\b(?:i\s+am|i'm|my\s+name\s+is|name\s+is|this\s+is)\s+{escaped_candidate}\b",
        text_lower,
        flags=re.IGNORECASE
    ):
        return "PERSON"

    return None


def _extract_named_entities(text):
    entities_by_key = {}

    def upsert_entity(entity_text, entity_type, source):
        normalized = _normalize_entity_text(entity_text)
        if not normalized or not entity_type:
            return

        current = entities_by_key.get(normalized)
        candidate = {"text": entity_text.strip(), "type": entity_type, "_source": source}

        if current is None or _entity_rank(entity_type, source) > _entity_rank(current["type"], current["_source"]):
            entities_by_key[normalized] = candidate

    if nlp is not None:
        try:
            doc = nlp(text)
            for ent in doc.ents:
                mapped_type = ENTITY_LABEL_MAP.get(ent.label_)
                if mapped_type:
                    upsert_entity(ent.text, mapped_type, "nlp")
        except Exception:
            pass

    regex_candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", text)
    for candidate in regex_candidates:
        inferred_type = _infer_regex_entity_type(candidate, text)
        if inferred_type:
            upsert_entity(candidate, inferred_type, "regex")

    unique_entities = []
    for entity in entities_by_key.values():
        unique_entities.append({"text": entity["text"], "type": entity["type"]})
    return unique_entities


def _is_entity_supported(response_entity, reference_entities):
    response_norm = _normalize_entity_text(response_entity["text"])
    response_tokens = set(response_norm.split())
    if not response_norm:
        return False

    for reference_entity in reference_entities:
        reference_norm = _normalize_entity_text(reference_entity["text"])
        reference_tokens = set(reference_norm.split())

        if response_norm == reference_norm:
            return True
        if response_norm in reference_norm:
            return True
        if response_tokens and response_tokens.issubset(reference_tokens):
            return True
    return False

def detect_hallucination(reference, response):
    """Detect hallucinations using entity support + semantic/lexical consistency."""
    reference_entities = _extract_named_entities(reference)
    response_entities = _extract_named_entities(response)
    hallucinated_entities = []

    if response_entities:
        supported_response_entities = [
            entity for entity in response_entities if _is_entity_supported(entity, reference_entities)
        ]
        hallucinated_entities = [
            entity for entity in response_entities if not _is_entity_supported(entity, reference_entities)
        ]

        if reference_entities:
            supported_reference_entities = [
                entity for entity in reference_entities if _is_entity_supported(entity, response_entities)
            ]
            precision = len(supported_response_entities) / (len(response_entities) + 1e-5)
            recall = len(supported_reference_entities) / (len(reference_entities) + 1e-5)
            f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))
        else:
            f1 = len(supported_response_entities) / (len(response_entities) + 1e-5)
    else:
        reference_tokens = set(unigrams(reference, drop_stopwords=True))
        response_tokens = set(unigrams(response, drop_stopwords=True))
        lexical_similarity = len(reference_tokens & response_tokens) / (len(reference_tokens | response_tokens) + 1e-5)
        semantic_similarity = _safe_cosine_similarity(reference, response)
        f1 = (0.4 * lexical_similarity) + (0.6 * semantic_similarity)

    toks = unigrams(response)
    unc_density = sum(1 for t in toks if t in UNCERTAINTY_MARKERS) / (len(toks) + 1e-5)
    hall_risk_unc = min(1.0, unc_density * HALL_MULT)
    entity_penalty = min(0.25, len(hallucinated_entities) * 0.1)
    risk = (1 - f1) * 0.75 + hall_risk_unc * 0.25 + entity_penalty

    if _is_small_talk_prompt(reference) and _is_small_talk_response(response):
        risk *= 0.7

    risk = _clamp(risk)
    return round(risk, 3), hallucinated_entities

def relevance_score(query, response):
    """Calculate relevance with lexical, semantic, and conversational intent signals."""
    query_words = set(unigrams(query, drop_stopwords=True))
    response_words = set(unigrams(response, drop_stopwords=True))

    if not query_words and not response_words:
        return 0.0

    overlap = len(query_words & response_words) / (len(query_words) + 1e-5) if query_words else 0.0
    semantic = _safe_cosine_similarity(query, response)
    small_talk_boost = 1.0 if _is_small_talk_prompt(query) and _is_small_talk_response(response) else 0.0

    score = 0.45 * overlap + 0.45 * semantic + 0.10 * small_talk_boost
    if _is_echo_response(query, response):
        score *= 0.35
    return _clamp(score)

def composite_score(metrics):
    """Calculate composite score from all metrics"""
    return sum(metrics.get(k, 0) * v for k, v in WEIGHTS.items())

def evaluate_response(reference, response):
    """Evaluate a single response"""
    ensure_models_loaded()

    cosine_sim = _safe_cosine_similarity(reference, response)
    
    r1 = rouge1_f1(reference, response)
    len_fit = length_fit(reference, response)
    tox_pen = toxicity_penalty(response)
    hall_risk, hall_entities = detect_hallucination(reference, response)
    bias_info = detect_bias(response)
    bias_pen = bias_info["Bias Penalty"]
    rel = relevance_score(reference, response)
    coh = coherence_score(reference, response)
    
    final = composite_score({
        "relevance": rel,
        "length_fit": len_fit,
        "coherence": coh,
        "rouge1_f1": r1,
        "toxicity_inv": 1 - tox_pen,
        "bias_inv": 1 - bias_pen,
        "hall_inv": 1 - hall_risk
    })
    
    return {
        "semantic_similarity": round(cosine_sim, 3),
        "rouge1_f1": round(r1, 3),
        "length_fit": round(len_fit, 3),
        "relevance": round(rel, 3),
        "coherence": round(coh, 3),
        "toxicity_penalty": round(tox_pen, 3),
        "bias_penalty": round(bias_pen, 3),
        "hallucination_risk": round(hall_risk, 3),
        "final_score": round(final, 3),
        "hallucinated_entities": hall_entities
    }

# --- Flask Routes ---
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Main evaluation dashboard"""
    return render_template('dashboard.html')

@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    """API endpoint for single response evaluation"""
    try:
        ensure_models_loaded()
        data = request.json
        reference = data.get('reference', '').strip()
        response = data.get('response', '').strip()
        
        if not reference or not response:
            return jsonify({"error": "Reference and response are required"}), 400
        
        results = evaluate_response(reference, response)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": f"Evaluation failed: {str(e)}"}), 500

@app.route('/api/detect-hallucination', methods=['POST'])
def api_detect_hallucination():
    """API endpoint for hallucination detection"""
    try:
        ensure_models_loaded()
        data = request.json
        reference = data.get('reference', '').strip()
        response = data.get('response', '').strip()
        
        if not reference or not response:
            return jsonify({"error": "Reference and response are required"}), 400
        
        risk, entities = detect_hallucination(reference, response)
        return jsonify({
            "hallucination_risk": round(risk, 3),
            "hallucinated_entities": entities,
            "explanation": f"Risk Score: {risk:.2f}/1.0 (higher = more likely to contain hallucinations)"
        })
    except Exception as e:
        return jsonify({"error": f"Hallucination detection failed: {str(e)}"}), 500

@app.route('/api/detect-bias', methods=['POST'])
def api_detect_bias():
    """API endpoint for bias detection"""
    try:
        ensure_models_loaded()
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "Text is required"}), 400
        
        bias_info = detect_bias(text)
        return jsonify({
            "bias_score": round(bias_info["Bias Penalty"], 3),
            "entity_analysis": bias_info["Entity Analysis"],
            "named_entities": bias_info["Named Entities"],
            "explanation": f"Bias Score: {bias_info['Bias Penalty']:.2f}/1.0"
        })
    except Exception as e:
        return jsonify({"error": f"Bias detection failed: {str(e)}"}), 500

@app.route('/api/check-toxicity', methods=['POST'])
def api_check_toxicity():
    """API endpoint for toxicity check"""
    try:
        ensure_models_loaded()
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "Text is required"}), 400
        
        toxicity = toxicity_penalty(text)
        return jsonify({
            "toxicity_score": round(toxicity, 3),
            "is_safe": toxicity < SAFETY_THRESHOLD,
            "explanation": f"Toxicity Level: {toxicity:.2f}/1.0"
        })
    except Exception as e:
        return jsonify({"error": f"Toxicity check failed: {str(e)}"}), 500

@app.route('/api/compare-models', methods=['POST'])
def api_compare_models():
    """API endpoint for multi-model comparison"""
    try:
        ensure_models_loaded()
        data = request.json
        question = data.get('question', '').strip()
        models = data.get('models', [])
        
        if not question or not models or len(models) < 2:
            return jsonify({"error": "Question and at least 2 models required"}), 400
        
        comparisons = []
        for model in models:
            model_name = model.get('name', 'Unknown')
            model_response = model.get('response', '').strip()
            
            if not model_response:
                continue
            
            eval_result = evaluate_response(question, model_response)
            eval_result['model_name'] = model_name
            comparisons.append(eval_result)
        
        if not comparisons:
            return jsonify({"error": "No valid model responses provided"}), 400
        
        # Find winner (highest final score)
        winner_index = max(range(len(comparisons)), key=lambda i: comparisons[i]['final_score'])
        
        return jsonify({
            "comparisons": comparisons,
            "winner_index": winner_index,
            "winner": comparisons[winner_index]['model_name']
        })
    except Exception as e:
        return jsonify({"error": f"Model comparison failed: {str(e)}"}), 500

def analyze_image_properties(image_data):
    """Analyze image properties and extract pixel features for prompt alignment."""
    try:
        img_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
        raw_img = Image.open(io.BytesIO(img_bytes))
        img_format = raw_img.format or "Unknown"
        rgb_img = raw_img.convert("RGB")

        analysis_img = rgb_img.copy()
        analysis_img.thumbnail((256, 256))
        img_array = np.array(analysis_img).astype(np.float32)

        avg_color = img_array.mean(axis=(0, 1))
        gray = img_array.mean(axis=2)

        brightness = float(np.clip(gray.mean() / 255.0, 0.0, 1.0))
        contrast = float(np.clip(gray.std() / 128.0, 0.0, 1.0))

        if gray.shape[1] > 1:
            gx = np.abs(np.diff(gray, axis=1)).mean()
        else:
            gx = 0.0

        if gray.shape[0] > 1:
            gy = np.abs(np.diff(gray, axis=0)).mean()
        else:
            gy = 0.0

        edge_density = float(np.clip((gx + gy) / (2 * 255.0), 0.0, 1.0))

        channel_delta = (
            np.abs(img_array[:, :, 0] - img_array[:, :, 1]) +
            np.abs(img_array[:, :, 1] - img_array[:, :, 2]) +
            np.abs(img_array[:, :, 0] - img_array[:, :, 2])
        ) / 3.0
        grayscale_score = float(np.clip(1.0 - (channel_delta.mean() / 255.0), 0.0, 1.0))

        color_strengths = {}
        for color_name, color_rgb in COLOR_KEYWORDS.items():
            distance = np.linalg.norm(img_array - color_rgb, axis=2)
            color_strengths[color_name] = float(np.mean(distance < 80.0))

        dominant_color = max(color_strengths.items(), key=lambda item: item[1])[0]

        properties = {
            "Format": img_format,
            "Size (pixels)": f"{raw_img.width}x{raw_img.height}",
            "Mode": raw_img.mode,
            "Avg Color (RGB)": f"({int(avg_color[0])}, {int(avg_color[1])}, {int(avg_color[2])})",
            "Dominant Color": dominant_color.title(),
            "Brightness": f"{brightness * 100:.1f}%",
            "Contrast": f"{contrast * 100:.1f}%"
        }

        pixel_features = {
            "brightness": brightness,
            "contrast": contrast,
            "edge_density": edge_density,
            "grayscale_score": grayscale_score,
            "color_strengths": color_strengths,
            "dominant_color": dominant_color
        }

        return properties, pixel_features
    except Exception as e:
        return {"error": str(e)}, {}


def _pixel_prompt_alignment(prompt_text, pixel_features):
    """Estimate prompt-image alignment directly from pixel features."""
    prompt_lower = prompt_text.lower()
    prompt_tokens = set(unigrams(prompt_text))

    color_mentions = [
        color for color in COLOR_KEYWORDS.keys()
        if re.search(rf"\\b{re.escape(color)}\\b", prompt_lower)
    ]

    if color_mentions:
        color_scores = [
            min(1.0, pixel_features.get("color_strengths", {}).get(color, 0.0) * 8.0)
            for color in color_mentions
        ]
        color_score = sum(color_scores) / len(color_scores)
    else:
        color_score = 0.65

    if prompt_tokens & BRIGHT_WORDS:
        brightness_score = pixel_features.get("brightness", 0.5)
    elif prompt_tokens & DARK_WORDS:
        brightness_score = 1.0 - pixel_features.get("brightness", 0.5)
    else:
        brightness_score = 0.65

    if any(keyword in prompt_lower for keyword in GRAYSCALE_WORDS):
        grayscale_score = pixel_features.get("grayscale_score", 0.5)
    else:
        grayscale_score = 0.7

    if any(keyword in prompt_lower for keyword in DETAIL_WORDS):
        detail_score = pixel_features.get("edge_density", 0.5)
    else:
        detail_score = 0.6 + (0.4 * pixel_features.get("edge_density", 0.5))

    pixel_match = _clamp(
        0.45 * color_score +
        0.20 * brightness_score +
        0.15 * grayscale_score +
        0.20 * detail_score
    )

    keyword_coverage = _clamp(0.55 * color_score + 0.20 * brightness_score + 0.25 * detail_score)

    auto_tags = [pixel_features.get("dominant_color", "unknown")]
    auto_tags.append("bright" if pixel_features.get("brightness", 0.5) >= 0.6 else "dark")
    if pixel_features.get("grayscale_score", 0.0) >= 0.85:
        auto_tags.append("monochrome")
    auto_tags.append("detailed" if pixel_features.get("edge_density", 0.0) >= 0.22 else "minimal")

    return {
        "pixel_match": pixel_match,
        "keyword_coverage": keyword_coverage,
        "detail_score": _clamp(detail_score),
        "auto_tags": auto_tags
    }

def multimodal_evaluate(image_data, prompt_text, description_text=None):
    """Evaluate AI-generated image against prompt using pixel-first analysis."""
    if not image_data or not prompt_text.strip():
        return None, None, {}, "Please provide both an image and the prompt used to generate it."

    try:
        image_props, pixel_features = analyze_image_properties(image_data)
        if "error" in image_props:
            return None, image_props, {}, f"❌ Error reading image: {image_props['error']}"

        pixel_eval = _pixel_prompt_alignment(prompt_text, pixel_features)
        has_description = bool(description_text and description_text.strip())

        text_similarity = 0.0
        text_rouge = 0.0
        text_relevance = 0.0
        text_coherence = 0.0
        text_hall_risk = 1.0 - pixel_eval["pixel_match"]
        hallucinated = []

        if has_description:
            text_similarity = _safe_cosine_similarity(prompt_text, description_text)
            text_rouge = rouge1_f1(prompt_text, description_text)
            text_relevance = relevance_score(prompt_text, description_text)
            text_coherence = coherence_score(prompt_text, description_text)
            text_hall_risk, hallucinated = detect_hallucination(prompt_text, description_text)

        prompt_match_score = _clamp(
            (0.75 * pixel_eval["pixel_match"]) +
            ((0.25 * text_similarity) if has_description else 0.0)
        )

        keyword_overlap = _clamp(
            (0.7 * pixel_eval["keyword_coverage"]) +
            ((0.3 * text_rouge) if has_description else 0.0)
        )

        relevance = _clamp(
            0.6 * prompt_match_score +
            0.15 * pixel_eval["keyword_coverage"] +
            ((0.25 * text_relevance) if has_description else 0.0)
        )

        visual_coherence = _clamp(
            (0.7 * pixel_eval["detail_score"]) +
            (0.3 * (1.0 - abs(0.5 - pixel_features.get("brightness", 0.5))))
        )
        coherence = _clamp((0.4 * text_coherence if has_description else 0.0) + (0.6 * visual_coherence))

        hall_risk = _clamp((0.6 * (1.0 - pixel_eval["pixel_match"])) + (0.4 * text_hall_risk))

        toxicity_input = description_text if has_description else prompt_text
        toxicity = toxicity_penalty(toxicity_input)
        safety_score = _clamp(1.0 - toxicity)

        accuracy_score = _clamp(
            0.32 * prompt_match_score +
            0.20 * keyword_overlap +
            0.16 * relevance +
            0.12 * coherence +
            0.12 * (1.0 - hall_risk) +
            0.08 * safety_score
        )

        eval_results = {
            "Prompt-Image Match Score": round(prompt_match_score, 3),
            "Keyword Overlap (ROUGE-1)": round(keyword_overlap, 3),
            "Relevance to Prompt": round(relevance, 3),
            "Description Coherence": round(coherence, 3),
            "Hallucination Risk": round(hall_risk, 3),
            "Overall Accuracy": round(accuracy_score, 3),
            "Safety Score": round(safety_score, 3)
        }

        explanation = "🎨 AI Image Generation Evaluation\n\n"
        explanation += "🧠 Method: Pixel-first automatic analysis"
        explanation += " (dominant color, brightness, contrast, detail, and prompt cues)"
        explanation += " with optional text-description refinement.\n\n"

        explanation += "📊 Accuracy Analysis:\n"
        explanation += f"✅ Overall Accuracy: {round(accuracy_score * 100, 1)}%\n"
        explanation += f"🎯 Prompt Match: {round(prompt_match_score * 100, 1)}%\n"
        explanation += f"🔤 Keyword Coverage: {round(keyword_overlap * 100, 1)}%\n"
        explanation += f"🎲 Relevance: {round(relevance * 100, 1)}%\n"
        explanation += f"🧠 Coherence: {round(coherence * 100, 1)}%\n"
        explanation += f"🛡️ Safety Score: {round(safety_score * 100, 1)}%\n"
        explanation += f"🖼️ Pixel Tags: {', '.join(pixel_eval['auto_tags'])}\n\n"

        if not has_description:
            explanation += "ℹ️ No manual description provided. Scores were computed directly from image pixels + prompt text.\n\n"

        if hall_risk > 0.5:
            explanation += "⚠️ High hallucination risk: generated image likely diverges from requested intent.\n"
            if hallucinated:
                explanation += f"   Found {len(hallucinated)} potential unsupported entities in description.\n"
        elif hall_risk > 0.3:
            explanation += "⚡ Moderate hallucination risk: partial creative deviation detected.\n"
        else:
            explanation += "✅ Low hallucination risk: image closely follows prompt intent.\n"

        explanation += f"\n📐 Image Properties: {image_props.get('Size (pixels)', 'N/A')} | {image_props.get('Mode', 'N/A')}\n"
        explanation += "\n💡 Recommendations:\n"
        if accuracy_score > 0.8:
            explanation += "- Excellent match! The AI accurately interpreted your prompt.\n"
        elif accuracy_score > 0.6:
            explanation += "- Good match with minor deviations from the prompt.\n"
        elif accuracy_score > 0.4:
            explanation += "- Moderate match. Consider refining your prompt for better results.\n"
        else:
            explanation += "- Low match. The AI may have misunderstood the prompt. Try being more specific.\n"

        if hall_risk > 0.4:
            explanation += "- The AI added unexpected elements. Specify what NOT to include in prompts.\n"

        return eval_results, image_props, eval_results, explanation

    except Exception as e:
        return None, {}, {}, f"❌ Error analyzing multimodal content: {str(e)}"

@app.route('/api/evaluate-image', methods=['POST'])
def api_evaluate_image():
    """API endpoint for AI image generation evaluation"""
    ensure_models_loaded()
    data = request.json
    image_data = data.get('image', '')
    prompt = data.get('prompt', '').strip()
    description = data.get('description', '').strip()
    
    if not image_data or not prompt:
        return jsonify({"error": "Image and prompt are required"}), 400
    
    try:
        metrics, props, _, explanation = multimodal_evaluate(image_data, prompt, description)
        
        if metrics is None:
            return jsonify({"error": explanation}), 500
        
        return jsonify({
            "metrics": metrics,
            "image_properties": props,
            "explanation": explanation
        })
    except Exception as e:
        return jsonify({"error": f"Image evaluation failed: {str(e)}"}), 500

def check_code_quality(code_str):
    """Evaluate code quality metrics with explanation and improvements"""
    results = {
        "syntax_valid": False,
        "errors": [],
        "metrics": {},
        "suggestions": [],
        "explanation": "",
        "improved_code": ""
    }
    
    if not code_str.strip():
        results["explanation"] = "No code provided."
        return results
    
    # Check syntax
    try:
        tree = ast.parse(code_str)
        results["syntax_valid"] = True
    except SyntaxError as e:
        results["errors"].append(f"Syntax Error: {str(e)}")
        results["explanation"] = f"❌ Code contains syntax errors and cannot be executed.\n\nError: {str(e)}"
        return results
    
    # Count lines and complexity
    lines = code_str.split('\n')
    non_empty_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
    comment_lines = [l for l in lines if l.strip().startswith('#')]
    
    # Analyze code structure
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    
    results["metrics"] = {
        "Total Lines": len(lines),
        "Code Lines": len(non_empty_lines),
        "Comment Lines": len(comment_lines),
        "Comment Ratio": round(len(comment_lines) / (len(lines) + 1e-5), 3),
        "Functions": len(functions),
        "Classes": len(classes),
        "Imports": len(imports)
    }
    
    # Generate explanation
    explanation = "📝 Code Analysis Summary\n\n"
    
    if functions:
        explanation += f"🔧 Functions Found: {', '.join(functions[:5])}"
        if len(functions) > 5:
            explanation += f" and {len(functions) - 5} more"
        explanation += "\n"
    
    if classes:
        explanation += f"📦 Classes Found: {', '.join(classes)}\n"
    
    if imports:
        explanation += f"📚 Imports: {len(imports)} module(s) imported\n"
    
    explanation += f"\n📊 Code Structure:\n"
    explanation += f"- Total Lines: {len(lines)}\n"
    explanation += f"- Actual Code: {len(non_empty_lines)} lines\n"
    explanation += f"- Comments: {len(comment_lines)} lines ({round(len(comment_lines) / (len(lines) + 1e-5) * 100, 1)}%)\n"
    explanation += f"- Blank Lines: {len(lines) - len(non_empty_lines) - len(comment_lines)}\n\n"
    
    # Purpose analysis
    explanation += "🎯 Code Purpose:\n"
    if functions and classes:
        explanation += "This code defines both functions and classes, suggesting it's part of a larger module or library.\n"
    elif functions:
        explanation += f"This code defines {len(functions)} function(s) for specific tasks.\n"
    elif classes:
        explanation += f"This code defines {len(classes)} class(es) for object-oriented programming.\n"
    else:
        explanation += "This appears to be a script with sequential instructions.\n"
    
    results["explanation"] = explanation
    
    # Check for code smells and issues
    improvements = []
    
    if len(non_empty_lines) > 100:
        results["suggestions"].append("⚠️ Function is quite long (>100 lines). Consider breaking it down.")
        improvements.append("Break down large functions into smaller, reusable components")
    
    if len(comment_lines) == 0:
        results["suggestions"].append("⚠️ No comments found. Add documentation.")
        improvements.append("Add comments to explain complex logic")
    
    if code_str.count('try:') == 0 and code_str.count('except') == 0:
        results["suggestions"].append("⚠️ No error handling detected. Consider adding try-catch blocks.")
        improvements.append("Add try-except blocks for robust error handling")
    
    # Security checks
    if 'eval(' in code_str:
        results["errors"].append("🚨 Security Issue: Use of eval() detected!")
        improvements.append("Replace eval() with safer alternatives like ast.literal_eval()")
    
    if 'exec(' in code_str:
        results["errors"].append("🚨 Security Issue: Use of exec() detected!")
        improvements.append("Avoid exec() - redesign code to not execute arbitrary strings")
    
    # Best practices
    if not functions and not classes and len(non_empty_lines) > 20:
        improvements.append("Organize code into functions for better maintainability")
    
    if len(comment_lines) / (len(lines) + 1e-5) < 0.1 and len(lines) > 50:
        improvements.append("Increase comment coverage for better code documentation")
    
    # Generate improved code
    if improvements:
        improved_code = f"# IMPROVED VERSION\n"
        improved_code += f"# Improvements suggested:\n"
        for i, imp in enumerate(improvements, 1):
            improved_code += f"# {i}. {imp}\n"
        improved_code += f"\n"
        
        # Add missing docstring if functions exist
        if functions and '"""' not in code_str and "'''" not in code_str:
            improved_code += '"""\nModule for [describe purpose].\n"""\n\n'
        
        # Add error handling wrapper if missing
        if 'try:' not in code_str and len(functions) > 0:
            improved_code += "# Add error handling:\n"
            improved_code += "try:\n"
            improved_code += "    " + code_str.replace("\n", "\n    ")
            improved_code += "\nexcept Exception as e:\n"
            improved_code += "    print(f'Error: {e}')\n"
            improved_code += "    # Handle error appropriately\n"
        else:
            improved_code += code_str
        
        results["improved_code"] = improved_code
    else:
        results["improved_code"] = code_str
        results["suggestions"].append("✅ Code quality is good!")
    
    if not results["suggestions"]:
        results["suggestions"].append("✅ Code looks good!")

    # Calculate a more variable quality score (0-1)
    total_lines = results["metrics"].get("Total Lines", 1) or 1
    comment_lines = results["metrics"].get("Comment Lines", 0)
    code_lines = results["metrics"].get("Code Lines", 1) or 1
    functions_count = results["metrics"].get("Functions", 0)
    comment_ratio = comment_lines / total_lines
    avg_function_length = code_lines / max(functions_count, 1)

    score = 0.9
    if results["errors"]:
        score -= 0.25

    score -= min(0.3, len(results["suggestions"]) * 0.04)

    if comment_ratio >= 0.15:
        score += 0.05
    elif comment_ratio < 0.03:
        score -= 0.12
    elif comment_ratio < 0.08:
        score -= 0.05

    if avg_function_length > 120:
        score -= 0.15
    elif avg_function_length > 60:
        score -= 0.07
    elif avg_function_length < 20:
        score += 0.03

    if total_lines > 300:
        score -= 0.1
    elif total_lines < 15:
        score -= 0.05

    if functions_count == 0 and total_lines > 30:
        score -= 0.05

    results["quality_score"] = round(max(0.1, min(0.98, score)), 3)
    
    return results

@app.route('/api/analyze-code', methods=['POST'])
def api_analyze_code():
    """API endpoint for code quality analysis"""
    ensure_models_loaded()
    data = request.json
    code = data.get('code', '').strip()
    language = data.get('language', 'python').lower()
    
    if not code:
        return jsonify({"error": "Code is required"}), 400
    
    try:
        results = check_code_quality(code)
        return jsonify({
            "syntax_valid": results["syntax_valid"],
            "metrics": results["metrics"],
            "errors": results["errors"],
            "suggestions": results["suggestions"],
            "explanation": results["explanation"],
            "improved_code": results["improved_code"],
            "quality_score": results.get("quality_score"),
            "language": language
        })
    except Exception as e:
        return jsonify({"error": f"Code analysis failed: {str(e)}"}), 500


# ===== Enhanced Code Analysis Endpoints (Ollama-powered) =====

@app.route('/api/analyze-code-enhanced', methods=['POST'])
def api_analyze_code_enhanced():
    """Enhanced code analysis using Ollama AI"""
    if not CODE_ANALYZER_AVAILABLE:
        return jsonify({"error": "Enhanced analyzer not available. Install dependencies and start Ollama."}), 503
    
    data = request.json
    code = data.get('code', '').strip()
    language = data.get('language', 'python').lower()
    analysis_type = data.get('analysis_type', 'full')  # full, bugs, security, improve
    
    if not code:
        return jsonify({"error": "Code is required"}), 400
    
    try:
        analyzer = get_analyzer()
        
        if analysis_type == 'bugs':
            result = analyzer.find_bugs(code, language)
        elif analysis_type == 'security':
            result = analyzer.security_analysis(code, language)
        elif analysis_type == 'improve':
            result = analyzer.generate_improved_code(code, language)
        elif analysis_type == 'documentation':
            result = analyzer.generate_documentation(code, language)
        else:  # full analysis
            result = analyzer.analyze_code_snippet(code, language)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Enhanced analysis failed: {str(e)}"}), 500


@app.route('/api/analyze-llm-code', methods=['POST'])
def api_analyze_llm_code():
    """Analyze LLM-generated code quality"""
    if not CODE_ANALYZER_AVAILABLE:
        return jsonify({"error": "Enhanced analyzer not available"}), 503
    
    data = request.json
    original_prompt = data.get('prompt', '').strip()
    generated_code = data.get('code', '').strip()
    language = data.get('language', 'python').lower()
    
    if not original_prompt or not generated_code:
        return jsonify({"error": "Both prompt and code are required"}), 400
    
    try:
        analyzer = get_analyzer()
        result = analyzer.analyze_llm_generated_code(original_prompt, generated_code, language)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"LLM code analysis failed: {str(e)}"}), 500


@app.route('/api/analyze-repository', methods=['POST'])
def api_analyze_repository():
    """Analyze entire GitHub repository"""
    if not CODE_ANALYZER_AVAILABLE:
        return jsonify({"error": "Enhanced analyzer not available"}), 503
    
    data = request.json
    repo_url = data.get('repo_url', '').strip()
    
    if not repo_url:
        return jsonify({"error": "Repository URL is required"}), 400
    
    if not repo_url.startswith('https://github.com/'):
        return jsonify({"error": "Only GitHub repositories are supported"}), 400
    
    try:
        analyzer = get_analyzer()
        result = analyzer.analyze_repository(repo_url)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Repository analysis failed: {str(e)}"}), 500


@app.route('/api/analyze-git-diff', methods=['POST'])
def api_analyze_git_diff():
    """Analyze git diff for changelog"""
    if not CODE_ANALYZER_AVAILABLE:
        return jsonify({"error": "Enhanced analyzer not available"}), 503
    
    data = request.json
    repo_path = data.get('repo_path', '').strip()
    
    if not repo_path:
        return jsonify({"error": "Repository path is required"}), 400
    
    try:
        analyzer = get_analyzer()
        result = analyzer.analyze_git_diff(repo_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Git diff analysis failed: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    print("🚀 Starting Flask server at http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)
