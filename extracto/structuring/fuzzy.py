"""Fuzzy text matching utilities for OCR-tolerant anchor finding.

OCR garbles common medical/legal labels in predictable ways:
- "Fatigue" → "Gighttatique" (visual similarity errors)
- "Patient" → "Patlent" (l/i confusion)
- "Female" → "Femaie" (l/i confusion)
- "Diabetic" → "Dlabetic"

This module provides multi-strategy fuzzy matching:
1. Exact substring match (fastest path)
2. Whitespace-normalized match
3. Levenshtein-distance ratio match
4. Token-overlap match

Plus an OCR vocabulary corrector that snaps garbled words to known terms.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher


# Common medical/legal vocabulary that OCR often garbles
MEDICAL_VOCAB = {
    # Body parts / symptoms
    "fatigue", "headache", "nausea", "dizziness", "fever", "cough",
    "vomiting", "diarrhea", "constipation", "shortness of breath",
    "chest pain", "back pain", "abdominal pain",
    # Conditions
    "diabetic", "smoker", "pregnant", "hypertension", "asthma",
    # Allergies
    "penicillin", "peanuts", "latex", "shellfish", "pollen", "dust",
    "none known",
    # Patient info labels
    "patient", "patient's", "name", "address", "phone", "email",
    "date of birth", "date of injury", "date of service", "date of accident",
    "social security", "occupation", "employer", "physician",
    # Sex/gender
    "male", "female", "other", "sex", "gender",
    # Yes/no
    "yes", "no", "true", "false",
    # Insurance / claims
    "claim", "policy", "member", "subscriber", "group", "deductible",
    "copay", "coinsurance", "allowed", "billed", "paid",
    # Form headings
    "patient intake form", "insurance claim form",
    "explanation of benefits", "first report of injury",
    "patient health questionnaire", "phq-9",
    "authorization for release",
    # Body parts for FROI
    "shoulder", "elbow", "wrist", "hand", "fingers", "knee", "ankle",
    "foot", "hip", "thigh", "neck", "back", "head", "eye", "ear",
}


def normalize(s: str) -> str:
    """Lowercase + collapse whitespace + strip."""
    return re.sub(r"\s+", " ", s.lower()).strip()


def normalize_aggressive(s: str) -> str:
    """Strip all non-alphanumeric — for very fuzzy matching when OCR adds junk."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def edit_ratio(a: str, b: str) -> float:
    """SequenceMatcher ratio (1.0 = identical, 0.0 = no overlap)."""
    return SequenceMatcher(None, a, b).ratio()


def token_overlap(a: str, b: str) -> float:
    """Jaccard similarity of word tokens."""
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def fuzzy_contains(text: str, needle: str, threshold: float = 0.75) -> bool:
    """Check if text 'contains' needle even with OCR errors.

    Tries multiple strategies in order of strictness:
    1. Exact substring (case + whitespace insensitive)
    2. Aggressive normalization substring (no spaces, no punct)
    3. Sliding window edit-distance match
    4. Token-overlap match for multi-word needles
    """
    text_norm = normalize(text)
    needle_norm = normalize(needle)

    # Strategy 1: exact normalized substring
    if needle_norm in text_norm:
        return True

    # Strategy 2: aggressive normalization (handles "1.MEDICARE" vs "1. medicare")
    text_agg = normalize_aggressive(text)
    needle_agg = normalize_aggressive(needle)
    if needle_agg and needle_agg in text_agg:
        return True

    # Strategy 3: sliding window edit-distance (for OCR garbling within a single word)
    needle_len = len(needle_norm)
    if needle_len < 4:
        return False  # too short for fuzzy matching to be safe
    # Slide a window of ~needle_len through text_norm and check ratio
    if len(text_norm) >= needle_len:
        best_ratio = 0.0
        for i in range(0, max(1, len(text_norm) - needle_len + 1), max(1, needle_len // 4)):
            window = text_norm[i : i + needle_len + 4]
            r = edit_ratio(window, needle_norm)
            if r > best_ratio:
                best_ratio = r
                if r >= threshold:
                    return True

    # Strategy 4: token overlap for multi-word needles
    if " " in needle_norm:
        if token_overlap(text_norm, needle_norm) >= 0.5:
            return True

    return False


def find_line_fuzzy(lines, needle: str, threshold: float = 0.78):
    """Find first line that fuzzy-matches the needle.

    Each `lines` element should be a dict with 'text' and 'bbox' keys
    OR a tuple of (text, bbox).
    """
    for ln in lines:
        if isinstance(ln, dict):
            text = ln.get("text", "")
        else:
            text = ln[0] if ln else ""
        if fuzzy_contains(text, needle, threshold=threshold):
            return ln
    return None


def correct_word(word: str, vocab: set[str] | None = None, threshold: float = 0.85) -> str:
    """Snap a possibly-garbled OCR word to its closest vocabulary match.

    Conservative: only corrects when the word is clearly mangled (not in vocab,
    no plain-english punctuation match) and a vocab entry is a high-confidence
    edit-distance match. This avoids corrupting correctly-OCR'd words that
    happen to be similar to vocab entries.
    """
    if vocab is None:
        vocab = MEDICAL_VOCAB

    word_lower = word.lower().strip(".,;:!?'\"()-_")
    if not word_lower or len(word_lower) < 5:  # don't correct short words (too risky)
        return word

    # Exact match — no correction needed
    if word_lower in vocab:
        return word

    # Find closest vocab entry by edit distance — single-word vocab entries only
    best_match = None
    best_ratio = threshold

    for vocab_word in vocab:
        if " " in vocab_word:
            continue  # skip multi-word entries (handled by correct_text)
        # Quick length filter
        if abs(len(vocab_word) - len(word_lower)) > 2:
            continue
        r = edit_ratio(word_lower, vocab_word)
        if r > best_ratio:
            best_ratio = r
            best_match = vocab_word

    if best_match:
        return best_match
    return word


def correct_text(text: str, vocab: set[str] | None = None) -> str:
    """Snap garbled OCR words in a text string to known vocabulary."""
    if vocab is None:
        vocab = MEDICAL_VOCAB

    # Try to match multi-word phrases first (greedy)
    text_lower = text.lower()
    for vocab_phrase in sorted(vocab, key=len, reverse=True):
        if " " not in vocab_phrase:
            continue
        # Look for an OCR-garbled version of this phrase in the text
        if vocab_phrase in text_lower:
            continue  # already correct
        words = text.split()
        target_words = vocab_phrase.split()
        if len(words) >= len(target_words):
            for i in range(len(words) - len(target_words) + 1):
                candidate = " ".join(words[i : i + len(target_words)]).lower()
                if edit_ratio(candidate, vocab_phrase) >= 0.85:
                    # Replace with corrected phrase
                    return " ".join(words[:i] + vocab_phrase.split() + words[i + len(target_words):])

    # Word-by-word correction
    out_words = []
    for w in text.split():
        out_words.append(correct_word(w, vocab=vocab))
    return " ".join(out_words)
