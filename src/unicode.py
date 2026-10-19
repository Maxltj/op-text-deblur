import unicodedata
from collections import Counter


def normalize_text(text):
    """Normalize recognized text to NFC and collapse whitespace."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def detect_script(text):
    """Return the most common Unicode script represented by alphabetic text."""
    scripts = Counter()
    for character in text:
        if not character.isalpha():
            continue

        name = unicodedata.name(character, "")
        if "LATIN" in name:
            scripts["Latin"] += 1
        elif "CYRILLIC" in name:
            scripts["Cyrillic"] += 1
        elif "GREEK" in name:
            scripts["Greek"] += 1
        elif "ARABIC" in name:
            scripts["Arabic"] += 1
        elif "HEBREW" in name:
            scripts["Hebrew"] += 1
        elif "DEVANAGARI" in name:
            scripts["Devanagari"] += 1
        elif "THAI" in name:
            scripts["Thai"] += 1
        elif "HIRAGANA" in name or "KATAKANA" in name:
            scripts["Japanese"] += 1
        elif "CJK" in name or "IDEOGRAPH" in name:
            scripts["Chinese"] += 1
        elif "HANGUL" in name:
            scripts["Korean"] += 1

    return scripts.most_common(1)[0][0] if scripts else None