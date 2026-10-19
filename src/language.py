import subprocess
import re

from src.unicode import detect_script


SCRIPT_LANGUAGES = {
    "Latin": ("eng", "fra", "deu", "spa", "ita", "por", "nld"),
    "Cyrillic": ("rus", "ukr", "bul", "srp"),
    "Greek": ("ell",),
    "Arabic": ("ara",),
    "Hebrew": ("heb",),
    "Devanagari": ("hin",),
    "Thai": ("tha",),
    "Japanese": ("jpn",),
    "Chinese": ("chi_sim", "chi_tra"),
    "Korean": ("kor",),
}

OSD_SCRIPTS = {
    "Latin": ("eng", "fra", "deu", "spa", "ita", "por", "nld"),
    "Cyrillic": ("rus", "ukr", "bul", "srp"),
    "Greek": ("ell",),
    "Arabic": ("ara",),
    "Hebrew": ("heb",),
    "Devanagari": ("hin",),
    "Thai": ("tha",),
    "Japanese": ("jpn",),
    "Han": ("chi_sim", "chi_tra", "jpn"),
    "Hangul": ("kor",),
}


def available_languages():
    """Return language models installed for the Tesseract executable."""
    try:
        result = subprocess.run(
            ["tesseract", "--list-langs"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("Tesseract is required; install it to enable OCR.") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Could not list installed Tesseract language models.") from error

    return tuple(
        language
        for line in result.stdout.splitlines()[1:]
        if (language := line.strip()) and language != "osd"
    )


def detect_language(text, languages):
    """Choose an installed OCR model using the recognized text's Unicode script."""
    available = set(languages)
    script = detect_script(text)
    for language in SCRIPT_LANGUAGES.get(script, ()):
        if language in available:
            return language

    if "eng" in available:
        return "eng"
    if languages:
        return languages[0]
    raise RuntimeError("No Tesseract language models are installed.")


def detect_language_first(image_path, languages):
    """Use Tesseract orientation/script detection before Unicode OCR."""
    try:
        result = subprocess.run(
            ["tesseract", str(image_path), "stdout", "--psm", "0"],
            check=True,
            capture_output=True,
            text=True,
        )
        match = re.search(r"^Script:\s*(.+?)\s*$", result.stdout, re.MULTILINE)
        if match:
            for language in OSD_SCRIPTS.get(match.group(1), ()):
                if language in languages:
                    return language, match.group(1)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    if "eng" in languages:
        return "eng", "unknown (OSD unavailable; using English fallback)"
    if languages:
        return languages[0], "unknown (OSD unavailable; using installed fallback)"
    raise RuntimeError("No Tesseract language models are installed.")