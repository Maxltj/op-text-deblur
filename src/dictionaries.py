import logging
import subprocess
from functools import lru_cache
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


DICTIONARY_SOURCES = {
    "eng": ("hunspell", "en/en_US"),
    "fra": ("hunspell", "fr_FR/dictionaries/fr"),
    "deu": ("hunspell", "de/de_DE_frami"),
    "spa": ("hunspell", "es/es_ES"),
    "ita": ("hunspell", "it_IT/it_IT"),
    "por": ("hunspell", "pt_PT/pt_PT"),
    "nld": ("hunspell", "nl_NL/nl_NL"),
    "pol": ("hunspell", "pl_PL/pl_PL"),
    "rus": ("hunspell", "ru_RU/ru_RU"),
    "ukr": ("hunspell", "uk_UA/uk_UA"),
    "bul": ("hunspell", "bg_BG/bg_BG"),
    "srp": ("hunspell", "sr/sr"),
    "ell": ("hunspell", "el_GR/el_GR"),
    "ara": ("hunspell", "ar/ar"),
    "heb": ("hunspell", "he_IL/he_IL"),
    "hin": ("hunspell", "hi_IN/hi_IN"),
    "tha": ("hunspell", "th_TH/th_TH"),
    "jpn": ("frequency", "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/ja/ja_full.txt"),
    "chi_sim": ("jieba", "https://raw.githubusercontent.com/fxsjy/jieba/master/jieba/dict.txt"),
    "chi_tra": ("jieba", "https://raw.githubusercontent.com/fxsjy/jieba/master/jieba/dict.txt"),
    "kor": ("hunspell", "ko_KR/ko_KR"),
}

LIBREOFFICE_RAW = "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/{}"
DEFAULT_CACHE = Path.home() / ".cache" / "deblur" / "dictionaries"


def _download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as response, target.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def download_dictionaries(languages=None, cache_dir=DEFAULT_CACHE):
    """Download supported Hunspell dictionaries and word lists into the local cache."""
    selected = tuple(languages or DICTIONARY_SOURCES)
    cache_dir = Path(cache_dir)
    downloaded = []
    for language in selected:
        if language not in DICTIONARY_SOURCES:
            logging.warning("No dictionary source is configured for %s", language)
            continue
        kind, source = DICTIONARY_SOURCES[language]
        folder = cache_dir / language
        if kind == "hunspell":
            base = source.rsplit("/", 1)[-1]
            for extension in ("aff", "dic"):
                target = folder / f"{base}.{extension}"
                if not target.is_file():
                    try:
                        _download(LIBREOFFICE_RAW.format(f"{source}.{extension}"), target)
                    except (OSError, URLError) as error:
                        logging.warning("Could not download %s dictionary: %s", language, error)
                        break
            else:
                downloaded.append(language)
        else:
            target = folder / "words.txt"
            if not target.is_file():
                try:
                    _download(source, target)
                except (OSError, URLError) as error:
                    logging.warning("Could not download %s word list: %s", language, error)
                    continue
            downloaded.append(language)
    return downloaded


def _hunspell_suggestions(language, tokens, cache_dir):
    folder = Path(cache_dir) / language
    dictionaries = tuple(folder.glob("*.dic"))
    if not dictionaries:
        return {}
    dictionary_base = dictionaries[0].with_suffix("")
    try:
        result = subprocess.run(
            ["hunspell", "-a", "-d", str(dictionary_base)],
            input="\n".join(tokens) + "\n",
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}

    suggestions = {}
    token_index = 0
    for line in result.stdout.splitlines():
        if not line or line.startswith("@(#)"):
            continue
        if line.startswith(("*", "+", "-")):
            token_index += 1
            continue
        if line.startswith("& ") or line.startswith("? "):
            try:
                payload = line.split(":", 1)[1].strip()
                suggestions[tokens[token_index].casefold()] = [
                    option.strip() for option in payload.split(",") if option.strip()
                ]
            except (IndexError, ValueError):
                pass
            token_index += 1
        elif line.startswith("# "):
            token_index += 1
    return suggestions


def word_suggestions(language, tokens, cache_dir=DEFAULT_CACHE):
    """Get spelling or nearby-word candidates from the cached language resource."""
    kind = DICTIONARY_SOURCES.get(language, (None,))[0]
    tokens = list(dict.fromkeys(token for token in tokens if token))
    if kind == "hunspell":
        return _hunspell_suggestions(language, tokens, cache_dir)
    if kind not in {"frequency", "jieba"}:
        return {}

    words = _word_list_words(language, str(Path(cache_dir).resolve()))
    suggestions = {}
    for token in tokens:
        folded = token.casefold()
        if folded in words:
            continue
        initial = folded[:1]
        candidates = words.get(initial, ())
        ranked = sorted(
            (
                _similarity(folded, candidate),
                candidate,
            )
            for candidate in candidates
            if abs(len(candidate) - len(folded)) <= 2
        )
        options = [candidate for similarity, candidate in ranked[-8:] if similarity >= 0.45]
        if options:
            suggestions[folded] = list(reversed(options))
    return suggestions


def _similarity(first, second):
    from difflib import SequenceMatcher

    return SequenceMatcher(None, first, second, autojunk=False).ratio()


@lru_cache(maxsize=8)
def _word_list_words(language, cache_path):
    word_list = Path(cache_path) / language / "words.txt"
    by_initial = {}
    try:
        with word_list.open(encoding="utf-8", errors="ignore") as source:
            for line in source:
                fields = line.split()
                if not fields:
                    continue
                word = fields[0].casefold()
                if word:
                    by_initial.setdefault(word[:1], set()).add(word)
    except OSError:
        return {}
    return {initial: tuple(words) for initial, words in by_initial.items()}


def word_list_size(language, cache_dir=DEFAULT_CACHE):
    """Return the local dictionary entry count, if a downloaded word list exists."""
    folder = Path(cache_dir) / language
    dictionaries = tuple(folder.glob("*.dic"))
    word_list = folder / "words.txt"
    if dictionaries:
        try:
            with dictionaries[0].open(encoding="utf-8", errors="ignore") as source:
                return max(0, sum(1 for _ in source) - 1)
        except OSError:
            return 0
    if word_list.is_file():
        try:
            with word_list.open(encoding="utf-8", errors="ignore") as source:
                return sum(1 for line in source if line.strip())
        except OSError:
            return 0
    return 0