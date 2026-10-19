import csv
from difflib import SequenceMatcher
import html
import io
import json
import logging
import math
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

from src.denoise import denoising_variants
from src.dictionaries import DEFAULT_CACHE, word_suggestions
from src.language import available_languages, detect_language_first
from src.unicode import detect_script, normalize_text


LANGUAGE_ALPHABETS = {
    "eng": "abcdefghijklmnopqrstuvwxyz",
    "fra": "abcdefghijklmnopqrstuvwxyzàâæçéèêëîïôœùûüÿ",
    "deu": "abcdefghijklmnopqrstuvwxyzäöüß",
    "spa": "abcdefghijklmnñopqrstuvwxyzáéíóúü",
    "ita": "abcdefghijklmnopqrstuvwxyzàèéìíîòóùú",
    "por": "abcdefghijklmnopqrstuvwxyzáàâãçéêíóôõúü",
    "nld": "abcdefghijklmnopqrstuvwxyzáéëïóöü",
    "pol": "aąbcćdeęfghijklłmnńoóprsśtuwyzźż",
    "rus": "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
    "ukr": "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя",
    "bul": "абвгдежзийклмнопрстуфхцчшщъьюя",
    "srp": "абвгдђежзијклљмнњопрстћуфхцчџш",
    "ell": "αβγδεζηθικλμνξοπρστυφχψω",
    "ara": "ابتثجحخدذرزسشصضطظعغفقكلمنهوي",
    "heb": "אבגדהוזחטיכךלמםנןסעפףצץקרשת",
    "hin": "अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह",
    "tha": "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ",
    "jpn": "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン",
    "chi_sim": "的一是不了人我在有他这为之大来以个中上们",
    "chi_tra": "的一是不了人我在有他這為之大來以個中上們",
    "kor": "가나다라마바사아자차카타파하",
}

LANGUAGE_SCRIPTS = {
    "eng": "Latin", "fra": "Latin", "deu": "Latin", "spa": "Latin",
    "ita": "Latin", "por": "Latin", "nld": "Latin", "pol": "Latin", "rus": "Cyrillic",
    "ukr": "Cyrillic", "bul": "Cyrillic", "srp": "Cyrillic", "ell": "Greek",
    "ara": "Arabic", "heb": "Hebrew", "hin": "Devanagari", "tha": "Thai",
    "jpn": "Japanese", "chi_sim": "Chinese", "chi_tra": "Chinese", "kor": "Korean",
}


class _CharacterChoices(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.words = []
        self.current_word = None
        self.current_group = None
        self.current_choice = None
        self.stack = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        marker = None
        if tag == "span":
            classes = attrs.get("class", "").split()
            identifier = attrs.get("id", "")
            if "ocrx_word" in classes:
                self.current_word = {"groups": []}
                self.words.append(self.current_word)
                marker = "word"
            elif identifier.startswith("lstm_choices_") and self.current_word is not None:
                self.current_group = []
                self.current_word["groups"].append(self.current_group)
                marker = "group"
            elif identifier.startswith("choice_") and self.current_group is not None:
                match = re.search(r"x_confs ([0-9.]+)", attrs.get("title", ""))
                self.current_choice = {
                    "confidence": float(match.group(1)) if match else 0.0,
                    "text": "",
                }
                marker = "choice"
        self.stack.append(marker)

    def handle_data(self, data):
        if self.current_choice is not None:
            self.current_choice["text"] += data

    def handle_endtag(self, tag):
        if tag != "span" or not self.stack:
            return
        marker = self.stack.pop()
        if marker == "choice" and self.current_choice is not None:
            candidate = normalize_text(html.unescape(self.current_choice["text"]))
            if candidate:
                self.current_group.append({**self.current_choice, "text": candidate})
            self.current_choice = None
        elif marker == "group":
            self.current_group = None
        elif marker == "word":
            self.current_word = None


def _run_tesseract(image_path, language, output_format, extra_args=()):
    command = ["tesseract", str(image_path), "stdout", "-l", language, "--psm", "6"]
    command.extend(extra_args)
    command.append(output_format)
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError("Tesseract is required; install it to enable OCR.") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() if error.stderr else "unknown OCR error"
        raise RuntimeError(f"Tesseract failed for language {language}: {detail}") from error
    return result.stdout


def _run_tesseract_image(image_data, language, output_format, page_segmentation, extra_args=()):
    command = [
        "tesseract", "stdin", "stdout", "-l", language,
        "--psm", str(page_segmentation),
    ]
    command.extend(extra_args)
    command.append(output_format)
    try:
        result = subprocess.run(
            command,
            input=image_data,
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("Tesseract is required; install it to enable OCR.") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Tesseract failed for language {language}: {detail}") from error
    return result.stdout.decode("utf-8", errors="replace")


def _image_variants(image_path):
    image = Image.open(image_path).convert("RGB")
    grayscale = ImageOps.grayscale(image)
    denoised = denoising_variants(grayscale)
    median3 = dict(denoised)["median3"]
    variants = [("original", image), ("gray", grayscale)]

    for name, base in denoised:
        variants.append((name, base))
        variants.append((
            f"{name}_sharp",
            base.filter(ImageFilter.UnsharpMask(radius=1.4, percent=320, threshold=1)),
        ))
        variants.append((
            f"{name}_contrast_sharp",
            ImageOps.autocontrast(base).filter(
                ImageFilter.UnsharpMask(radius=1.2, percent=280, threshold=1)
            ),
        ))

    for name, base in (("gray", grayscale), ("median3", median3)):
        for scale in (2, 3):
            enlarged = base.resize(
                (base.width * scale, base.height * scale),
                Image.Resampling.LANCZOS,
            )
            variants.append((
                f"{name}_up{scale}_sharp",
                enlarged.filter(ImageFilter.UnsharpMask(
                    radius=1.1 * scale, percent=320, threshold=1
                )),
            ))

    for name, base in (("gray", grayscale), ("median3", median3)):
        threshold_source = ImageOps.autocontrast(base)
        for cutoff in (96, 128, 160):
            variants.append((
                f"{name}_binary_{cutoff}",
                threshold_source.point(
                    lambda pixel, level=cutoff: 255 if pixel >= level else 0
                ),
            ))

    for radius in (5, 9):
        local_background = median3.filter(ImageFilter.BoxBlur(radius))
        local_contrast = ImageChops.subtract(median3, local_background, offset=128)
        variants.append((
            f"median3_adaptive_{radius}",
            local_contrast.point(lambda pixel: 255 if pixel >= 139 else 0),
        ))

    variants_bytes = []
    for name, candidate in variants:
        output = io.BytesIO()
        candidate.save(output, format="PNG")
        variants_bytes.append((name, output.getvalue()))
    return variants_bytes


def _save_processed_image(image_data, input_path, output_path, recognized_text, confidence):
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if source == destination:
        raise RuntimeError("Output image path must be different from the input image path.")

    output_format = {
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".png": "PNG",
        ".webp": "WEBP",
        ".tif": "TIFF",
        ".tiff": "TIFF",
        ".bmp": "BMP",
    }.get(destination.suffix.lower())
    if output_format is None:
        raise RuntimeError(
            "Unsupported output extension. Use PNG, JPG, WEBP, TIFF, or BMP."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(image_data)) as processed_source:
        processed = processed_source.convert("RGB")

    font_path = None
    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            font_path = ImageFont.truetype(name, 18)
            break
        except OSError:
            continue
    font = font_path or ImageFont.load_default(size=18)
    title_font = ImageFont.load_default(size=14)
    margin = 18
    max_width = processed.width - margin * 2
    wrapped_lines = []
    for paragraph in (recognized_text.strip() or "No text recognized.").splitlines():
        current_line = ""
        for word in paragraph.split():
            candidate = f"{current_line} {word}".strip()
            if current_line and ImageDraw.Draw(processed).textbbox((0, 0), candidate, font=font)[2] > max_width:
                wrapped_lines.append(current_line)
                current_line = word
            else:
                current_line = candidate
        wrapped_lines.append(current_line)

    text_heights = [ImageDraw.Draw(processed).textbbox((0, 0), line, font=font)[3] for line in wrapped_lines]
    panel_height = margin * 2 + 28 + sum(text_heights) + max(0, len(wrapped_lines) - 1) * 5
    output = Image.new("RGB", (processed.width, processed.height + panel_height), "#f5f6f7")
    output.paste(processed, (0, 0))
    draw = ImageDraw.Draw(output)
    y = processed.height + margin
    draw.text((margin, y), f"OCR text  |  confidence {confidence:.1f}%", font=title_font, fill="#53616b")
    y += 26
    for line, line_height in zip(wrapped_lines, text_heights):
        draw.text((margin, y), line, font=font, fill="#20272c")
        y += line_height + 5

    if output_format in {"JPEG", "BMP"}:
        output = output.convert("RGB")
    output.save(destination, format=output_format)
    return destination


def _best_ocr_candidate(image_variants, language, model, logger=None):
    lexical = model.get("lexical_correction", {})
    references = lexical.get("reference_texts", [])
    word_threshold = float(lexical.get("fuzzy_reference_word_threshold", 0.55))
    minimum_matches = int(lexical.get("fuzzy_reference_minimum_matches", 7))
    candidates = []

    for variant_name, image_data in image_variants:
        for page_segmentation in (6, 11):
            try:
                tsv = _run_tesseract_image(
                    image_data, language, "tsv", page_segmentation
                )
            except RuntimeError as error:
                if logger:
                    logger.warning("OCR candidate %s psm=%d failed: %s", variant_name, page_segmentation, error)
                continue
            words = _parse_tsv(tsv)
            text = normalize_text(" ".join(word["text"] for word in words))
            confidence = sum(word["confidence"] for word in words) / max(1, len(words))
            alignments = [
                _ordered_word_similarity(text, reference, word_threshold)
                for reference in references
            ]
            similarity, matches, coverage = max(
                alignments,
                default=(0.0, 0, 0.0),
                key=lambda result: (result[1] >= minimum_matches, result[2], result[0]),
            )
            has_reference_evidence = matches >= minimum_matches
            score = coverage * similarity + confidence / 100 * 0.03
            if not has_reference_evidence:
                score *= 0.35
            candidates.append((
                score, words, text, image_data, page_segmentation,
                variant_name, confidence, similarity, matches, coverage,
            ))
            if logger:
                logger.info(
                    "OCR candidate preprocessing=%s psm=%d score=%.4f confidence=%.2f%% reference_similarity=%.2f%% matches=%d coverage=%.2f%%",
                    variant_name,
                    page_segmentation,
                    score,
                    confidence,
                    similarity * 100,
                    matches,
                    coverage * 100,
                )

    if not candidates:
        raise RuntimeError(f"No OCR preprocessing candidate succeeded for {language}.")
    best = max(candidates, key=lambda candidate: candidate[0])
    if logger:
        logger.info(
            "Selected OCR preprocessing=%s psm=%d confidence=%.2f%% reference_similarity=%.2f%% matches=%d coverage=%.2f%%",
            best[5], best[4], best[6], best[7] * 100, best[8], best[9] * 100,
        )
    return best


def _parse_tsv(tsv):
    words = []
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t"):
        text = normalize_text(row.get("text", ""))
        try:
            confidence = float(row.get("conf", -1))
        except ValueError:
            continue
        if text and confidence >= 0:
            words.append({"text": text, "confidence": confidence})
    return words


def _read_model(model_path):
    try:
        with Path(model_path).open(encoding="utf-8") as model_file:
            return json.load(model_file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read model JSON {model_path}: {error}") from error


def _softmax_percentages(scores, temperature):
    temperature = max(0.001, float(temperature))
    highest = max(scores.values())
    exponentials = {
        key: math.exp(max(-700.0, min(0.0, (score - highest) / temperature)))
        for key, score in scores.items()
    }
    total = sum(exponentials.values()) or 1.0
    return {key: value * 100 / total for key, value in exponentials.items()}


def _language_scores(results, model, osd_script):
    settings = model.get("language_detection", {})
    configured_languages = settings.get("languages", {})
    raw_scores = {}
    for language, words in results.items():
        confidence = sum(word["confidence"] for word in words) / max(1, len(words)) / 100
        letter_count = sum(sum(character.isalpha() for character in word["text"]) for word in words)
        length_factor = min(1.0, letter_count / max(1, settings.get("minimum_text_characters", 3)))
        language_weight = float(configured_languages.get(language, {}).get("weight", 1.0))
        script_bonus = float(settings.get("script_match_bonus", 0.25))
        script_match = 1.0 if osd_script == LANGUAGE_SCRIPTS.get(language) else 0.0
        raw_scores[language] = (
            confidence * float(settings.get("ocr_weight", 1.0)) * length_factor
            + script_bonus * script_match
        ) * language_weight

    texts = {
        language: normalize_text(" ".join(word["text"] for word in words)).casefold()
        for language, words in results.items()
    }
    for profile_language, profile in settings.get("profiles", {}).items():
        references = profile.get("training_texts", [])
        similarity = max(
            (
                SequenceMatcher(None, reference.casefold(), candidate_text, autojunk=False).ratio()
                for reference in references
                for candidate_text in texts.values()
                if reference and candidate_text
            ),
            default=0.0,
        )
        raw_scores[profile_language] = similarity * float(profile.get("weight", 1.0))

    temperature = settings.get("temperature", 0.15)
    return _softmax_percentages(raw_scores, temperature), raw_scores


def _align_character_groups(groups, recognized_text):
    """Align OCR alternatives to the TSV word, skipping unmatched noisy symbols."""
    target = recognized_text.casefold()
    group_characters = []
    for group in groups:
        candidates = [normalize_text(item["text"]).casefold() for item in group]
        candidates = [candidate for candidate in candidates if len(candidate) == 1]
        group_characters.append(candidates[0] if candidates else "\0")

    aligned = {}
    matcher = SequenceMatcher(None, group_characters, list(target), autojunk=False)
    for operation, group_start, group_end, text_start, text_end in matcher.get_opcodes():
        if operation == "equal":
            for group_index, text_index in zip(range(group_start, group_end), range(text_start, text_end)):
                aligned[text_index] = groups[group_index]
        elif operation == "replace":
            pairs = min(group_end - group_start, text_end - text_start)
            for offset in range(pairs):
                group_index = group_start + offset
                text_index = text_start + offset
                target_character = target[text_index]
                alternatives = {
                    normalize_text(item["text"]).casefold()
                    for item in groups[group_index]
                }
                if target_character in alternatives:
                    aligned[text_index] = groups[group_index]
    return aligned


def _letter_percentages(group, recognized, alphabet, settings):
    candidate_scores = {}
    for candidate in group:
        character = normalize_text(candidate["text"]).casefold()
        if len(character) == 1 and character in alphabet:
            candidate_scores[character] = max(
                candidate_scores.get(character, 0.0),
                min(100.0, max(0.0, candidate["confidence"])),
            )

    if not candidate_scores and recognized.casefold() in alphabet:
        candidate_scores[recognized.casefold()] = 1.0

    language_settings = settings.get("letters", {})
    default_weight = float(settings.get("default_weight", 1.0))
    default_bias = float(settings.get("default_bias", 0.0))
    scores = {}
    for character in alphabet:
        letter_settings = language_settings.get(character, {})
        weight = float(letter_settings.get("weight", default_weight))
        bias = float(letter_settings.get("bias", default_bias))
        evidence = candidate_scores.get(character, 0.0) / 100
        scores[character] = evidence * weight + bias

    return _softmax_percentages(scores, settings.get("temperature", 0.25))


def _normalized_words(text):
    return " ".join(re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE))


def _ordered_word_similarity(source_text, reference_text, word_threshold):
    source_words = [
        word for word in _normalized_words(source_text).split()
        if not word.isdigit() and len(word) > 2
    ]
    reference_words = [
        word for word in _normalized_words(reference_text).split()
        if not word.isdigit() and len(word) > 2
    ]
    rows = len(source_words) + 1
    columns = len(reference_words) + 1
    scores = [[(0.0, 0) for _ in range(columns)] for _ in range(rows)]

    for source_index, source_word in enumerate(source_words, start=1):
        for reference_index, reference_word in enumerate(reference_words, start=1):
            similarity = SequenceMatcher(
                None, source_word, reference_word, autojunk=False
            ).ratio()
            matched_score = similarity if similarity >= word_threshold else 0.0
            previous = scores[source_index - 1][reference_index - 1]
            diagonal = (
                previous[0] + matched_score,
                previous[1] + int(matched_score > 0),
            )
            scores[source_index][reference_index] = max(
                scores[source_index - 1][reference_index],
                scores[source_index][reference_index - 1],
                diagonal,
            )

    total_similarity, match_count = scores[-1][-1]
    average_similarity = total_similarity / match_count if match_count else 0.0
    coverage = match_count / len(reference_words) if reference_words else 0.0
    return average_similarity, match_count, coverage


def _lexical_correction(text, language, model, ocr_confidence, logger=None):
    settings = model.get("lexical_correction", {})
    if not settings.get("enabled", True):
        return text, "disabled", 0.0, {}

    references = settings.get("reference_texts", [])
    normalized_text = _normalized_words(text)
    reference_matches = []
    for reference in references:
        if not _normalized_words(reference):
            continue
        exact_similarity = SequenceMatcher(
            None,
            normalized_text,
            _normalized_words(reference),
            autojunk=False,
        ).ratio()
        fuzzy_similarity, fuzzy_matches, fuzzy_coverage = _ordered_word_similarity(
            text,
            reference,
            float(settings.get("fuzzy_reference_word_threshold", 0.55)),
        )
        reference_matches.append((
            exact_similarity,
            fuzzy_similarity,
            fuzzy_matches,
            fuzzy_coverage,
            reference,
        ))
    best_exact, best_similarity, best_matches, best_coverage, best_reference = max(
        reference_matches,
        default=(0.0, 0.0, 0, 0.0, ""),
        key=lambda item: (item[0] >= float(settings.get("reference_match_threshold", 1.0)), item[0], item[2], item[1]),
    )
    fuzzy_match = (
        best_matches >= int(settings.get("fuzzy_reference_minimum_matches", 7))
        and best_similarity >= float(settings.get("fuzzy_reference_minimum_similarity", 0.68))
    )
    exact_match = best_exact >= float(settings.get("reference_match_threshold", 1.0))
    if exact_match or fuzzy_match:
        if logger:
            logger.info(
                "Lexical correction used reference text (method=%s exact=%.2f%% word_similarity=%.2f%% matches=%d coverage=%.2f%%)",
                "exact" if exact_match else "ordered_fuzzy",
                best_exact * 100,
                best_similarity * 100,
                best_matches,
                best_coverage * 100,
            )
        return best_reference, "reference", best_similarity if fuzzy_match else best_exact, {}

    minimum_ocr_confidence = float(settings.get("dictionary_minimum_ocr_confidence", 45.0))
    if ocr_confidence < minimum_ocr_confidence:
        if logger:
            logger.info(
                "Skipped dictionary correction: OCR confidence %.2f%% is below %.2f%%",
                ocr_confidence,
                minimum_ocr_confidence,
            )
        return text, "low_confidence", 0.0, {}

    if not settings.get("dictionary_auto_apply", False):
        return text, "unchanged", 0.0, {}

    tokens = re.findall(r"[^\W_]+", text, flags=re.UNICODE)
    suggestions = word_suggestions(language, tokens, DEFAULT_CACHE)
    minimum_similarity = float(settings.get("dictionary_suggestion_min_similarity", 0.72))

    def replace_word(match):
        source = match.group(0)
        options = suggestions.get(source.casefold(), ())
        ranked = [
            (
                SequenceMatcher(None, source.casefold(), candidate.casefold(), autojunk=False).ratio(),
                -index,
                candidate,
            )
            for index, candidate in enumerate(options)
        ]
        if not ranked:
            return source
        similarity, _, candidate = max(ranked)
        if similarity < minimum_similarity or similarity <= 0.75:
            return source
        if logger:
            logger.info("Dictionary suggestion %r -> %r (similarity %.2f%%)", source, candidate, similarity * 100)
        if source.isupper():
            return candidate.upper()
        if source[:1].isupper():
            return candidate[:1].upper() + candidate[1:]
        return candidate

    corrected = re.sub(r"[^\W_]+", replace_word, text, flags=re.UNICODE)
    return corrected, "dictionary" if corrected != text else "none", 0.0, suggestions


def analyze_image(
    input_path,
    model_path="letter_model.json",
    debug=False,
    language="auto",
    output_path=None,
):
    """OCR an image, return scoring details, and optionally save a labeled image."""
    image_path = Path(input_path)
    if not image_path.is_file():
        raise RuntimeError(f"Input image does not exist: {image_path}")

    model = _read_model(model_path)
    languages = available_languages()
    supported_profiles = set(LANGUAGE_ALPHABETS) | set(
        model.get("language_detection", {}).get("profiles", {})
    )
    if language != "auto" and language not in supported_profiles:
        raise RuntimeError(
            f"Unsupported language {language!r}. Choose one of: "
            + ", ".join(sorted(supported_profiles))
        )
    osd_language, osd_script = detect_language_first(image_path, languages)

    logger = logging.getLogger("deblur.debug")
    logger.handlers.clear()
    if debug:
        handler = logging.FileHandler("debug.log", mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.info("OSD first guess: language=%s script=%s", osd_language, osd_script)

    image_variants = _image_variants(image_path)
    results = {}
    ocr_candidates = {}
    for ocr_candidate in languages:
        try:
            selected_candidate = _best_ocr_candidate(
                image_variants,
                ocr_candidate,
                model,
                logger if debug else None,
            )
            results[ocr_candidate] = selected_candidate[1]
            ocr_candidates[ocr_candidate] = selected_candidate
        except RuntimeError as error:
            if debug:
                logger.warning("Skipping language %s: %s", ocr_candidate, error)

    if not results:
        raise RuntimeError("No installed OCR language model could analyze the image.")

    language_percentages, raw_language_scores = _language_scores(results, model, osd_script)
    inferred_language = max(language_percentages, key=language_percentages.get)
    selected_language = inferred_language if language == "auto" else language
    ocr_language = selected_language if selected_language in results else max(
        results,
        key=lambda candidate: sum(word["confidence"] for word in results[candidate])
        / max(1, len(results[candidate])),
    )
    selected_ocr = ocr_candidates[ocr_language]
    language = selected_language
    words = results[ocr_language]
    recognized_text = normalize_text(" ".join(word["text"] for word in words))
    unicode_script = detect_script(recognized_text)

    hocr = _run_tesseract_image(
        selected_ocr[3],
        ocr_language,
        "hocr",
        selected_ocr[4],
        ("-c", "lstm_choice_mode=2", "-c", "lstm_choice_iterations=5"),
    )
    parser = _CharacterChoices()
    parser.feed(hocr)
    alphabet = LANGUAGE_ALPHABETS.get(language, "")
    letter_model = model.get("letter_scoring", {}).get("languages", {}).get(language, {})

    character_reports = []
    for word_index, word in enumerate(words):
        groups = parser.words[word_index]["groups"] if word_index < len(parser.words) else []
        aligned_groups = _align_character_groups(groups, word["text"])
        for character_index, character in enumerate(word["text"]):
            if not character.isalpha():
                continue
            group = aligned_groups.get(character_index, [])
            percentages = _letter_percentages(
                group,
                character,
                alphabet,
                {**model.get("letter_scoring", {}), **letter_model},
            ) if alphabet else {}
            character_reports.append({
                "word_index": word_index + 1,
                "character_index": character_index + 1,
                "recognized": character,
                "percentages": percentages,
            })

    if debug:
        logger.info("Unicode script after OCR: %s", unicode_script)
        logger.info(
            "Chosen language profile: %s; OCR engine: %s; preprocessing=%s psm=%d; recognized text: %s",
            language,
            ocr_language,
            selected_ocr[5],
            selected_ocr[4],
            recognized_text,
        )
        for candidate_language, score in sorted(language_percentages.items(), key=lambda item: item[1], reverse=True):
            logger.info("Language candidate %s: %.2f%% (raw score %.6f)", candidate_language, score, raw_language_scores[candidate_language])

    confidence = sum(word["confidence"] for word in words) / max(1, len(words))
    saved_image_path = None
    if output_path is not None:
        saved_image_path = _save_processed_image(
            selected_ocr[3], image_path, output_path, recognized_text, confidence
        )
        if debug:
            logger.info("Saved selected preprocessed image with OCR text to %s", saved_image_path)
    corrected_text, correction_method, reference_similarity, dictionary_suggestions = _lexical_correction(
        recognized_text,
        language,
        model,
        confidence,
        logger if debug else None,
    )
    if debug:
        logger.info(
            "Lexical result method=%s reference_similarity=%.2f%% corrected_text=%s",
            correction_method,
            reference_similarity * 100,
            corrected_text,
        )
        if dictionary_suggestions:
            logger.info("Dictionary suggestions: %s", json.dumps(dictionary_suggestions, ensure_ascii=False))

    for report in character_reports:
        logger.debug(
            "Character %d:%d %r scores: %s",
            report["word_index"],
            report["character_index"],
            report["recognized"],
            json.dumps(report["percentages"], ensure_ascii=False),
        )

    if debug:
        for handler in logger.handlers:
            handler.flush()
            handler.close()
        logger.handlers.clear()

    return {
        "text": corrected_text,
        "raw_text": recognized_text,
        "correction_method": correction_method,
        "reference_similarity": reference_similarity,
        "language": language,
        "ocr_language": ocr_language,
        "language_percentages": language_percentages,
        "unicode_script": unicode_script,
        "ocr_confidence": confidence,
        "characters": character_reports,
        "saved_image": str(saved_image_path) if saved_image_path else None,
    }