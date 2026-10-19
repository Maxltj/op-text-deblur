import argparse
import sys

from src.deblur import LANGUAGE_ALPHABETS, analyze_image
from src.dictionaries import DEFAULT_CACHE, DICTIONARY_SOURCES, download_dictionaries


def main():
    parser = argparse.ArgumentParser(
        description="Analyze blurred text and optionally save a preprocessed image with OCR text."
    )
    parser.add_argument("input", nargs="?", help="Path to the input image")
    parser.add_argument(
        "output",
        nargs="?",
        help="Optional path for the selected denoised/sharpened image",
    )
    parser.add_argument("--model", default="letter_model.json", help="Path to the editable scoring model JSON")
    parser.add_argument(
        "--language",
        default="auto",
        choices=("auto", *sorted(LANGUAGE_ALPHABETS)),
        help="Language profile code to use (default: auto; e.g. pol, eng, fra)",
    )
    parser.add_argument("--debug", action="store_true", help="Write detailed scores to debug.log")
    parser.add_argument(
        "--download-dictionaries",
        action="store_true",
        help="Download word dictionaries for configured languages and exit",
    )
    args = parser.parse_args()

    if args.download_dictionaries:
        downloaded = download_dictionaries(cache_dir=DEFAULT_CACHE)
        print(f"Downloaded {len(downloaded)}/{len(DICTIONARY_SOURCES)} language dictionaries to {DEFAULT_CACHE}")
        missing = sorted(set(DICTIONARY_SOURCES) - set(downloaded))
        if missing:
            print("Unavailable or failed sources: " + ", ".join(missing))
        return 0 if downloaded else 1
    if not args.input:
        parser.error("input is required unless --download-dictionaries is used")

    try:
        report = analyze_image(
            args.input,
            model_path=args.model,
            debug=args.debug,
            language=args.language,
            output_path=args.output,
        )
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(
        f"Language profile ({'automatic' if args.language == 'auto' else 'selected'}): {report['language']} "
        f"(OCR engine {report['ocr_language']}, OCR confidence {report['ocr_confidence']:.1f}%)"
    )
    print("Language scores (installed OCR engines and configured text profiles):")
    for language, percentage in sorted(
        report["language_percentages"].items(), key=lambda item: item[1], reverse=True
    ):
        print(f"  {language}: {percentage:.2f}%")
    print(f"Unicode script: {report['unicode_script'] or 'unknown'}")
    print(f"Raw OCR text: {report['raw_text'] or '(no text detected)'}")
    if report["saved_image"]:
        print(f"Saved preprocessed image: {report['saved_image']}")
    if report["correction_method"] == "low_confidence":
        print(
            f"Correction skipped (OCR confidence {report['ocr_confidence']:.1f}% is too low). "
            "Raw text is shown above."
        )
    elif report["correction_method"] in {"reference", "dictionary"}:
        print(f"Corrected text ({report['correction_method']}): {report['text'] or '(no text detected)'}")
    else:
        print(f"Text (not auto-corrected): {report['text'] or '(no text detected)'}")
    print("Character letter scores:")
    if not report["characters"]:
        print("  (no alphabetic characters detected)")
    for character in report["characters"]:
        scores = " ".join(
            f"{letter}={percentage:.2f}%"
            for letter, percentage in character["percentages"].items()
        )
        print(
            f"  word {character['word_index']} character {character['character_index']} "
            f"{character['recognized']!r}: {scores}"
        )
    if args.debug:
        print("Debug details saved to debug.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())