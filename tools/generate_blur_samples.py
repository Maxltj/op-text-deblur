import csv
import io
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "blur_samples"
CANVAS_SIZE = (720, 220)
SUPERSAMPLE = 2
BLUR_PERCENTAGES = tuple(round(index * 100 / 9) for index in range(10))
MAX_BLUR_RADIUS = 3.0
SAMPLES = (
    {
        "id": "chat_message",
        "background": (31, 33, 36),
        "foreground": (241, 243, 244),
        "lines": (
            "Project update",
            "The review is scheduled for Thursday at 10:30.",
            "Please bring the revised draft.",
        ),
    },
    {
        "id": "delivery_notice",
        "background": (245, 243, 235),
        "foreground": (35, 42, 48),
        "lines": (
            "DELIVERY NOTICE",
            "Package 483920 is ready for collection.",
            "Pickup location: North reception, room 14B.",
        ),
    },
    {
        "id": "receipt",
        "background": (250, 250, 247),
        "foreground": (42, 44, 41),
        "lines": (
            "MARKET STREET CAFE",
            "2 coffees                 $7.50",
            "TOTAL                     $12.80",
        ),
    },
    {
        "id": "station_sign",
        "background": (31, 74, 98),
        "foreground": (250, 250, 247),
        "lines": (
            "RIVERSIDE STATION",
            "Platform 3 - Eastbound trains",
            "Next departure: 08:42",
        ),
    },
)


def _font(size):
    for font_name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(font_name, size * SUPERSAMPLE)
        except OSError:
            continue
    return ImageFont.load_default(size=size * SUPERSAMPLE)


def _font_at_size(size):
    for font_name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _render_sample(sample):
    width, height = CANVAS_SIZE
    image = Image.new(
        "RGB",
        (width * SUPERSAMPLE, height * SUPERSAMPLE),
        sample["background"],
    )
    draw = ImageDraw.Draw(image)
    for line, size, top in zip(sample["lines"], (25, 22, 22), (40, 91, 140)):
        draw.text(
            (36 * SUPERSAMPLE, top * SUPERSAMPLE),
            line,
            font=_font(size),
            fill=sample["foreground"],
        )
    return image.resize(CANVAS_SIZE, Image.Resampling.LANCZOS)


def _preprocess(image):
    grayscale = ImageOps.grayscale(image)
    denoised = grayscale.filter(ImageFilter.MedianFilter(size=3))
    sharpened = denoised.filter(
        ImageFilter.UnsharpMask(radius=1.4, percent=280, threshold=1)
    )
    return ImageEnhance.Contrast(sharpened).enhance(1.2).convert("RGB")


def _recognize(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    result = subprocess.run(
        ["tesseract", "stdin", "stdout", "-l", "eng", "--psm", "6", "tsv"],
        input=buffer.getvalue(),
        capture_output=True,
        check=True,
    )
    words = []
    confidences = []
    rows = csv.DictReader(
        io.StringIO(result.stdout.decode("utf-8", errors="replace")),
        delimiter="\t",
    )
    for row in rows:
        word = row.get("text", "").strip()
        try:
            confidence = float(row.get("conf", -1))
        except ValueError:
            continue
        if word and confidence >= 0:
            words.append(word)
            confidences.append(confidence)
    text = " ".join(words) or "No text recognized."
    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return text, confidence


def _save_output_with_text(processed_image, output_path, text, confidence):
    margin = 18
    text_font = _font_at_size(17)
    title_font = _font_at_size(13)
    max_width = processed_image.width - margin * 2
    wrapped_lines = []
    for paragraph in (text.strip() or "No text recognized.").splitlines():
        current = ""
        for word in paragraph.split():
            candidate = f"{current} {word}".strip()
            bounds = ImageDraw.Draw(processed_image).textbbox((0, 0), candidate, font=text_font)
            if current and bounds[2] > max_width:
                wrapped_lines.append(current)
                current = word
            else:
                current = candidate
        wrapped_lines.append(current)

    draw = ImageDraw.Draw(processed_image)
    line_heights = [draw.textbbox((0, 0), line, font=text_font)[3] for line in wrapped_lines]
    panel_height = margin * 2 + 24 + sum(line_heights) + max(0, len(line_heights) - 1) * 4
    output = Image.new(
        "RGB",
        (processed_image.width, processed_image.height + panel_height),
        (246, 247, 248),
    )
    output.paste(processed_image.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(output)
    y = processed_image.height + margin
    draw.text((margin, y), f"OCR TEXT | {confidence:.1f}% confidence", font=title_font, fill=(75, 86, 94))
    y += 23
    for line, height in zip(wrapped_lines, line_heights):
        draw.text((margin, y), line, font=text_font, fill=(28, 34, 39))
        y += height + 4
    output.save(output_path)
    return output


def _blurred_image(clean_image, blur_percent):
    radius = MAX_BLUR_RADIUS * blur_percent / 100
    if radius == 0:
        return clean_image.copy()
    return clean_image.filter(ImageFilter.GaussianBlur(radius=radius))


def _render_gallery(output_dir, manifest):
    samples_by_id = {sample["id"]: sample for sample in SAMPLES}
    for entry in manifest["samples"]:
        clean = _render_sample(samples_by_id[entry["source_id"]])
        blurred = _blurred_image(clean, entry["blur_percent"])
        processed = _preprocess(blurred)
        recognized_text, confidence = _recognize(processed)
        blurred.save(output_dir / entry["file"])
        saved_output = _save_output_with_text(
            processed,
            output_dir / entry["preprocessed_file"],
            recognized_text,
            confidence,
        )
        comparison = Image.new(
            "RGB",
            (blurred.width + saved_output.width, max(blurred.height, saved_output.height)),
            (242, 244, 246),
        )
        comparison.paste(blurred.convert("RGB"), (0, 0))
        comparison.paste(saved_output.convert("RGB"), (blurred.width, 0))
        comparison.save(output_dir / entry["comparison_file"])
        entry["recognized_text"] = recognized_text
        entry["ocr_confidence"] = confidence

    margin = 20
    board_width = 680
    column_gap = 16
    column_width = (board_width - margin * 2 - column_gap) // 2
    heading_font = _font_at_size(18)
    section_font = _font_at_size(13)
    label_font = _font_at_size(11)
    board_paths = []

    for sample in SAMPLES:
        clean = _render_sample(sample)
        entries = [entry for entry in manifest["samples"] if entry["source_id"] == sample["id"]]
        entry_zero = next(entry for entry in entries if entry["blur_percent"] == 0)
        output_zero = Image.open(output_dir / entry_zero["preprocessed_file"]).convert("RGB")

        top_source_width = board_width - margin * 2
        source_scale = top_source_width / clean.width
        source_height = round(clean.height * source_scale)
        output_scale = top_source_width / output_zero.width
        output_zero_height = round(output_zero.height * output_scale)
        pair_image_height = round(column_width * CANVAS_SIZE[1] / CANVAS_SIZE[0])
        pair_output_height = round(column_width * output_zero.height / output_zero.width)
        pair_row_height = 24 + max(pair_image_height, pair_output_height) + 10

        board_height = (
            margin + 27 + source_height + 24 + 22 + output_zero_height
            + 28 + 22 + len(entries) * pair_row_height + margin
        )
        board = Image.new("RGB", (board_width, board_height), (245, 247, 248))
        draw = ImageDraw.Draw(board)
        y = margin
        draw.text((margin, y), f"{sample['id']} | BEFORE BLUR", font=heading_font, fill=(35, 42, 48))
        y += 27
        board.paste(clean.resize((top_source_width, source_height), Image.Resampling.LANCZOS), (margin, y))
        y += source_height + 24

        draw.text((margin, y), "0% BLUR | PROGRAM OUTPUT WITH OCR TEXT", font=section_font, fill=(35, 42, 48))
        y += 22
        board.paste(output_zero.resize((top_source_width, output_zero_height), Image.Resampling.LANCZOS), (margin, y))
        y += output_zero_height + 28

        draw.text((margin, y), "BLUR SCALE | INPUT / SAVED OUTPUT", font=section_font, fill=(35, 42, 48))
        y += 22
        for entry in entries:
            input_image = Image.open(output_dir / entry["file"]).convert("RGB")
            output_image = Image.open(output_dir / entry["preprocessed_file"]).convert("RGB")
            input_thumb = input_image.resize((column_width, pair_image_height), Image.Resampling.LANCZOS)
            output_thumb = output_image.resize((column_width, pair_output_height), Image.Resampling.LANCZOS)
            draw.text((margin, y), f"{entry['blur_percent']}% INPUT", font=label_font, fill=(45, 52, 58))
            draw.text((margin + column_width + column_gap, y), f"{entry['blur_percent']}% OUTPUT + OCR", font=label_font, fill=(45, 52, 58))
            board.paste(input_thumb, (margin, y + 24))
            board.paste(output_thumb, (margin + column_width + column_gap, y + 24))
            y += pair_row_height

        board_path = output_dir / f"{sample['id']}_before_after.png"
        board.save(board_path)
        board_paths.append((sample["id"], board_path))

    overview_width = 760
    overview_margin = 18
    overview_cell_width = (overview_width - overview_margin * 3) // 2
    overview_cell_height = 500
    overview = Image.new(
        "RGB",
        (overview_width, overview_margin + 2 * (overview_cell_height + overview_margin)),
        (238, 241, 243),
    )
    overview_draw = ImageDraw.Draw(overview)
    overview_font = _font_at_size(12)
    for index, (sample_id, board_path) in enumerate(board_paths):
        with Image.open(board_path) as board:
            thumbnail = ImageOps.contain(board.convert("RGB"), (overview_cell_width, overview_cell_height - 26))
        left = overview_margin + (index % 2) * (overview_cell_width + overview_margin)
        top = overview_margin + (index // 2) * (overview_cell_height + overview_margin)
        overview_draw.text((left, top), sample_id, font=overview_font, fill=(35, 42, 48))
        overview.paste(thumbnail, (left, top + 24))
    overview.save(output_dir / "before_after_gallery.png")


def generate_samples(output_dir=OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_manifest_path = output_dir / "metadata.json"
    if previous_manifest_path.is_file():
        try:
            previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_manifest = {"samples": []}
        for entry in previous_manifest.get("samples", []):
            for key in ("file", "preprocessed_file", "comparison_file"):
                generated_path = entry.get(key)
                if generated_path:
                    (output_dir / generated_path).unlink(missing_ok=True)
    for old_name in ("preview.png", "before_after_gallery.png"):
        (output_dir / old_name).unlink(missing_ok=True)
    (output_dir / "comparisons").mkdir(exist_ok=True)

    manifest = {
        "description": "Synthetic text images generated locally; no user photographs used.",
        "canvas_size": list(CANVAS_SIZE),
        "blur_method": "Pillow GaussianBlur radius in pixels",
        "blur_percentages": list(BLUR_PERCENTAGES),
        "maximum_blur_radius": MAX_BLUR_RADIUS,
        "samples": [],
    }
    for source_index, sample in enumerate(SAMPLES):
        clean = _render_sample(sample)
        for blur_index, blur_percent in enumerate(BLUR_PERCENTAGES):
            stem = f"{sample['id']}_blur_{blur_percent:03d}"
            manifest["samples"].append({
                "file": f"{stem}.png",
                "preprocessed_file": f"{stem}_program_output.png",
                "comparison_file": f"comparisons/{stem}_comparison.png",
                "source_id": sample["id"],
                "blur_percent": blur_percent,
                "blur_radius": MAX_BLUR_RADIUS * blur_percent / 100,
                "blur_level_index": blur_index,
                "source_index": source_index,
                "ground_truth": "\n".join(sample["lines"]),
            })

    _render_gallery(output_dir, manifest)
    (output_dir / "metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return len(manifest["samples"])


if __name__ == "__main__":
    count = generate_samples()
    print(f"Generated {count} synthetic blur/input/output samples in {OUTPUT_DIR}")
