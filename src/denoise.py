from PIL import ImageFilter


def denoising_variants(grayscale_image):
    """Return conservative grayscale denoising candidates for OCR comparison."""
    return (
        ("median3", grayscale_image.filter(ImageFilter.MedianFilter(size=3))),
        ("median5", grayscale_image.filter(ImageFilter.MedianFilter(size=5))),
        ("mode3", grayscale_image.filter(ImageFilter.ModeFilter(size=3))),
        ("gaussian_mild", grayscale_image.filter(ImageFilter.GaussianBlur(radius=0.45))),
    )