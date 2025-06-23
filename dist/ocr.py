from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import cv2
import easyocr
from multi_tool_ai import MultitoolAI


# ────────────────────────────────
# 2️⃣  Helper: save image + OCR
# ────────────────────────────────
def save_and_ocr(pil_img,           # PIL.Image.Image
                 save_dir: Path,
                 save_name: str,
                 page_no: int,
                 reader,
                 use_filter: bool):
    """
    • Saves `pil_img` as PNG inside `save_dir/save_name`
    • Runs OCR via _ocr_image()
    • Returns [(page_no, bbox, text, conf), …]
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    img_path = save_dir / save_name
    pil_img.save(img_path)

    # Re-load via cv2 so we get a NumPy array (BGR) → convert to gray
    gray = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2GRAY)

    results = []
    ocr_ai = MultitoolAI()
    for bbox, text, prob in ocr_ai._ocr_image(gray, reader, use_filter):
        results.append((page_no, bbox, text, prob))
    return results


# ────────────────────────────────
# 3️⃣  Public entry-point
# ────────────────────────────────
def ocr_file(file_path: str | Path,
             out_root: str | Path = "out_imgs",
             use_filter: bool = True,
             languages: list[str] | None = None):
    """
    • If `file_path` is a PDF → render pages → save+OCR each page
    • If it’s any other image → save copy → OCR it
    Returns list[(page_no, bbox, text, prob), …]
    """
    file_path = Path(file_path)
    out_root = Path(out_root)
    languages = languages or ["en"]

    reader = easyocr.Reader(languages, gpu=False)
    results = []
    POPPLER_BIN = r"C:\poppler\poppler-24.08.0\Library\bin"
    if file_path.suffix.lower() == ".pdf":
        # Sub-folder named after the PDF (without extension)
        pdf_dir = out_root / file_path.stem
        pages = convert_from_path(file_path, dpi=200, fmt="png", thread_count=8, poppler_path=POPPLER_BIN)

        for idx, page in enumerate(pages, 1):
            results.extend(
                save_and_ocr(
                    page,
                    pdf_dir,
                    f"page_{idx:03}.png",
                    idx,
                    reader,
                    use_filter,
                )
            )
    else:
        # Single image
        img_dir = out_root / file_path.stem
        pil_img = Image.open(file_path)      # keeps original colours
        results.extend(
            save_and_ocr(
                pil_img,
                img_dir,
                f"{file_path.stem}.png",
                1,
                reader,
                use_filter,
            )
        )

    return results


# ────────────────────────────────
# 4️⃣  Example CLI usage
# ────────────────────────────────