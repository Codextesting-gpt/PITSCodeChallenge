# ---- main.py --------------------------------------------------------------
from fastapi import FastAPI, UploadFile, File, Form
from typing import List
import os, shutil, uuid, time
import numpy as np            #  ← new
from ocr import ocr_file
from embedding import Embeddings
from llm_google_genai_nlp import extract_values

app = FastAPI()
DOCS_DIR = "docs"

def reset_docs_dir() -> None:
    os.makedirs(DOCS_DIR, exist_ok=True)
    for f in os.listdir(DOCS_DIR):
        fp = os.path.join(DOCS_DIR, f)
        if os.path.isfile(fp):
            os.remove(fp)

# --- helper that strips NumPy dtypes ---------------------------------------
def to_builtin(x):
    """Convert NumPy scalars/arrays to plain Python so JSON encoding works."""
    if isinstance(x, np.generic):      # any NumPy scalar (int32, float64, …)
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

def clean_ocr_result(raw):
    """raw is [(page,bbox,text,prob), …] from ocr_file; make it JSON-safe."""
    cleaned = []
    for page, bbox, text, prob in raw:
        cleaned.append({
            "page": int(page),
            "bbox": [[to_builtin(pt) for pt in corner] for corner in bbox],
            "text": str(text),
            "prob": float(prob),
        })
    return cleaned
# ---------------------------------------------------------------------------

@app.post("/extract_entities")
async def extract_entities(
    documents: List[UploadFile] = File(...),
    entities:  List[str]       = Form(...)
):
    start_time = time.time()

    ocr_results = {}
    for upload in documents:
        filename = upload.filename or f"{uuid.uuid4()}"
        path = os.path.join(DOCS_DIR, filename)

        # save the upload
        with open(path, "wb") as dst:
            shutil.copyfileobj(upload.file, dst)
        upload.file.close()

        # OCR and sanitise
        raw   = ocr_file(path, use_filter=False)
        ocr_results[filename] = clean_ocr_result(raw)
    matches = Embeddings(ocr_results, entities, min_seg_len=100)
    rendered_result_with = extract_values(matches, entities, time=30)
    end_time = time.time()
    actual_processing_time = round(end_time - start_time, 3)  # Round to 3 decimal places
    
    # Update the processing_time with the actual measured time
    rendered_result_with["processing_time"] = f"{actual_processing_time}s"
    
    reset_docs_dir()
    return rendered_result_with