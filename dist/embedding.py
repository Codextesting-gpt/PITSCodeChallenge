import faiss
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sentence_transformers import SentenceTransformer
from multi_tool_ai import MultitoolAI

# Global encoder – load once
encoder = SentenceTransformer("paraphrase-mpnet-base-v2")


# ──────────────────────────────────────────────────────────
# Helper ─ page-aware, single-pass segmentation
# ──────────────────────────────────────────────────────────
def page_to_segments(page_records: List[Dict[str, str]],
                     *,
                     min_len: int = 25) -> List[str]:
    """
    Merge consecutive short OCR lines (within the SAME page) until their
    combined length crosses `min_len`.  Long lines (≥ min_len) are kept
    as-is.  Nothing is discarded.
    """
    segments, buffer, buf_len = [], [], 0

    for rec in page_records:
        line = rec["text"].strip()
        if not line:
            continue

        if len(line) >= min_len:            # long → flush buffer first
            if buffer:
                segments.append(" ".join(buffer))
                buffer, buf_len = [], 0
            segments.append(line)
        else:                               # short → accumulate
            buffer.append(line)
            buf_len += len(line)
            if buf_len >= min_len:
                segments.append(" ".join(buffer))
                buffer, buf_len = [], 0

    if buffer:                              # leftovers
        segments.append(" ".join(buffer))

    return segments


# ──────────────────────────────────────────────────────────
# Main function ─ build embeddings, FAISS index, search
# ──────────────────────────────────────────────────────────
def Embeddings(
    ocr_results: Dict[str, List[dict]],
    entities: List[str],
    *,
    min_seg_len: int = 25
) -> List[Dict[str, Any]]:
    """

    """
    # 1. Flatten OCR into (filename, page, segment) rows ──────────────
    rows: List[Tuple[str, int, str]] = []

    for fname, records in ocr_results.items():
        # group each file's records by page number
        pages: Dict[int, list] = {}
        for rec in records:
            pages.setdefault(rec["page"], []).append(rec)

        # process every page separately
        for page_no, page_recs in pages.items():
            for seg in page_to_segments(page_recs, min_len=min_seg_len):
                rows.append((fname, page_no, seg))

    if not rows:
        return []

    df = pd.DataFrame(rows, columns=["filename", "page", "segment"])

    # 2. Encode all segments & build FAISS index ──────────────────────
    vectors = encoder.encode(
        df["segment"].tolist(), show_progress_bar=False
    ).astype("float32")

    faiss.normalize_L2(vectors)
    d = vectors.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(vectors)
    ai = MultitoolAI()
    
    # returns {} with entity, df as each category, matched segment, prev and next segment
    #matches.append({ "entity": entity, "category": "page": int(df.at[row_id, "page"]), "matched_text": cur_text  })

    matches = ai._search_and_extract_matches(df, entities, encoder, index)
    return matches