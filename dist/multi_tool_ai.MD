# MultitoolAI Technical Documentation

## Methodology of the protected methods
Below is a more detailed explanation of the protected methods and its funciton

## Vector Search & Entity Matching

### Document Processing Implementation
Building upon the embedding functionality established in the code implementation (embedding.py)
```python
# Key implementation elements from the existing code:
vectors = encoder.encode(segments).astype("float32")  # Memory-efficient encoding
faiss.normalize_L2(vectors)  # Fair comparison normalization  
index = faiss.IndexFlatL2(d)  # Fast similarity search index
```
This private function utilizes the `paraphrase-mpnet-base-v2` model, and the vectorized segments becomes on the mathematical plane as a plain basis for the query. The query is then transform into mathemathical dimensions. Enabling semantic matching while extracting categorical context.

### Advanced OCR Image Processing
The `_ocr_image(gray_img, reader, use_filter: bool)` method processes images through the easyOCR library. When disabled, raw images are processed directly by easyOCR without preprocessing. When enabled, the method implements a three-stage enhancement pipeline.

#### Image Enhancement Pipeline
Filter to removes noise and speckles while preserving text edges for cleaner OCR input.
A Bilateral Filter to accentuates character boundaries and edges.
Otsu Thresholding Filter, allowing for high-contrast black and white alongside different filters
#### Implementation Flow
The pipeline then transforms unclear images into optimized binary text through the use of the easyOCR library with a greater or lesser accuracy depending on if the user selected for it to use_filter in case the image may not produce a rsult as desirable.
