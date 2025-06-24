# MultitoolAI Technical Documentation

##Methodology of the protected methods
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
This methodology utilizes the `paraphrase-mpnet-base-v2` model, and the previous searchable database (vectorized segments becomes on a  mathematical plane), tranforming the query into a mathemathical value which can then later be enabling semantic matching against the document database, leverages the established embedding model to vectors. Allowing us to compareagainst document segments while extracting categorical context.

### Advanced OCR Image Processing
The `_ocr_image(gray_img, reader, use_filter: bool)` method processes images through the easyOCR library. When disabled, raw images are processed directly by easyOCR without preprocessing. When enabled, the method implements a three-stage enhancement pipeline.

#### Image Enhancement Pipeline

**Median Blur Filter**
Removes noise and speckles while preserving text edges for cleaner OCR input.

**Bilateral Filter**  
Accentuates character boundaries and smooths backgrounds without creating artifacts.

**Otsu Thresholding**
Converts images to high-contrast black and white for optimal easyOCR performance.

#### Implementation Flow
The pipeline then transforms unclear images into optimized binary text through the use of the easyOCR library.
