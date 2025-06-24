# Document Processing API

A secure, AI-powered document processing API that extracts structured data from images and PDFs using OCR, semantic search, and Large Language Models.

## 🎯 Project Overview 

This API provides a complete document processing pipeline that can:

1. **Accept document uploads** - Support for images (PNG, JPG) and PDF files
2. **Extract text content** - Advanced Optical Character Recognition (OCR) with preprocessing
3. **Identify document types** - Semantic search against vector embeddings for intelligent categorization
4. **Extract structured data** - Use Large Language Models to identify and extract specific entities
5. **Return standardized responses** - Clean JSON output with confidence scores and processing metadata


## 🚀 Installation Instructions

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU (optional, for faster processing)
- Poppler utilities for PDF processing (works for linux and windows)

### Step 1: Install Python Dependencies

```bash
pip install fastapi uvicorn
pip install sentence-transformers
pip install langchain langchain-google-genai
pip install pdf2image pillow opencv-python easyocr
pip install faiss-cpu  # or faiss-gpu for CUDA support
pip install pandas numpy pydantic python-dotenv
pip install time
```


### Step 2: Install Poppler (for PDF support)

**Windows:**
1. Download Poppler from [poppler-windows releases](https://github.com/oschwartz10612/poppler-windows/releases)
2. Extract to `C:\poppler\`
3. Update the `POPPLER_BIN` path in `ocr.py` if needed

**Linux/macOS:**
```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils
```

### Step 3: Set up Project Structure

```
dist/
├── main.py
├── ocr.py
├── embedding.py
├── llm_google_genai_nlp.py
├── multi_tool_ai.py
├── .env
├── pyarmor_runtime_000000
└── README.md
```

## ⚙️ Configuration Guidelines

### Environment Variables

Create a `.env` file in your project root:

```env
# Google AI API Key (required for LLM processing)
GOOGLE_API_KEY=your_google_ai_api_key_here

# MultitoolAI Authentication for OCR
fallout=fc82b26aecb47d42e649f972c91dabc4dae7cdace893e8ae7e48b6c23ab4f5cf
```

### OCR Configuration

The system supports multiple OCR languages. Modify in `ocr.py`:

```python
# Default: English only
languages = ["en"]

# Multiple languages
languages = ["en", "es", "fr"]  # English, Spanish, French
```

### Model Configuration

Default models can be customized:

```python
# Embedding model (in embedding.py)
encoder = SentenceTransformer("paraphrase-mpnet-base-v2")

# LLM model (in llm_google_genai_nlp.py)  
model_name = "gemini-2.0-flash"  # or "gemini-pro"
```

## 📚 API Documentation

### Endpoint: `POST /extract_entities`

This is the endpoint which fastapi is covering

**Request expected Format:**
```bash
curl -X POST http://127.0.0.1:8000/extract_entities \
  -F "documents=@path/to/document1.pdf" \
  -F "documents=@path/to/document2.png" \
  -F "entities=name" \
  -F "entities=organization" \
  -F "entities=date"
  ...
```

**Parameters:**
- `documents` (files): One or more document files (PDF, PNG, JPG)
- `entities` (strings): List of data fields to extract

**Response Format:**
```json
{
  "document_type": "invoice",
  "confidence": 0.85,
  "processing_time": "45.2 seconds",
  "entities": {
    "name": "John Doe",
    "organization": "Acme Corp",
    "date": "2024-03-15"
  }
}
```

**Response Fields:**
- `document_type`: Identified document category
- `confidence`: Overall extraction confidence (LLM Generated)
- `processing_time`: Total processing duration
- `entities`: Key-value pairs of extracted data


## 🧪 Testing Procedures

### 1. Start the Development Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

### 2. Basic Functionality Test

Test with sample documents:

```bash
# Windows
curl -X POST http://127.0.0.1:8000/extract_entities ^
  -F "documents=@C:\Users\username\Documents\invoice.pdf" ^
  -F "entities=total" ^
  -F "entities=date"

# Linux/macOS  
curl -X POST http://127.0.0.1:8000/extract_entities \
  -F "documents=@/path/to/invoice.pdf" \
  -F "entities=total" \
  -F "entities=date"
```

### 3. Multi-Document Test

```bash
curl -X POST http://127.0.0.1:8000/extract_entities ^
  -F "documents=@project.pdf" ^
  -F "documents=@page_001.png" ^
  -F "entities=nombre" ^
  -F "entities=organizacion" ^
  -F "entities=date"
```
This API uses the `MultitoolAI` code which requires a proper API Key


## 🔒 Security Features

- **Authentication Required**: Production use requires proper environment variable configuration
- **Use of external libraries or propietary code**: The use of propietary code like langchain is used in the model

## 🐛 Troubleshooting

**Common Issues:**

1. **Poppler Path Error**: Update `POPPLER_BIN` in `ocr.py` to match your installation
3. **GPU Memory Issues**: Switch from `faiss-gpu` to `faiss-cpu` if encountering memory problems
4. **API Key Error**: Verify `GOOGLE_API_KEY` is valid and has proper permissions

**Debug Mode:**
Add debug logging to see detailed processing steps:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Elements that could improve

- Use GPU acceleration for FAISS operations when available
- Batch multiple documents in single API calls
- Adjust RAG to ensure better exactitude or speed
- Consider caching embeddings for repeated document types

## For a more detailed explanation of how the code works you are more than welcome to delve into the code comments, the image for the flow process or the explanation of complex functions in the multi_tool_ai.MD
---
