import os
import asyncio
import json
import hashlib
import time
import base64
import logging
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv


from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from tempfile import NamedTemporaryFile
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from PIL import Image
import io
import re
from mistralai import Mistral

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MedicalAssistant")

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("❌ GOOGLE_API_KEY environment variable missing.")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

class Config:
    CACHE_MAX_SIZE = 200
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    SIMILARITY_SEARCH_K = 5
    MAX_IMAGE_SIZE = (1024, 1024)
    IMAGE_QUALITY = 85

config = Config()

class AppState:
    def __init__(self):
        self.vectorstore: Optional[FAISS] = None
        self.cache: Dict[str, str] = {}
        self.file_metadata: Dict[str, Dict[str, Any]] = {}
        self.start_time = time.time()

    def add_to_cache(self, key: str, value: str, meta: Dict[str, Any] = None):
        if len(self.cache) >= config.CACHE_MAX_SIZE:
            oldest = next(iter(self.cache))
            self.cache.pop(oldest, None)
            self.file_metadata.pop(oldest, None)
        self.cache[key] = value
        if meta:
            self.file_metadata[key] = meta

    def get_from_cache(self, key: str) -> Optional[str]:
        return self.cache.get(key)

app_state = AppState()

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)

    @validator("question")
    def check_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()

class UploadResponse(BaseModel):
    status: str
    chunks: int
    file_id: str
    message: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    vectorstore_loaded: bool
    cached_files: int

def hash_content(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

def validate_file_size(size: int):
    if size > config.MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large.")

def clean_markdown(text: str) -> str:
    text = re.sub(r'(\*\*|\*)(.*?)\1', r'\2', text)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.M)
    text = re.sub(r'^\s*[-*•]\s*', '', text, flags=re.M)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
async def call_llm_safe(llm, msgs):
    return await llm.ainvoke(msgs)

def init_models():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        max_tokens=2048,
        google_api_key=GOOGLE_API_KEY,
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )
    return llm, embeddings

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, embeddings
    llm, embeddings = init_models()
    logger.info("🚀 Warming up...")
    await llm.ainvoke("System check")
    yield
    logger.info("🛑 Shutting down...")
    
app = FastAPI(title="Medical Assistant", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        timestamp=str(int(time.time())),
        vectorstore_loaded=app_state.vectorstore is not None,
        cached_files=len(app_state.cache)
    )

@app.post("/upload_pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, detail="Only PDF files supported.")

    content = await file.read()
    validate_file_size(len(content))
    key = hash_content(content)

    if app_state.get_from_cache(key):
        return UploadResponse(status="cached", chunks=0, file_id=key[:8], message="Already processed.")

    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        path = tmp.name

    try:
        docs = PyPDFLoader(path).load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(docs)
        if app_state.vectorstore:
            app_state.vectorstore.add_documents(chunks)
        else:
            app_state.vectorstore = FAISS.from_documents(chunks, embeddings)
        app_state.add_to_cache(key, "processed", {"filename": file.filename, "chunks": len(chunks)})
        return UploadResponse(status="success", chunks=len(chunks), file_id=key[:8], message="PDF processed.")
    finally:
        os.unlink(path)

@app.post("/upload_image", response_model=UploadResponse)
async def upload_image(file: UploadFile):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="Only image files supported.")
    content = await file.read()
    validate_file_size(len(content))
    key = hash_content(content)

    if app_state.get_from_cache(key):
        return UploadResponse(status="cached", chunks=0, file_id=key[:8], message="Already processed.")

    image = Image.open(io.BytesIO(content)).convert("RGB")
    image.thumbnail(config.MAX_IMAGE_SIZE)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=config.IMAGE_QUALITY)
    b64 = base64.b64encode(buf.getvalue()).decode()

    ocr_prompt = HumanMessage(content=[
        {"type": "text", "text": "Extract visible medical text."},
        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64}"}
    ])
    result = await call_llm_safe(llm, [ocr_prompt])
    text = result.content.strip()
    if not text:
        return UploadResponse(status="no_text", chunks=0, file_id=key[:8], message="No text found.")

    docs = [Document(page_content=text)]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    if app_state.vectorstore:
        app_state.vectorstore.add_documents(chunks)
    else:
        app_state.vectorstore = FAISS.from_documents(chunks, embeddings)
    app_state.add_to_cache(key, text, {"filename": file.filename, "chunks": len(chunks)})
    return UploadResponse(status="success", chunks=len(chunks), file_id=key[:8], message="Image OCR done.")

@app.post("/upload_image", response_model=UploadResponse)
async def upload_image(file: UploadFile):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="Only image files supported.")
    content = await file.read()
    validate_file_size(len(content))
    key = hash_content(content)

    if app_state.get_from_cache(key):
        return UploadResponse(status="cached", chunks=0, file_id=key[:8], message="Already processed.")

    image = Image.open(io.BytesIO(content)).convert("RGB")
    image.thumbnail(config.MAX_IMAGE_SIZE)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=config.IMAGE_QUALITY)
    b64 = base64.b64encode(buf.getvalue()).decode()

    logger.info("Running Mistral OCR...")
    ocr_response = mistral_client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{b64}"
        },
        include_image_base64=True
    )
    text = ocr_response.model_dump_json()
    if not text:
        return UploadResponse(status="no_text", chunks=0, file_id=key[:8], message="No text found.")

    docs = [Document(page_content=text)]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    if app_state.vectorstore:
        app_state.vectorstore.add_documents(chunks)
    else:
        app_state.vectorstore = FAISS.from_documents(chunks, embeddings)
    app_state.add_to_cache(key, text, {"filename": file.filename, "chunks": len(chunks)})
    return UploadResponse(status="success", chunks=len(chunks), file_id=key[:8], message="Image OCR done.")

@app.post("/ask")
async def ask(req: AskRequest):
    context = ""
    if app_state.vectorstore:
        docs = app_state.vectorstore.similarity_search(req.question, k=config.SIMILARITY_SEARCH_K)
        context = "\n\n".join(doc.page_content for doc in docs)

    if context:
        prompt = f"""
You are a professional and cautious medical assistant. Below is the extracted text from the patient’s prescription or medical record, if available. Your primary task is to answer the user’s question using this context when relevant. 
If the question is not directly answered by the context or no context is provided, you may provide safe, generalized medical advice based on common medical knowledge, ensuring it remains cautious and appropriate.
✅ If the answer is clearly found in the prescription or medical record, provide it directly, referencing the context explicitly.
✅ If the answer is not present or only partially present in the context, or if the question is unrelated to the context, provide safe and general medical advice or possible over-the-counter options relevant to the question.
✅ NEVER guess exact dosages, frequencies, or make direct diagnoses that are not clearly mentioned in the prescription or medical notes.
✅ If the question cannot be answered confidently with the context or general knowledge, state that you do not know and recommend consulting a doctor.
✅ For non-medical questions, respond with: "I am a medical bot and cannot answer the following question."
✅ Respond in clear plain text only — no markdown, no bullet points, no headings.

PRESCRIPTION / MEDICAL RECORD:
{context}

USER QUESTION:
{req.question}

ANSWER:
"""
    else:
        prompt = f"""
You are a helpful medical assistant. No prescription was provided, so rely on general medical knowledge.
You can answer basic greetings like hey!,hi! etc... in a concise way.
Give precautions, first-aid, or OTC suggestions based on common guidelines.
Never prescribe specific dosages without a doctor’s review.
If asked about a topic out of medical information or a question which is not related to medical field, you shouldnt answer it,Instead you can answer as 'Im a medical Bot and i cannot answer teh following question 
And make answer more concise and medium range.
USER QUESTION:
{req.question}

ANSWER:
"""

    async def gen():
        async for chunk in llm.astream(prompt):
            if chunk.content:
                yield clean_markdown(chunk.content)

    return StreamingResponse(gen(), media_type="text/plain")

# ---------------------------------------------
# Run dev server
# ---------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
