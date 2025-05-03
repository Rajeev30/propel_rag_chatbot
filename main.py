# main.py
import os
from dotenv import load_dotenv

# Load .env into os.environ (does not override existing vars by default)
load_dotenv()

from typing import Any, Dict, List
import logging
import time
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from duckduckgo_search import DDGS

from rag_service import process_document, build_rag_chain

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure the key is now available
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("⚠️ OPENAI_API_KEY not set. Please define it in your .env file.")

app = FastAPI(title="Document RAG Assistant API")

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Request started: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Request completed: {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s")
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.exception(f"Request failed: {request.method} {request.url.path} - Time: {process_time:.3f}s")
        raise

# CORS (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProcessRequest(BaseModel):
    document_path: str
    vector_db_dir: str = "faiss_index"
    content_dir: str = "content"

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

class WebSearchResponse(BaseModel):
    results: List[Dict[str, Any]]

@app.get("/")
def health_check():
    """Simple health check endpoint that always returns 200 OK"""
    return {"status": "healthy", "message": "API is running"}

@app.post("/process_document")
def process_document_endpoint(req: ProcessRequest):
    try:
        logger.info(f"Processing document request: {req.document_path}")
        result = process_document(
            document_path=req.document_path,
            vector_db_dir=req.vector_db_dir,
            content_dir=req.content_dir
        )
        logger.info(f"Document processed successfully: {req.document_path}")
        return {"status": "processed"}
    except FileNotFoundError as e:
        logger.error(f"File not found: {req.document_path}")
        raise HTTPException(status_code=404, detail=f"File not found: {req.document_path}")
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    try:
        logger.info(f"Chat request received: {req.question[:50]}...")
        answer = build_rag_chain(req.question)
        logger.info("Generated answer successfully")
        return {"answer": answer}
    except RuntimeError as e:
        if "Document not processed" in str(e):
            logger.warning("Chat attempted before document processing")
            raise HTTPException(status_code=400, detail=str(e))
        else:
            logger.error(f"Runtime error in chat: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/web_search", response_model=WebSearchResponse)
def web_search_endpoint(req: ChatRequest):
    try:
        logger.info(f"Web search request: {req.question[:50]}...")
        ddgs = DDGS()
        results = ddgs.text(req.question, max_results=10)
        results_list = list(results)  # Convert generator to list
        logger.info(f"Web search completed with {len(results_list)} results")
        return {"results": results_list}
    except Exception as e:
        logger.error(f"Error in web search: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
