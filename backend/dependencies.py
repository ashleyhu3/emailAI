"""
Shared singletons for FastAPI dependency injection.

canvas_db is imported first so Canvas/CanvasState are registered on Base
before DatabaseManager calls Base.metadata.create_all().
"""
import os
import sys

# Resolve paths
_backend_dir = os.path.dirname(os.path.abspath(__file__))
_pdf_summarizer_dir = os.path.join(_backend_dir, "..", "PDF_summarizer")
sys.path.insert(0, _pdf_summarizer_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(_backend_dir, "..", ".env"), override=True)

# Must import before DatabaseManager so models are on Base before create_all
import canvas_db  # noqa: F401 — side-effect registers Canvas, CanvasState on Base

from database import DatabaseManager
from pipeline import PDFSummarizerPipeline
from rag_gemini import GeminiRAGPipeline

DB_URL = os.environ["PDF_SUMMARIZER_DB_URL"]

_db_manager: DatabaseManager = DatabaseManager(database_url=DB_URL)
_pipeline: PDFSummarizerPipeline = PDFSummarizerPipeline(database_url=DB_URL)
_rag_pipeline: GeminiRAGPipeline = GeminiRAGPipeline(database_url=DB_URL)


def get_db() -> DatabaseManager:
    return _db_manager


def get_pipeline() -> PDFSummarizerPipeline:
    return _pipeline


def get_rag() -> GeminiRAGPipeline:
    return _rag_pipeline
