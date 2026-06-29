"""FastAPI application entry point."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import documents, queries, agent_routes, canvas, ingest, charts


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start hot-folder watcher on startup (watches ~/Downloads by default)
    from hot_folder import start_hot_folder_watcher
    watched = start_hot_folder_watcher()
    if watched:
        print(f"[startup] Hot-folder watcher active: {watched}")
    yield


app = FastAPI(
    title="Financial RAG API",
    description="PDF ingestion, RAG queries, agentic research, and canvas persistence.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(queries.router,   prefix="/queries",   tags=["queries"])
app.include_router(agent_routes.router, prefix="/agent",  tags=["agent"])
app.include_router(canvas.router,    prefix="/canvas",    tags=["canvas"])
app.include_router(ingest.router,    prefix="/ingest",    tags=["ingest"])
app.include_router(charts.router,    prefix="/charts",    tags=["charts"])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"message": "Financial RAG API. See /docs for endpoints."}
