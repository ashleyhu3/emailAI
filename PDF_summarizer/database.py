"""
Database module: Golden Schema for financial RAG.

Store verbalization for searching, raw content for answering.
"""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import (
    cast,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    create_engine,
    func,
    literal,
    or_,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import joinedload, relationship, sessionmaker
from pgvector.sqlalchemy import Vector

Base = declarative_base()


def _sender_company_filter(sender_companies):
    """Build a fuzzy, bidirectional substring filter on PDFDocument.sender_company.

    The same authoring firm is often stored under inconsistent names ("SinoPac" vs
    "SinoPac Securities"). Exact matching means a query for one form silently excludes
    docs stored under the other. Instead, match a document when its stored name contains
    the requested name OR the requested name contains the stored name — so any SinoPac
    variant matches every SinoPac document regardless of which form was extracted.
    OR logic across the requested companies.
    """
    clauses = []
    for c in sender_companies:
        clauses.append(
            (PDFDocument.sender_company.isnot(None)) &
            or_(
                PDFDocument.sender_company.ilike(f"%{c}%"),
                literal(c).ilike(func.concat("%", PDFDocument.sender_company, "%")),
            )
        )
    return or_(*clauses)


class PDFDocument(Base):
    """File-level metadata."""

    __tablename__ = "pdf_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(500), nullable=False, unique=True)
    file_path = Column(String(1000), nullable=False)
    total_pages = Column(Integer, nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    file_hash = Column(String(64), nullable=False, unique=True)
    sender_name = Column(String(500), nullable=True)
    sender_company = Column(String(500), nullable=True)
    sent_date = Column(Date, nullable=True)
    written_date = Column(Date, nullable=True)

    # Extended metadata — extracted by Gemini during ingestion
    tickers = Column(JSONB, nullable=True)               # ["BTC", "AAPL"]
    report_type = Column(String(100), nullable=True)     # "equity_research" | "technical_analysis" | …
    sector = Column(String(200), nullable=True)          # GICS sector, e.g. "Technology"
    asset_class = Column(String(100), nullable=True)     # "equity" | "crypto" | "fixed_income" | …
    coverage_period_from = Column(Date, nullable=True)   # Start of the period being analysed
    coverage_period_to = Column(Date, nullable=True)     # End of the period being analysed

    # ── Broker pipeline fields (Phase 3 Agent 2 output) ──────────────────────
    broker = Column(String(200), nullable=True)           # Normalized institution (e.g. "Morgan Stanley")
    broker_action = Column(String(10), nullable=True)     # 'u' | 'd' | 'id' | 'm'
    rating = Column(String(50), nullable=True)            # Overweight | Equal-weight | Underweight
    target_price = Column(Float, nullable=True)           # Extracted price target
    eps_pe = Column(JSONB, nullable=True)                 # {"2024E": {"eps": 5.2, "pe": 18.3}, ...}
    dense_summary = Column(Text, nullable=True)           # Keyword-dense paragraph for SQL routing

    # ── Email ingest provenance ───────────────────────────────────────────────
    email_message_id = Column(String(500), nullable=True, unique=True)  # SHA-256 of email body+attachments
    ingest_source = Column(String(50), nullable=True, default="upload") # 'email' | 'upload'

    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    processed_at = Column(DateTime, nullable=True)

    chunks = relationship(
        "PDFChunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class PDFChunk(Base):
    """
    Golden Schema: one row per page/section.

    - embedding: Verbalization vector (search against this)
    - raw_content: Original Docling markdown (use for answering)
    - verbalized_summary: Gemini chart description (search uses this)
    - metadata: page_number, company_ticker, report_type, file_path
    - image_blob: Optional high-res chart crop for final LLM check
    """

    __tablename__ = "pdf_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(Integer, ForeignKey("pdf_documents.id"), nullable=False)

    # Verbalization vector — embed verbalized_summary, search against this
    embedding = Column(Vector(768), nullable=True)

    # Original Docling markdown (text + reconstructed tables) — use for answering
    raw_content = Column(Text, nullable=False)

    # Gemini plain-text description of charts — used for embedding/search
    verbalized_summary = Column(Text, nullable=True)

    # Denormalized for filtering; also in metadata
    page_number = Column(Integer, nullable=True)

    # page_number, company_ticker, report_type, file_path, etc.
    metadata_ = Column("metadata", JSONB, nullable=True)

    # Optional: high-res chart crop for final LLM check
    image_blob = Column(LargeBinary, nullable=True)

    # Full-text search vector — auto-populated by trigger on insert/update
    tsv = Column(TSVECTOR, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    document = relationship("PDFDocument", back_populates="chunks")


class EmailTriageLog(Base):
    """
    Per-email triage decision log.

    Records every accept/reject decision so the pipeline can:
    1. Skip re-processing already-seen rejected emails (dedup complement to pdf_documents).
    2. Build per-sender reputation scores (auto-block repeat offenders; trust clean senders).
    """

    __tablename__ = "email_triage_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(String(64), nullable=True, index=True)   # SHA-256, for dedup
    sender_email = Column(Text, nullable=False, index=True)
    sender_domain = Column(Text, nullable=False)
    broker = Column(Text, nullable=True)
    subject = Column(Text, nullable=True)
    decision = Column(String(16), nullable=False)                  # 'accepted' | 'rejected'
    rejection_reason = Column(Text, nullable=True)                 # null when accepted
    doc_id = Column(Integer, nullable=True)                        # pdf_documents.id when accepted
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class DatabaseManager:
    """Manages database connections and operations."""

    def __init__(self, database_url: str = "sqlite:///pdf_summarizer.db"):
        """
        Args:
            database_url: postgresql+psycopg://user:pass@localhost/pdf_summarizer
        """
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)

        if database_url.startswith("postgresql"):
            with self.engine.connect() as conn:
                # Schema migrations — always safe (idempotent IF NOT EXISTS)
                conn.execute(text("ALTER TABLE pdf_documents ADD COLUMN IF NOT EXISTS sender_name VARCHAR(500)"))
                conn.execute(text("ALTER TABLE pdf_documents ADD COLUMN IF NOT EXISTS sender_company VARCHAR(500)"))
                conn.execute(text("ALTER TABLE pdf_documents ADD COLUMN IF NOT EXISTS sent_date DATE"))
                conn.execute(text("ALTER TABLE pdf_documents ADD COLUMN IF NOT EXISTS written_date DATE"))
                conn.execute(text("ALTER TABLE pdf_documents ADD COLUMN IF NOT EXISTS tickers JSONB"))
                conn.execute(text("ALTER TABLE pdf_documents ADD COLUMN IF NOT EXISTS report_type VARCHAR(100)"))
                conn.execute(text("ALTER TABLE pdf_documents ADD COLUMN IF NOT EXISTS sector VARCHAR(200)"))
                conn.execute(text("ALTER TABLE pdf_documents ADD COLUMN IF NOT EXISTS asset_class VARCHAR(100)"))
                conn.execute(text("ALTER TABLE pdf_documents ADD COLUMN IF NOT EXISTS coverage_period_from DATE"))
                conn.execute(text("ALTER TABLE pdf_documents ADD COLUMN IF NOT EXISTS coverage_period_to DATE"))
                # Broker pipeline columns
                conn.execute(text("ALTER TABLE pdf_documents ADD COLUMN IF NOT EXISTS broker VARCHAR(200)"))
                conn.execute(text("ALTER TABLE pdf_documents ADD COLUMN IF NOT EXISTS broker_action VARCHAR(10)"))
                conn.execute(text("ALTER TABLE pdf_documents ADD COLUMN IF NOT EXISTS rating VARCHAR(50)"))
                conn.execute(text("ALTER TABLE pdf_documents ADD COLUMN IF NOT EXISTS target_price FLOAT"))
                conn.execute(text("ALTER TABLE pdf_documents ADD COLUMN IF NOT EXISTS eps_pe JSONB"))
                conn.execute(text("ALTER TABLE pdf_documents ADD COLUMN IF NOT EXISTS dense_summary TEXT"))
                conn.execute(text("ALTER TABLE pdf_documents ADD COLUMN IF NOT EXISTS email_message_id VARCHAR(500)"))
                conn.execute(text("ALTER TABLE pdf_documents ADD COLUMN IF NOT EXISTS ingest_source VARCHAR(50) DEFAULT 'upload'"))
                # Triage learning log
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS email_triage_log (
                        id SERIAL PRIMARY KEY,
                        message_id VARCHAR(64),
                        sender_email TEXT NOT NULL,
                        sender_domain TEXT NOT NULL,
                        broker TEXT,
                        subject TEXT,
                        decision VARCHAR(16) NOT NULL,
                        rejection_reason TEXT,
                        doc_id INTEGER,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_triage_sender ON email_triage_log(sender_email)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_triage_message ON email_triage_log(message_id)"))
                conn.commit()

            # pgvector index — requires the vector extension; skip gracefully if not installed
            try:
                with self.engine.connect() as conn:
                    conn.execute(text("""
                        CREATE INDEX IF NOT EXISTS pdf_chunks_embedding_idx
                        ON pdf_chunks
                        USING hnsw (embedding vector_cosine_ops);
                    """))
                    conn.commit()
            except Exception as e:
                print(f"[WARNING] Could not create pgvector index (is the 'vector' extension installed?): {e}")

            # BM25 / tsvector migration — tsvector column, trigger, GIN index
            try:
                with self.engine.connect() as conn:
                    conn.execute(text(
                        "ALTER TABLE pdf_chunks ADD COLUMN IF NOT EXISTS tsv tsvector"
                    ))
                    conn.execute(text("""
                        CREATE OR REPLACE FUNCTION pdf_chunks_tsv_update() RETURNS trigger AS $$
                        BEGIN
                            NEW.tsv := setweight(to_tsvector('english', COALESCE(NEW.verbalized_summary, '')), 'A')
                                    || setweight(to_tsvector('english', COALESCE(NEW.raw_content, '')), 'B');
                            RETURN NEW;
                        END;
                        $$ LANGUAGE plpgsql;
                    """))
                    conn.execute(text(
                        "DROP TRIGGER IF EXISTS pdf_chunks_tsv_trigger ON pdf_chunks"
                    ))
                    conn.execute(text("""
                        CREATE TRIGGER pdf_chunks_tsv_trigger
                            BEFORE INSERT OR UPDATE ON pdf_chunks
                            FOR EACH ROW EXECUTE FUNCTION pdf_chunks_tsv_update();
                    """))
                    conn.execute(text(
                        "CREATE INDEX IF NOT EXISTS pdf_chunks_tsv_gin ON pdf_chunks USING GIN(tsv)"
                    ))
                    conn.commit()
            except Exception as e:
                print(f"[WARNING] Could not create BM25 tsvector column/trigger/index: {e}")

            # Backfill existing rows that have no tsv yet
            try:
                with self.engine.connect() as conn:
                    conn.execute(text("""
                        UPDATE pdf_chunks
                        SET tsv = setweight(to_tsvector('english', COALESCE(verbalized_summary, '')), 'A')
                               || setweight(to_tsvector('english', COALESCE(raw_content, '')), 'B')
                    """))
                    conn.commit()
            except Exception as e:
                print(f"[WARNING] BM25 tsvector backfill failed (will retry on next start): {e}")

    def get_session(self):
        return self.SessionLocal()

    # -------- Document & Chunk CRUD --------

    def add_document(
        self,
        filename: str,
        file_path: str,
        total_pages: int,
        file_size_bytes: int,
        file_hash: str,
        sender_name: Optional[str] = None,
        sender_company: Optional[str] = None,
        sent_date=None,
        tickers=None,
        report_type: Optional[str] = None,
        sector: Optional[str] = None,
        asset_class: Optional[str] = None,
        coverage_period_from=None,
        coverage_period_to=None,
        broker: Optional[str] = None,
        broker_action: Optional[str] = None,
        rating: Optional[str] = None,
        target_price: Optional[float] = None,
        eps_pe=None,
        dense_summary: Optional[str] = None,
        email_message_id: Optional[str] = None,
        ingest_source: Optional[str] = "upload",
    ) -> PDFDocument:
        session = self.get_session()
        try:
            doc = PDFDocument(
                filename=filename,
                file_path=file_path,
                total_pages=total_pages,
                file_size_bytes=file_size_bytes,
                file_hash=file_hash,
                sender_name=sender_name,
                sender_company=sender_company,
                sent_date=sent_date,
                tickers=tickers,
                report_type=report_type,
                sector=sector,
                asset_class=asset_class,
                coverage_period_from=coverage_period_from,
                coverage_period_to=coverage_period_to,
                broker=broker,
                broker_action=broker_action,
                rating=rating,
                target_price=target_price,
                eps_pe=eps_pe,
                dense_summary=dense_summary,
                email_message_id=email_message_id,
                ingest_source=ingest_source,
            )
            session.add(doc)
            session.commit()
            session.refresh(doc)
            return doc
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_document_by_email_id(self, email_message_id: str) -> Optional[PDFDocument]:
        session = self.get_session()
        try:
            return session.query(PDFDocument).filter_by(email_message_id=email_message_id).first()
        finally:
            session.close()

    def get_document_by_hash(self, file_hash: str) -> Optional[PDFDocument]:
        session = self.get_session()
        try:
            return session.query(PDFDocument).filter_by(file_hash=file_hash).first()
        finally:
            session.close()
    def update_document_metadata(
        self,
        document_id: int,
        tickers=None,
        report_type: Optional[str] = None,
        sector: Optional[str] = None,
        asset_class: Optional[str] = None,
        coverage_period_from=None,
        coverage_period_to=None,
        sender_name: Optional[str] = None,
        sender_company: Optional[str] = None,
        sent_date=None,
    ) -> bool:
        """Update extended metadata fields for an existing document.
        Only overwrites a field if the supplied value is not None, so calling this
        with a partial result won't blank out fields already populated.
        Returns True if the document was found, False otherwise.
        """
        session = self.get_session()
        try:
            doc = session.query(PDFDocument).filter_by(id=document_id).first()
            if not doc:
                return False
            if tickers is not None:
                doc.tickers = tickers
            if report_type is not None:
                doc.report_type = report_type
            if sector is not None:
                doc.sector = sector
            if asset_class is not None:
                doc.asset_class = asset_class
            if coverage_period_from is not None:
                doc.coverage_period_from = coverage_period_from
            if coverage_period_to is not None:
                doc.coverage_period_to = coverage_period_to
            if sender_name is not None:
                doc.sender_name = sender_name
            if sender_company is not None:
                doc.sender_company = sender_company
            if sent_date is not None:
                doc.sent_date = sent_date
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_documents_missing_metadata(self, force: bool = False) -> List[PDFDocument]:
        """Return documents that need metadata extraction.

        Args:
            force: If True, return ALL documents regardless of whether metadata
                   is already populated (useful for re-running with an improved prompt).
        """
        session = self.get_session()
        try:
            q = session.query(PDFDocument).order_by(PDFDocument.id)
            if not force:
                q = q.filter(
                    PDFDocument.tickers.is_(None) |
                    PDFDocument.report_type.is_(None) |
                    PDFDocument.asset_class.is_(None) |
                    PDFDocument.sector.is_(None)
                )
            return q.all()
        finally:
            session.close()

    def get_document_chunk_text(self, document_id: int) -> str:
        """Return the raw_content of the document-level chunk (level=document).
        This is the full Docling markdown, usable for metadata re-extraction
        without re-parsing the original PDF file.
        """
        session = self.get_session()
        try:
            chunk = (
                session.query(PDFChunk)
                .filter(
                    PDFChunk.document_id == document_id,
                    PDFChunk.metadata_["level"].astext == "document",
                )
                .first()
            )
            return (chunk.raw_content or "") if chunk else ""
        finally:
            session.close()

    def add_chunks(self, document_id: int, chunks: List[dict]) -> List[uuid.UUID]:
        """
        Add chunks for a document. Returns list of chunk IDs (in same order as chunks).
        Each chunk dict: raw_content, verbalized_summary, metadata, image_blob (optional)
        """
        session = self.get_session()
        try:
            chunk_objects: List[PDFChunk] = []
            for c in chunks:
                obj = PDFChunk(
                    document_id=document_id,
                    raw_content=c["raw_content"],
                    verbalized_summary=c.get("verbalized_summary"),
                    page_number=c.get("metadata", {}).get("page_number"),
                    metadata_=c.get("metadata"),
                    image_blob=c.get("image_blob"),
                )
                chunk_objects.append(obj)
                session.add(obj)

            doc = session.query(PDFDocument).filter_by(id=document_id).first()
            if doc:
                doc.processed_at = datetime.utcnow()

            session.commit()
            # Return IDs while session is still open so callers don't touch detached objects
            return [obj.id for obj in chunk_objects]
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_document_by_filename(self, filename: str) -> Optional[PDFDocument]:
        session = self.get_session()
        try:
            return session.query(PDFDocument).filter_by(filename=filename).first()
        finally:
            session.close()

    def delete_document(self, document_id: int) -> bool:
        """
        Delete a document and all its chunks (cascade). Returns True if deleted, False if not found.
        """
        session = self.get_session()
        try:
            doc = session.query(PDFDocument).filter_by(id=document_id).first()
            if not doc:
                return False
            session.delete(doc)
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def delete_all_documents(self) -> int:
        """Delete all documents and their chunks (cascade). Returns number of documents deleted."""
        session = self.get_session()
        try:
            docs = session.query(PDFDocument).all()
            n = len(docs)
            for doc in docs:
                session.delete(doc)
            session.commit()
            return n
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # -------- Triage Learning --------

    def log_triage_decision(
        self,
        message_id: str,
        sender_email: str,
        sender_domain: str,
        broker: Optional[str],
        subject: Optional[str],
        decision: str,
        rejection_reason: Optional[str] = None,
        doc_id: Optional[int] = None,
    ) -> None:
        """Record one triage decision to the learning log."""
        session = self.get_session()
        try:
            entry = EmailTriageLog(
                message_id=message_id,
                sender_email=sender_email.lower(),
                sender_domain=sender_domain.lower(),
                broker=broker,
                subject=subject,
                decision=decision,
                rejection_reason=rejection_reason,
                doc_id=doc_id,
            )
            session.add(entry)
            session.commit()
        except Exception:
            session.rollback()
        finally:
            session.close()

    def get_sender_reputation(self, sender_email: str) -> dict:
        """
        Return a reputation summary for a sender based on their triage history.

        Fields:
          total        — total emails seen from this sender
          accepted     — emails that passed triage and were ingested
          rejected     — emails that were blocked
          auto_block   — True when sender has 3+ decisions and zero acceptances
          trusted      — True when sender has 5+ acceptances and <20% rejection rate
        """
        session = self.get_session()
        try:
            rows = session.execute(
                text("""
                    SELECT decision, COUNT(*) as cnt
                    FROM email_triage_log
                    WHERE sender_email = :email
                    GROUP BY decision
                """),
                {"email": sender_email.lower()},
            ).fetchall()
        finally:
            session.close()

        accepted = next((r[1] for r in rows if r[0] == "accepted"), 0)
        rejected = next((r[1] for r in rows if r[0] == "rejected"), 0)
        total = accepted + rejected

        auto_block = total >= 3 and accepted == 0
        trusted = accepted >= 5 and (rejected / total < 0.2 if total > 0 else False)

        return {
            "total": total,
            "accepted": accepted,
            "rejected": rejected,
            "auto_block": auto_block,
            "trusted": trusted,
        }

    def get_rejected_message_ids(self) -> set:
        """Return all message_ids that were previously rejected (for fetch-time dedup)."""
        session = self.get_session()
        try:
            rows = session.execute(
                text("SELECT message_id FROM email_triage_log WHERE decision = 'rejected' AND message_id IS NOT NULL")
            ).fetchall()
            return {r[0] for r in rows}
        finally:
            session.close()

    def get_chunks_by_document(self, document_id: int) -> List[PDFChunk]:
        session = self.get_session()
        try:
            return (
                session.query(PDFChunk)
                .filter_by(document_id=document_id)
                .order_by(PDFChunk.created_at)
                .all()
            )
        finally:
            session.close()

    def get_chunk_by_id(self, chunk_id: uuid.UUID) -> Optional[PDFChunk]:
        session = self.get_session()
        try:
            return (
                session.query(PDFChunk)
                .options(joinedload(PDFChunk.document))
                .filter_by(id=chunk_id)
                .first()
            )
        finally:
            session.close()

    def update_chunk_metadata(self, chunk_id: uuid.UUID, metadata: Dict[str, Any]) -> None:
        """Update the metadata JSONB for a chunk (merge with existing)."""
        session = self.get_session()
        try:
            chunk = session.query(PDFChunk).filter_by(id=chunk_id).first()
            if chunk:
                existing = dict(chunk.metadata_ or {})
                existing.update(metadata)
                chunk.metadata_ = existing
                session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # -------- Embeddings & Vector Search --------

    def upsert_chunk_embedding(self, chunk_id: uuid.UUID, embedding: Sequence[float]) -> None:
        session = self.get_session()
        try:
            chunk = session.query(PDFChunk).filter_by(id=chunk_id).first()
            if chunk:
                chunk.embedding = list(embedding)
                session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def clear_all_embeddings(self) -> int:
        """NULL every chunk's embedding so a full re-embed reprocesses the whole corpus.
        Used when switching embedding models — old vectors aren't comparable to new ones.
        Returns the number of rows cleared."""
        session = self.get_session()
        try:
            n = session.query(PDFChunk).update(
                {PDFChunk.embedding: None}, synchronize_session=False
            )
            session.commit()
            return n
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_chunks_without_embedding(self, limit: int = 100) -> List[PDFChunk]:
        session = self.get_session()
        try:
            return (
                session.query(PDFChunk)
                .filter(PDFChunk.embedding.is_(None))
                .order_by(PDFChunk.created_at)
                .limit(limit)
                .all()
            )
        finally:
            session.close()

    def semantic_search_chunks(
        self,
        query_embedding: Sequence[float],
        limit: int = 20,
        document_ids: Optional[Sequence[int]] = None,
        filenames: Optional[Sequence[str]] = None,
        page_min: Optional[int] = None,
        page_max: Optional[int] = None,
        sender_names: Optional[Sequence[str]] = None,
        sender_companies: Optional[Sequence[str]] = None,
        written_date_from=None,
        written_date_to=None,
        tickers: Optional[Sequence[str]] = None,
        report_type: Optional[str] = None,
        sector: Optional[str] = None,
        asset_class: Optional[str] = None,
        coverage_period_from=None,
        coverage_period_to=None,
        similarity_threshold: float = 0.0,
    ) -> List[PDFChunk]:
        """Vector search over verbalized_summary embeddings.

        Args:
            similarity_threshold: Minimum cosine similarity (0–1) a chunk must score
                to be returned. Cosine distance = 1 − similarity, so a threshold of
                0.4 discards anything with distance > 0.6. Defaults to 0.0 (no filter).
            tickers: Return only chunks from documents whose tickers array contains
                any of the given ticker symbols (OR logic).
            coverage_period_from/to: Overlap filter — returns documents whose coverage
                period overlaps the requested range. A document covering Jan–Mar will
                NOT match a Jul–Sep query.
        """
        session = self.get_session()
        try:
            embedding_list = list(query_embedding)
            distance_col = PDFChunk.embedding.cosine_distance(embedding_list)

            q = session.query(PDFChunk).join(PDFDocument).options(joinedload(PDFChunk.document))

            if document_ids:
                q = q.filter(PDFChunk.document_id.in_(document_ids))
            if filenames:
                q = q.filter(PDFDocument.filename.in_(filenames))
            if page_min is not None:
                q = q.filter(PDFChunk.page_number >= page_min)
            if page_max is not None:
                q = q.filter(PDFChunk.page_number <= page_max)
            if sender_names:
                q = q.filter(PDFDocument.sender_name.in_(sender_names))
            if sender_companies:
                q = q.filter(_sender_company_filter(sender_companies))
            if written_date_from is not None or written_date_to is not None:
                # Use COALESCE(written_date, sent_date) so documents where only
                # sent_date is populated (the common case from the ingestion pipeline)
                # are still matched by date filters.
                effective_date = func.coalesce(PDFDocument.written_date, PDFDocument.sent_date)
                if written_date_from is not None:
                    q = q.filter(effective_date >= written_date_from)
                if written_date_to is not None:
                    q = q.filter(effective_date <= written_date_to)

            # ── Extended metadata filters ──────────────────────────────────────
            if tickers:
                # Partial text match: cast the JSONB array to text and use ILIKE.
                # This means "9958" matches "9958 TT", "BTC" matches "BTC" or "XBTC", etc.
                # OR logic across all requested tickers.
                q = q.filter(or_(*[
                    cast(PDFDocument.tickers, Text).ilike(f"%{t}%")
                    for t in tickers
                ]))
            if report_type:
                q = q.filter(PDFDocument.report_type == report_type)
            if sector:
                q = q.filter(PDFDocument.sector.ilike(f"%{sector}%"))
            if asset_class:
                q = q.filter(PDFDocument.asset_class == asset_class)
            if coverage_period_from is not None or coverage_period_to is not None:
                # Overlap: doc's coverage window must intersect the requested range.
                # Documents with NULL coverage_period are treated as "unknown" and included
                # rather than excluded — semantic search will determine their relevance.
                # Condition: (doc.to IS NULL OR doc.to >= requested_from)
                #        AND (doc.from IS NULL OR doc.from <= requested_to)
                if coverage_period_from is not None:
                    q = q.filter(
                        PDFDocument.coverage_period_to.is_(None) |
                        (PDFDocument.coverage_period_to >= coverage_period_from)
                    )
                if coverage_period_to is not None:
                    q = q.filter(
                        PDFDocument.coverage_period_from.is_(None) |
                        (PDFDocument.coverage_period_from <= coverage_period_to)
                    )

            if similarity_threshold > 0.0:
                q = q.filter(distance_col <= (1.0 - similarity_threshold))

            q = q.order_by(distance_col)
            return q.limit(limit).all()
        finally:
            session.close()

    def _bm25_search_chunks(
        self,
        query_text: str,
        limit: int = 100,
        document_ids: Optional[Sequence[int]] = None,
        filenames: Optional[Sequence[str]] = None,
        page_min: Optional[int] = None,
        page_max: Optional[int] = None,
        sender_names: Optional[Sequence[str]] = None,
        sender_companies: Optional[Sequence[str]] = None,
        written_date_from=None,
        written_date_to=None,
        tickers: Optional[Sequence[str]] = None,
        report_type: Optional[str] = None,
        sector: Optional[str] = None,
        asset_class: Optional[str] = None,
        coverage_period_from=None,
        coverage_period_to=None,
    ) -> List["PDFChunk"]:
        """BM25 full-text search using PostgreSQL tsvector / websearch_to_tsquery.

        Returns chunks ranked by ts_rank_cd. Falls back to [] if the tsvector
        infrastructure is not yet installed (graceful degradation).
        """
        if not (query_text or "").strip():
            return []

        # Build WHERE clause fragments and parameter dict for metadata filters.
        where_clauses = [
            "c.tsv IS NOT NULL",
            "c.tsv @@ websearch_to_tsquery('english', :_bm25_q_text)",
        ]
        params: dict = {"_bm25_q_text": query_text, "_bm25_limit": limit}

        if document_ids:
            placeholders = ", ".join(f":_bm25_doc_id_{i}" for i in range(len(document_ids)))
            where_clauses.append(f"c.document_id IN ({placeholders})")
            for i, did in enumerate(document_ids):
                params[f"_bm25_doc_id_{i}"] = did

        if filenames:
            placeholders = ", ".join(f":_bm25_fn_{i}" for i in range(len(filenames)))
            where_clauses.append(f"d.filename IN ({placeholders})")
            for i, fn in enumerate(filenames):
                params[f"_bm25_fn_{i}"] = fn

        if page_min is not None:
            where_clauses.append("c.page_number >= :_bm25_page_min")
            params["_bm25_page_min"] = page_min

        if page_max is not None:
            where_clauses.append("c.page_number <= :_bm25_page_max")
            params["_bm25_page_max"] = page_max

        if sender_names:
            placeholders = ", ".join(f":_bm25_sn_{i}" for i in range(len(sender_names)))
            where_clauses.append(f"d.sender_name IN ({placeholders})")
            for i, sn in enumerate(sender_names):
                params[f"_bm25_sn_{i}"] = sn

        if sender_companies:
            company_clauses = []
            for i, c_name in enumerate(sender_companies):
                params[f"_bm25_sc_{i}"] = f"%{c_name}%"
                params[f"_bm25_sc_rev_{i}"] = f"%{c_name}%"
                params[f"_bm25_sc_val_{i}"] = c_name
                company_clauses.append(
                    f"(d.sender_company IS NOT NULL AND "
                    f"(d.sender_company ILIKE :_bm25_sc_{i} OR "
                    f":_bm25_sc_val_{i} ILIKE '%' || d.sender_company || '%'))"
                )
            where_clauses.append(f"({' OR '.join(company_clauses)})")

        if written_date_from is not None:
            where_clauses.append("COALESCE(d.written_date, d.sent_date) >= :_bm25_wdf")
            params["_bm25_wdf"] = written_date_from

        if written_date_to is not None:
            where_clauses.append("COALESCE(d.written_date, d.sent_date) <= :_bm25_wdt")
            params["_bm25_wdt"] = written_date_to

        if tickers:
            ticker_clauses = []
            for i, t in enumerate(tickers):
                params[f"_bm25_tk_{i}"] = f"%{t}%"
                ticker_clauses.append(f"CAST(d.tickers AS TEXT) ILIKE :_bm25_tk_{i}")
            where_clauses.append(f"({' OR '.join(ticker_clauses)})")

        if report_type:
            where_clauses.append("d.report_type = :_bm25_report_type")
            params["_bm25_report_type"] = report_type

        if sector:
            where_clauses.append("d.sector ILIKE :_bm25_sector")
            params["_bm25_sector"] = f"%{sector}%"

        if asset_class:
            where_clauses.append("d.asset_class = :_bm25_asset_class")
            params["_bm25_asset_class"] = asset_class

        if coverage_period_from is not None:
            where_clauses.append(
                "(d.coverage_period_to IS NULL OR d.coverage_period_to >= :_bm25_cpf)"
            )
            params["_bm25_cpf"] = coverage_period_from

        if coverage_period_to is not None:
            where_clauses.append(
                "(d.coverage_period_from IS NULL OR d.coverage_period_from <= :_bm25_cpt)"
            )
            params["_bm25_cpt"] = coverage_period_to

        where_sql = " AND ".join(where_clauses)
        sql = text(f"""
            SELECT c.id
            FROM pdf_chunks c
            JOIN pdf_documents d ON d.id = c.document_id
            WHERE {where_sql}
            ORDER BY ts_rank_cd(c.tsv, websearch_to_tsquery('english', :_bm25_q_text), 32) DESC
            LIMIT :_bm25_limit
        """)

        try:
            session = self.get_session()
            try:
                rows = session.execute(sql, params).fetchall()
                chunk_ids = [r[0] for r in rows]
            finally:
                session.close()
        except Exception as e:
            err_str = str(e).lower()
            if any(kw in err_str for kw in ("tsv", "tsvector", "websearch_to_tsquery")):
                print(f"[WARNING] BM25 search unavailable (migration not applied?): {e}")
                return []
            raise

        if not chunk_ids:
            return []

        # Load ORM objects in BM25 rank order
        session = self.get_session()
        try:
            chunks_unordered = (
                session.query(PDFChunk)
                .options(joinedload(PDFChunk.document))
                .filter(PDFChunk.id.in_(chunk_ids))
                .all()
            )
        finally:
            session.close()

        id_to_chunk = {c.id: c for c in chunks_unordered}
        return [id_to_chunk[cid] for cid in chunk_ids if cid in id_to_chunk]

    def hybrid_search_chunks(
        self,
        query_embedding: Sequence[float],
        query_text: str,
        limit: int = 40,
        rrf_k: int = 60,
        similarity_threshold: float = 0.0,
        candidate_k: int = 100,
        document_ids: Optional[Sequence[int]] = None,
        filenames: Optional[Sequence[str]] = None,
        page_min: Optional[int] = None,
        page_max: Optional[int] = None,
        sender_names: Optional[Sequence[str]] = None,
        sender_companies: Optional[Sequence[str]] = None,
        written_date_from=None,
        written_date_to=None,
        tickers: Optional[Sequence[str]] = None,
        report_type: Optional[str] = None,
        sector: Optional[str] = None,
        asset_class: Optional[str] = None,
        coverage_period_from=None,
        coverage_period_to=None,
    ) -> List["PDFChunk"]:
        """Hybrid vector + BM25 retrieval fused with Reciprocal Rank Fusion (RRF)."""
        filter_kwargs = dict(
            document_ids=document_ids,
            filenames=filenames,
            page_min=page_min,
            page_max=page_max,
            sender_names=sender_names,
            sender_companies=sender_companies,
            written_date_from=written_date_from,
            written_date_to=written_date_to,
            tickers=tickers,
            report_type=report_type,
            sector=sector,
            asset_class=asset_class,
            coverage_period_from=coverage_period_from,
            coverage_period_to=coverage_period_to,
        )

        vec_chunks = self.semantic_search_chunks(
            query_embedding=query_embedding,
            limit=candidate_k,
            similarity_threshold=similarity_threshold,
            **filter_kwargs,
        )

        bm25_chunks = self._bm25_search_chunks(
            query_text=query_text,
            limit=candidate_k,
            **filter_kwargs,
        )

        if not bm25_chunks:
            print(f"[hybrid] BM25 returned 0 — using vector-only ({len(vec_chunks)} chunks)")
            return vec_chunks[:limit]

        # Reciprocal Rank Fusion
        vec_rank = {c.id: i + 1 for i, c in enumerate(vec_chunks)}
        bm25_rank = {c.id: i + 1 for i, c in enumerate(bm25_chunks)}
        all_ids = set(vec_rank) | set(bm25_rank)

        scored = sorted(
            all_ids,
            key=lambda cid: (
                (1.0 / (rrf_k + vec_rank[cid]) if cid in vec_rank else 0.0) +
                (1.0 / (rrf_k + bm25_rank[cid]) if cid in bm25_rank else 0.0)
            ),
            reverse=True,
        )[:limit]

        all_chunks = {c.id: c for c in vec_chunks + bm25_chunks}
        result = [all_chunks[cid] for cid in scored if cid in all_chunks]

        both = sum(1 for cid in scored if cid in vec_rank and cid in bm25_rank)
        vec_only = sum(1 for cid in scored if cid in vec_rank and cid not in bm25_rank)
        bm25_only = sum(1 for cid in scored if cid in bm25_rank and cid not in vec_rank)
        print(f"[hybrid] {len(result)} chunks: {both} both, {vec_only} vec-only, {bm25_only} bm25-only")

        return result

    def list_documents_filtered(
        self,
        document_ids: Optional[Sequence[int]] = None,
        filenames: Optional[Sequence[str]] = None,
        sender_names: Optional[Sequence[str]] = None,
        sender_companies: Optional[Sequence[str]] = None,
        written_date_from=None,
        written_date_to=None,
        tickers: Optional[Sequence[str]] = None,
        report_type: Optional[str] = None,
        sector: Optional[str] = None,
        asset_class: Optional[str] = None,
        coverage_period_from=None,
        coverage_period_to=None,
    ) -> List[PDFDocument]:
        """Return all PDFDocument rows matching metadata filters, with no limit.

        Mirrors the filter logic of semantic_search_chunks but queries PDFDocument
        directly — no embedding, no similarity threshold, no chunk join.
        Used by the list_documents query path to return an uncapped document inventory.
        """
        session = self.get_session()
        try:
            q = session.query(PDFDocument)

            if document_ids:
                q = q.filter(PDFDocument.id.in_(document_ids))
            if filenames:
                q = q.filter(PDFDocument.filename.in_(filenames))
            if sender_names:
                q = q.filter(PDFDocument.sender_name.in_(sender_names))
            if sender_companies:
                q = q.filter(_sender_company_filter(sender_companies))
            if written_date_from is not None or written_date_to is not None:
                effective_date = func.coalesce(PDFDocument.written_date, PDFDocument.sent_date)
                if written_date_from is not None:
                    q = q.filter(effective_date >= written_date_from)
                if written_date_to is not None:
                    q = q.filter(effective_date <= written_date_to)
            if tickers:
                q = q.filter(or_(*[
                    cast(PDFDocument.tickers, Text).ilike(f"%{t}%")
                    for t in tickers
                ]))
            if report_type:
                q = q.filter(PDFDocument.report_type == report_type)
            if sector:
                q = q.filter(PDFDocument.sector.ilike(f"%{sector}%"))
            if asset_class:
                q = q.filter(PDFDocument.asset_class == asset_class)
            if coverage_period_from is not None or coverage_period_to is not None:
                if coverage_period_from is not None:
                    q = q.filter(
                        PDFDocument.coverage_period_to.is_(None) |
                        (PDFDocument.coverage_period_to >= coverage_period_from)
                    )
                if coverage_period_to is not None:
                    q = q.filter(
                        PDFDocument.coverage_period_from.is_(None) |
                        (PDFDocument.coverage_period_from <= coverage_period_to)
                    )

            q = q.order_by(PDFDocument.written_date.desc(), PDFDocument.id.desc())
            return q.all()
        finally:
            session.close()

    def get_metadata_for_point_fact(
        self,
        sender_companies: Optional[Sequence[str]] = None,
        tickers: Optional[Sequence[str]] = None,
        written_date_from=None,
        written_date_to=None,
        limit: int = 10,
    ) -> List[dict]:
        """Return structured metadata rows for point-fact queries (no chunk retrieval).

        Returns the most recent matching documents' rating, target_price,
        broker_action, dense_summary, and filename — enough to answer questions
        like "What is TSMC's current target price?" without embedding or generation.
        """
        session = self.get_session()
        try:
            q = session.query(PDFDocument)
            if sender_companies:
                q = q.filter(_sender_company_filter(sender_companies))
            if tickers:
                q = q.filter(or_(*[
                    cast(PDFDocument.tickers, Text).ilike(f"%{t}%")
                    for t in tickers
                ]))
            if written_date_from is not None:
                effective_date = func.coalesce(PDFDocument.written_date, PDFDocument.sent_date)
                q = q.filter(effective_date >= written_date_from)
            if written_date_to is not None:
                effective_date = func.coalesce(PDFDocument.written_date, PDFDocument.sent_date)
                q = q.filter(effective_date <= written_date_to)
            q = q.order_by(PDFDocument.written_date.desc(), PDFDocument.id.desc()).limit(limit)
            rows = q.all()
            return [
                {
                    "filename": doc.filename,
                    "broker": doc.sender_company or doc.sender_name,
                    "broker_action": doc.broker_action,
                    "rating": doc.rating,
                    "target_price": doc.target_price,
                    "written_date": doc.written_date.isoformat() if doc.written_date else None,
                    "dense_summary": doc.dense_summary,
                    "tickers": doc.tickers,
                }
                for doc in rows
            ]
        finally:
            session.close()
