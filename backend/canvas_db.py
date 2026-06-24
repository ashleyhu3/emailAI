"""
Canvas persistence models.

Import this module before instantiating DatabaseManager so that Canvas and
CanvasState are registered on Base and created by Base.metadata.create_all().
"""
import sys
import os
import uuid
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "PDF_summarizer"))

from database import Base  # reuse the same declarative base

from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship


class Canvas(Base):
    __tablename__ = "canvases"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(500), nullable=False, default="Untitled Canvas")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    state = relationship(
        "CanvasState",
        uselist=False,
        back_populates="canvas",
        cascade="all, delete-orphan",
    )


class CanvasState(Base):
    __tablename__ = "canvas_state"

    canvas_id = Column(UUID(as_uuid=True), ForeignKey("canvases.id"), primary_key=True)
    nodes_json = Column(JSONB, nullable=False, default=list)
    edges_json = Column(JSONB, nullable=False, default=list)

    canvas = relationship("Canvas", back_populates="state")
