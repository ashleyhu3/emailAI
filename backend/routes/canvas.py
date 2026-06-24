"""Canvas persistence routes."""
import uuid
from datetime import datetime
from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from dependencies import get_db
from database import DatabaseManager
from canvas_db import Canvas, CanvasState
from models import CanvasMeta, CanvasCreateRequest, CanvasSaveRequest, CanvasDetail

router = APIRouter()


@router.get("", response_model=List[CanvasMeta])
def list_canvases(db: DatabaseManager = Depends(get_db)):
    session = db.get_session()
    try:
        canvases = session.query(Canvas).order_by(Canvas.updated_at.desc()).all()
        return [
            CanvasMeta(
                id=c.id,
                name=c.name,
                created_at=c.created_at,
                updated_at=c.updated_at,
            )
            for c in canvases
        ]
    finally:
        session.close()


@router.post("", response_model=CanvasMeta, status_code=201)
def create_canvas(
    req: CanvasCreateRequest,
    db: DatabaseManager = Depends(get_db),
):
    session = db.get_session()
    try:
        canvas = Canvas(name=req.name)
        session.add(canvas)
        session.flush()  # get canvas.id before adding state

        state = CanvasState(canvas_id=canvas.id, nodes_json=[], edges_json=[])
        session.add(state)
        session.commit()
        session.refresh(canvas)

        return CanvasMeta(
            id=canvas.id,
            name=canvas.name,
            created_at=canvas.created_at,
            updated_at=canvas.updated_at,
        )
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.get("/{canvas_id}", response_model=CanvasDetail)
def get_canvas(canvas_id: UUID, db: DatabaseManager = Depends(get_db)):
    session = db.get_session()
    try:
        canvas = session.query(Canvas).filter_by(id=canvas_id).first()
        if not canvas:
            raise HTTPException(status_code=404, detail="Canvas not found")

        state = canvas.state
        return CanvasDetail(
            id=canvas.id,
            name=canvas.name,
            created_at=canvas.created_at,
            updated_at=canvas.updated_at,
            nodes=state.nodes_json if state else [],
            edges=state.edges_json if state else [],
        )
    finally:
        session.close()


@router.put("/{canvas_id}", response_model=CanvasMeta)
def save_canvas(
    canvas_id: UUID,
    req: CanvasSaveRequest,
    db: DatabaseManager = Depends(get_db),
):
    session = db.get_session()
    try:
        canvas = session.query(Canvas).filter_by(id=canvas_id).first()
        if not canvas:
            raise HTTPException(status_code=404, detail="Canvas not found")

        if req.name is not None:
            canvas.name = req.name
        canvas.updated_at = datetime.utcnow()

        state = canvas.state
        if state is None:
            state = CanvasState(canvas_id=canvas.id)
            session.add(state)

        state.nodes_json = req.nodes
        state.edges_json = req.edges

        session.commit()
        session.refresh(canvas)

        return CanvasMeta(
            id=canvas.id,
            name=canvas.name,
            created_at=canvas.created_at,
            updated_at=canvas.updated_at,
        )
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.delete("/{canvas_id}")
def delete_canvas(canvas_id: UUID, db: DatabaseManager = Depends(get_db)):
    session = db.get_session()
    try:
        canvas = session.query(Canvas).filter_by(id=canvas_id).first()
        if not canvas:
            raise HTTPException(status_code=404, detail="Canvas not found")
        session.delete(canvas)
        session.commit()
        return {"deleted": True, "canvas_id": str(canvas_id)}
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
