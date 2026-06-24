# utils.py
import hashlib
from pathlib import Path

def get_file_hash(file_path: str, chunk_size: int = 8192) -> str:
    """Return SHA256 hash of a file."""
    h = hashlib.sha256()
    file_path = Path(file_path)
    with file_path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()