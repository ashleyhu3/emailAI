#!/usr/bin/env bash
# ── Broker email ingest worker ────────────────────────────────────────────────
# Starts the Celery worker + beat scheduler from the repo root.
# Requires Redis running locally (brew install redis && brew services start redis).
#
# Usage:
#   chmod +x start_worker.sh
#   ./start_worker.sh
#
# Logs go to logs/worker.log and logs/beat.log.
# Stop with:  kill $(cat logs/worker.pid) $(cat logs/beat.pid)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

mkdir -p logs

# Load .env so Celery workers inherit all required vars
set -a
source .env
set +a

echo "[start_worker] Checking Redis..."
if ! redis-cli ping > /dev/null 2>&1; then
    echo "ERROR: Redis is not running. Start it with: brew services start redis"
    exit 1
fi

echo "[start_worker] Starting Celery worker..."
celery -A PDF_summarizer.ingest.worker worker \
    --loglevel=info \
    --concurrency=2 \
    --logfile=logs/worker.log \
    --pidfile=logs/worker.pid \
    --detach

echo "[start_worker] Starting Celery beat scheduler (07:00 + 18:00 UTC)..."
celery -A PDF_summarizer.ingest.worker beat \
    --loglevel=info \
    --logfile=logs/beat.log \
    --pidfile=logs/beat.pid \
    --detach

echo ""
echo "Worker and scheduler started."
echo "  Worker log:    $REPO_ROOT/logs/worker.log"
echo "  Beat log:      $REPO_ROOT/logs/beat.log"
echo ""
echo "To run a manual ingest right now:"
echo "  curl -X POST http://localhost:8000/ingest/trigger"
echo ""
echo "To test your email connection:"
echo "  curl http://localhost:8000/ingest/test-email"
echo ""
echo "To stop both processes:"
echo "  kill \$(cat logs/worker.pid) \$(cat logs/beat.pid)"
