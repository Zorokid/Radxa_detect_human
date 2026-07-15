#!/bin/sh
# selftest.sh — SAFE on-board test: runs the detector for a bounded time,
# grabs one annotated frame locally, then auto-stops. ALWAYS use this to test
# over SSH — camera capture saturates the WiFi-over-USB uplink and drops SSH,
# so a bare `run.sh` with no timeout ORPHANS the detector and deadlocks the
# board (only a power-cycle recovers it). This wrapper self-terminates.
#
# Usage: sh selftest.sh [run_seconds] [grab_at_seconds]
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
DUR="${1:-25}"
GRAB_AT="${2:-16}"

# Detached grabber: waits for warmup, pulls one frame over localhost.
( sleep "$GRAB_AT"; python3 "$HERE/grab.py" 8092 /tmp/shot.jpg ) >/tmp/grab.log 2>&1 &

# Bounded run — timeout guarantees the camera is released even if SSH drops.
timeout "$DUR" sh "$HERE/run.sh" >/tmp/dh.log 2>&1 || true

# Belt-and-suspenders reap.
pkill -9 -f build/detect_human 2>/dev/null || true
pkill -9 -f v4l2-ctl 2>/dev/null || true
echo "selftest done; log=/tmp/dh.log shot=/tmp/shot.jpg"
