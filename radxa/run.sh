#!/bin/sh
# run.sh — launch the NPU full-body detector (detect_human).
#
# The camera is owned by the separate `ascamera` service (Angstrong depth SDK),
# which publishes RGB + depth to /dev/shm/ascamera_*. This detector reads those,
# runs YOLO + attributes on the NPU, and serves the annotated RGB + colormapped
# depth (/depth) + detection JSON (/data). So there is no camera/v4l2 setup here.
#
# NOTE: the HTP/fastRPC layer locates the DSP skel (libQnnHtpV68Skel.so) via the
# CURRENT WORKING DIRECTORY, so we cd into ./weights before launching.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
QAIRT="${QAIRT_ROOT:-$HOME/qairt/2.42.0.251225}"
WEIGHTS="$HERE/weights"

export LD_LIBRARY_PATH="$QAIRT/lib/aarch64-oe-linux-gcc11.2:$QAIRT/lib/aarch64-ubuntu-gcc9.4:$LD_LIBRARY_PATH"

# Reap any previous instance.
pkill -9 -f 'build/detect_human' 2>/dev/null || true
sleep 1

cd "$WEIGHTS"   # skel is found relative to CWD
# --rotate 0: the ascamera SDK delivers frames upright (unlike the old v4l2 path
# which needed 180). RGB + depth are rotated together, so 0 keeps both correct.
exec "$HERE/build/detect_human" \
  --size 640x480 --port 8092 --rotate 0 \
  --weights "$WEIGHTS" --qairt "$QAIRT" "$@"
