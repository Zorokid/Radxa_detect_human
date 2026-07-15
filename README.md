# Radxa detect-human

Real-time full-body detection + depth on a Radxa Dragon Q6A (Qualcomm QCS6490,
Hexagon NPU) with an Angstrong/Nuwa HP60C RGB-D camera, streamed to an Orange Pi
web host.

## Pipeline

```
HP60C (USB RGB-D) ──> Radxa (NPU detect + depth) ──[LAN]──> Orange Pi (web UI)
```

- **Radxa** (`radxa/`): captures RGB + depth from the camera (via the Angstrong
  SDK exporting to `/dev/shm`), runs **YOLOv8n** (COCO 80: person / animals /
  objects) on the Hexagon NPU, adds motion (MOG2) and per-person attributes
  (**SCRFD** + **genderage**, plus depth-based height/weight), and serves an
  MJPEG stream, a colormapped depth stream (`/depth`), and detection JSON
  (`/data`) over HTTP on `:8092`.
- **Orange Pi** (`orangepi/`): a stdlib HTTP host (`:8090`) that proxies the
  Radxa's video + data and serves a single-page UI — live FPS, a 3-view switch
  (RGB / night / LiDAR-depth), a per-category show toggle (animals / unknown /
  devices), and per-attribute toggles (gender / age / height / weight) drawn
  client-side over the raw video.

## Build (on the Radxa)

```sh
cd radxa
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target detect_human -j4
sh run.sh                 # serves http://<radxa>:8092/
```

Requires the QAIRT SDK and the model context binaries in `radxa/weights/`
(`yolov8n_qcs6490.bin`, `det_10g_qcs6490.bin`, `genderage_qcs6490.bin`) plus the
QNN HTP runtime libs — none are committed here.

## Run the web host (on the Orange Pi)

```sh
cd orangepi
RADXA_URL=http://<radxa-ip>:8092 python3 -u server.py   # serves http://<host>:8090/
```

## Model conversion

`convert/` holds the YOLOv8n → QNN context-binary recipe (ONNX export with the
detection head split before the Concat so box and score tensors get separate
INT8 scales — otherwise class probabilities quantize to zero).

## Services

`radxa/*.service` and `orangepi/*.service` are systemd units for autostart on
boot (`ascamera` → `detect_human` on the Radxa; `detect_webview` on the Orange Pi).
