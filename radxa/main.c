/* main.c — entry point (pure C) for the NPU full-body detector.
 *
 * Parse CLI, install a SIGINT handler, then hand off to the C++/OpenCV+QNN
 * core through the C API in detect.h. The heavy lifting (YOLOv8n on the
 * Hexagon NPU + camera + MJPEG stream) lives behind this boundary. */
#include "detect.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

static void on_sigint(int sig) { (void)sig; dh_request_stop(); }

static void usage(const char* prog) {
    printf("Usage: %s [options]\n"
           "  --device PATH     camera device        (default: auto-detect ASJ ZNX)\n"
           "  --size WxH        capture size         (default 640x642)\n"
           "  --port N          MJPEG stream port    (default 8092)\n"
           "  --weights DIR     folder with yolov8n_qcs6490.bin + libQnnHtp.so (default ./weights)\n"
           "  --qairt DIR       QAIRT SDK root        (default ~/qairt/2.42.0.251225)\n"
           "  --conf F          detection threshold   (default 0.25)\n"
           "  --every N         detect every N frames (default 1)\n"
           "  --rotate N        rotate 0/90/180/270   (default 180)\n"
           "  --motion N        motion min area px, 0=off (default 1500)\n"
           "  -h, --help        this help\n", prog);
}

int main(int argc, char** argv) {
    dh_config cfg;
    cfg.device = "";         /* "" -> auto-detect the HP60C node by card name */
    cfg.width = 640;
    cfg.height = 642;
    cfg.port = 8092;
    cfg.weights_dir = "./weights";
    cfg.qairt_root = NULL;   /* NULL -> detect.cpp fills in $HOME/qairt/2.42.0.251225 */
    cfg.conf = 0.25f;
    cfg.detect_every = 1;
    cfg.rotate = 180;
    cfg.motion_min_area = 1500;
    cfg.annotate = 1;
    cfg.attrib = 1;            /* per-person gender/age/height/weight (M5) */
    cfg.head_cm = 23.0;        /* height-estimate calibration knob */
    cfg.bmi = 22.0;            /* weight-estimate calibration knob */
    cfg.age_offset = -5;       /* genderage reads high; correct down 5 years */
    cfg.focal = 570.0;         /* HP60C RGB fy (px) — real depth-based height */

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) cfg.device = argv[++i];
        else if (!strcmp(argv[i], "--size") && i + 1 < argc) {
            int w, h; if (sscanf(argv[++i], "%dx%d", &w, &h) == 2) { cfg.width = w; cfg.height = h; }
        }
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) cfg.port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--weights") && i + 1 < argc) cfg.weights_dir = argv[++i];
        else if (!strcmp(argv[i], "--qairt") && i + 1 < argc) cfg.qairt_root = argv[++i];
        else if (!strcmp(argv[i], "--conf") && i + 1 < argc) cfg.conf = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--every") && i + 1 < argc) cfg.detect_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rotate") && i + 1 < argc) cfg.rotate = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--motion") && i + 1 < argc) cfg.motion_min_area = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--annotate") && i + 1 < argc) cfg.annotate = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--attrib") && i + 1 < argc) cfg.attrib = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--head-cm") && i + 1 < argc) cfg.head_cm = atof(argv[++i]);
        else if (!strcmp(argv[i], "--bmi") && i + 1 < argc) cfg.bmi = atof(argv[++i]);
        else if (!strcmp(argv[i], "--age-offset") && i + 1 < argc) cfg.age_offset = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--focal") && i + 1 < argc) cfg.focal = atof(argv[++i]);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
    }
    if (cfg.detect_every < 1) cfg.detect_every = 1;

    setvbuf(stdout, NULL, _IOLBF, 0);   /* live logs even when piped to a file */

    signal(SIGINT, on_sigint);
    signal(SIGTERM, on_sigint);
    signal(SIGPIPE, SIG_IGN);           /* survive a browser disconnecting mid-stream */

    printf("detect_human (NPU): device=%s size=%dx%d port=%d weights=%s conf=%.2f\n",
           cfg.device[0] ? cfg.device : "auto", cfg.width, cfg.height, cfg.port,
           cfg.weights_dir, cfg.conf);
    return dh_run(&cfg);
}
