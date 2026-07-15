/* detect.h — C API for the NPU full-body detector.
 *
 * main.c (pure C) fills this config and calls dh_run(); all the OpenCV +
 * QNN/Hexagon-NPU work lives behind this boundary in detect.cpp. Detection is
 * YOLOv8n (COCO 80 classes: person + animals + objects) on the Hexagon NPU. */
#ifndef DH_APP_H
#define DH_APP_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char* device;      /* camera device; "" or NULL -> auto-detect ASJ ZNX_NVT */
    int         width;       /* capture width  (640 for HP60C MJPG)          */
    int         height;      /* capture height (642 for HP60C MJPG)          */
    int         port;        /* HTTP MJPEG stream port                       */
    const char* weights_dir; /* holds yolov8n_qcs6490.bin + libQnnHtp.so     */
    const char* qairt_root;  /* QAIRT SDK root; NULL -> $HOME/qairt/2.42.0.251225 */
    float       conf;        /* detection confidence threshold               */
    int         detect_every;/* run detection every N frames (>=1)           */
    int         rotate;      /* rotate frame: 0/90/180/270 degrees           */
    int         motion_min_area; /* min motion contour area px; 0 disables    */
    int         annotate;    /* 1 = burn boxes/HUD into the stream (default)  */
    int         attrib;      /* 1 = per-person gender/age/height/weight (M5)   */
    double      head_cm;     /* height-estimate knob: cm per head (~23)        */
    double      bmi;         /* weight-estimate knob: assumed BMI (~22)        */
    int         age_offset;  /* age correction added to genderage output (~-5) */
    double      focal;       /* RGB focal length px (rgb fy) for depth-height (~570) */
} dh_config;

/* Blocking; returns 0 on clean shutdown (SIGINT), non-zero on fatal error. */
int dh_run(const dh_config* cfg);

/* Ask dh_run() to stop (call from a signal handler). */
void dh_request_stop(void);

#ifdef __cplusplus
}
#endif
#endif
