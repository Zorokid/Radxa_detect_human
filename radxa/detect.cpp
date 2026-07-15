// detect.cpp — real-time full-body detector core, Hexagon-NPU edition.
//
// Pipeline:
//   Capture : v4l2-ctl streams MJPG (HP60C's only working mode, 640x642) to a
//             pipe; we frame the JPEGs on SOI boundaries and cv::imdecode them
//             (cv::VideoCapture is black on this camera).
//   Detect  : YOLOv8n (COCO 80 classes) on the NPU -> boxes + class + score.
//   Output  : an MJPEG HTTP stream (multipart/x-mixed-replace) with boxes,
//             labels and a live FPS HUD, watchable in a browser on the LAN.
//
// The model runs on the Hexagon NPU through qnn_py::QnnRuntime, a thin C++
// wrapper over the QNN SampleApp that loads a prebuilt QCS6490 context .bin and
// executes it on the HTP backend. Input/output are the graph's native
// UFIXED_POINT_8; we quantise the [0,1] letterboxed blob in and dequantise the
// [1,84,8400] head out (both adaptive in case a tensor came out float32).

#include "detect.h"

#include <opencv2/opencv.hpp>

#include "qnn/QnnRuntime.hpp"

#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <string>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <chrono>

#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

using std::vector;
using std::string;
using qnn_py::QnnRuntime;
using qnn_py::TensorInfo;

// ---- shutdown flag -------------------------------------------------------
static std::atomic<bool> g_stop{false};
extern "C" void dh_request_stop(void) { g_stop.store(true); }

// ---- COCO 80 class names (YOLOv8) ----------------------------------------
static const char* COCO[80] = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"};
// classes 14..23 are animals; index 0 is person. Everything else = object.
static bool is_animal(int c) { return c >= 14 && c <= 23; }

// ---- quant helpers (QNN convention: real = scale * (q + offset)) ----------
static size_t elems(const TensorInfo& t) {
    size_t n = 1; for (uint32_t d : t.dims) n *= (d ? d : 1); return n;
}
// Quantise a float blob to the graph's native input bytes. If the input tensor
// is u8 (byte_size == elems) we quantise; if it's float32 we pass raw bytes.
static vector<uint8_t> prepare_input(const vector<float>& f, const TensorInfo& in) {
    size_t n = f.size();
    if (in.byte_size == n) {              // UFIXED_POINT_8
        vector<uint8_t> q(n);
        double s = in.scale > 0 ? in.scale : 1.0;
        for (size_t i = 0; i < n; ++i) {
            long v = std::lround(f[i] / s) - in.offset;
            q[i] = (uint8_t)std::min(255L, std::max(0L, v));
        }
        return q;
    }
    // float32 input: reinterpret
    vector<uint8_t> b(n * sizeof(float));
    std::memcpy(b.data(), f.data(), b.size());
    return b;
}
// Dequantise a native output blob to float. u8 -> dequant, float32 -> copy.
static vector<float> read_output(const vector<uint8_t>& raw, const TensorInfo& out) {
    size_t n = elems(out);
    vector<float> o(n);
    if (raw.size() == n) {                // UFIXED_POINT_8
        double s = out.scale;
        for (size_t i = 0; i < n; ++i) o[i] = (float)(((double)raw[i] + out.offset) * s);
    } else if (raw.size() == n * sizeof(float)) {
        std::memcpy(o.data(), raw.data(), raw.size());
    } else {
        o.assign(std::min(n, raw.size() / sizeof(float)), 0.f);
        std::memcpy(o.data(), raw.data(), o.size() * sizeof(float));
    }
    return o;
}

// ================= latest annotated JPEG shared with HTTP clients ==========
struct FrameHub {
    std::mutex m;
    std::condition_variable cv;
    vector<uchar> jpeg;
    uint64_t seq = 0;
} g_hub;

static void publish(const vector<uchar>& jpg) {
    std::lock_guard<std::mutex> lk(g_hub.m);
    g_hub.jpeg = jpg;
    g_hub.seq++;
    g_hub.cv.notify_all();
}

// Second MJPEG stream: colormapped depth for the LiDAR view (served on /depth).
static FrameHub g_depth_hub;
static void publish_to(FrameHub& h, const vector<uchar>& jpg) {
    std::lock_guard<std::mutex> lk(h.m);
    h.jpeg = jpg;
    h.seq++;
    h.cv.notify_all();
}

// ---- Angstrong shm frame source (the ascamera service writes these) --------
struct ShmMeta { int rgb_w = 0, rgb_h = 0, depth_w = 0, depth_h = 0;
                 size_t rgb_size = 0, depth_size = 0; long frame_id = -1; };
static bool read_shm_meta(ShmMeta& m) {
    FILE* f = fopen("/dev/shm/ascamera_meta.txt", "r");
    if (!f) return false;
    char line[64]; long v;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "rgb_w=%d", &m.rgb_w) == 1) continue;
        if (sscanf(line, "rgb_h=%d", &m.rgb_h) == 1) continue;
        if (sscanf(line, "rgb_size=%zu", &m.rgb_size) == 1) continue;
        if (sscanf(line, "depth_w=%d", &m.depth_w) == 1) continue;
        if (sscanf(line, "depth_h=%d", &m.depth_h) == 1) continue;
        if (sscanf(line, "depth_size=%zu", &m.depth_size) == 1) continue;
        if (sscanf(line, "frame_id=%ld", &v) == 1) { m.frame_id = v; continue; }
    }
    fclose(f);
    return m.rgb_w > 0 && m.rgb_h > 0;
}
static bool read_shm_blob(const char* path, size_t expect, vector<uchar>& out) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    out.resize(expect);
    size_t got = fread(out.data(), 1, expect, f);
    fclose(f);
    return got == expect;
}

// Latest detection snapshot as JSON, served on /data for the Orange Pi webview.
static std::mutex g_data_m;
static string g_data = "{\"fps\":0,\"w\":0,\"h\":0,\"dets\":[]}";
static void publish_data(const string& j) {
    std::lock_guard<std::mutex> lk(g_data_m);
    g_data = j;
}

// ================= HTTP MJPEG server ======================================
static int make_listen_socket(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = INADDR_ANY;
    a.sin_port = htons((uint16_t)port);
    if (bind(fd, (sockaddr*)&a, sizeof(a)) < 0) { close(fd); return -1; }
    if (listen(fd, 8) < 0) { close(fd); return -1; }
    return fd;
}

static bool send_all(int fd, const char* p, size_t n) {
    while (n > 0) {
        ssize_t k = send(fd, p, n, MSG_NOSIGNAL);
        if (k <= 0) return false;
        p += k; n -= (size_t)k;
    }
    return true;
}

static void serve_client(int fd) {
    char req[1024] = {0};
    recv(fd, req, sizeof(req) - 1, 0);

    // /data -> one-shot JSON snapshot (CORS-open for the Orange Pi webview).
    if (strncmp(req, "GET /data", 9) == 0) {
        string body;
        { std::lock_guard<std::mutex> lk(g_data_m); body = g_data; }
        char h[256];
        int n = snprintf(h, sizeof(h),
            "HTTP/1.0 200 OK\r\nConnection: close\r\n"
            "Access-Control-Allow-Origin: *\r\nCache-Control: no-cache\r\n"
            "Content-Type: application/json\r\nContent-Length: %zu\r\n\r\n", body.size());
        send_all(fd, h, (size_t)n);
        send_all(fd, body.data(), body.size());
        close(fd);
        return;
    }

    // /depth -> the colormapped LiDAR stream; anything else -> the RGB stream.
    FrameHub& hub = (strncmp(req, "GET /depth", 10) == 0) ? g_depth_hub : g_hub;

    const char* hdr =
        "HTTP/1.0 200 OK\r\n"
        "Connection: close\r\n"
        "Cache-Control: no-cache\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
    if (!send_all(fd, hdr, strlen(hdr))) { close(fd); return; }

    uint64_t last = 0;
    while (!g_stop.load()) {
        vector<uchar> jpg;
        {
            std::unique_lock<std::mutex> lk(hub.m);
            hub.cv.wait_for(lk, std::chrono::milliseconds(500),
                              [&]{ return hub.seq != last || g_stop.load(); });
            if (g_stop.load()) break;
            if (hub.seq == last) continue;
            last = hub.seq;
            jpg = hub.jpeg;
        }
        char part[128];
        int n = snprintf(part, sizeof(part),
            "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n",
            jpg.size());
        if (!send_all(fd, part, (size_t)n)) break;
        if (!send_all(fd, (const char*)jpg.data(), jpg.size())) break;
        if (!send_all(fd, "\r\n", 2)) break;
    }
    close(fd);
}

// ================= YOLOv8n detection (on NPU) =============================
// input  "images"  [1,3,640,640] RGB, letterboxed, normalised /255.
// output "output0" [1,84,8400]  -> per anchor: [cx,cy,w,h (640px), 80 class probs]
static const int YOLO = 640;

static const int CLS_UNKNOWN = -1;   // moving thing YOLO didn't recognise

struct Det {
    cv::Rect2f box;   // in captured-frame pixels
    float score;
    int   cls;
    // M5 per-person attributes (only filled for cls 0 when a face is matched):
    int   gender = -1;   // 0=F, 1=M
    int   age    = -1;   // years
    int   height = -1;   // cm (estimate)
    int   weight = -1;   // kg (estimate)
};

static float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    float x1 = std::max(a.x, b.x), y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width, b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);
    float w = std::max(0.f, x2 - x1), h = std::max(0.f, y2 - y1);
    float inter = w * h;
    float uni = a.width * a.height + b.width * b.height - inter;
    return uni > 0 ? inter / uni : 0.f;
}

static void yolo_detect(QnnRuntime& net, const TensorInfo& in_info,
                        const vector<TensorInfo>& out_info, const cv::Mat& frame,
                        float conf, vector<Det>& out) {
    out.clear();
    const int W = frame.cols, H = frame.rows;

    // Letterbox (centred, gray 114) — MUST match the calibration preprocessing.
    float r = std::min((float)YOLO / W, (float)YOLO / H);
    int nw = (int)std::round(W * r), nh = (int)std::round(H * r);
    int padx = (YOLO - nw) / 2, pady = (YOLO - nh) / 2;
    cv::Mat resized; cv::resize(frame, resized, cv::Size(nw, nh));
    cv::Mat canvas(YOLO, YOLO, frame.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(padx, pady, nw, nh)));

    // float [0,1] NCHW RGB
    cv::Mat blob = cv::dnn::blobFromImage(canvas, 1.0 / 255.0, cv::Size(YOLO, YOLO),
                                          cv::Scalar(0, 0, 0), /*swapRB=*/true, false);
    vector<float> fin((float*)blob.datastart, (float*)blob.dataend);
    vector<uint8_t> qin = prepare_input(fin, in_info);
    vector<const uint8_t*> in_ptrs{ qin.data() };
    vector<size_t> in_sizes{ qin.size() };
    vector<vector<uint8_t>> outs = net.runRawPtrs(in_ptrs, in_sizes, 0);
    if (outs.size() < 2) return;

    // Two outputs (own quant scale each): boxes[1,4,8400], scores[1,80,8400].
    // Identify by channel count (dims[1]) — QNN output order isn't guaranteed.
    const float *fb = nullptr, *fs = nullptr;
    static vector<float> b0, b1;   // decoded buffers (reused)
    for (size_t i = 0; i < outs.size(); ++i) {
        uint32_t ch = out_info[i].dims.size() > 1 ? out_info[i].dims[1] : 0;
        vector<float>& dst = (i == 0) ? b0 : b1;
        dst = read_output(outs[i], out_info[i]);
        if (ch == 4)  fb = dst.data();
        else if (ch == 80) fs = dst.data();
    }
    if (!fb || !fs) return;

    const int A = 8400, NC = 80;
    vector<Det> cand;
    for (int a = 0; a < A; ++a) {
        int best = -1; float bs = conf;
        for (int c = 0; c < NC; ++c) {
            float v = fs[c * A + a];
            if (v > bs) { bs = v; best = c; }
        }
        if (best < 0) continue;
        float cx = fb[0 * A + a], cy = fb[1 * A + a];
        float w = fb[2 * A + a], h = fb[3 * A + a];
        // 640-letterbox coords -> frame pixels
        float x1 = ((cx - w * 0.5f) - padx) / r, y1 = ((cy - h * 0.5f) - pady) / r;
        float bw = w / r, bh = h / r;
        cand.push_back({ cv::Rect2f(x1, y1, bw, bh), bs, best });
    }

    // Class-aware NMS (greedy, score-desc, IoU 0.45).
    std::sort(cand.begin(), cand.end(), [](const Det& a, const Det& b){ return a.score > b.score; });
    vector<char> removed(cand.size(), 0);
    for (size_t i = 0; i < cand.size(); ++i) {
        if (removed[i]) continue;
        out.push_back(cand[i]);
        for (size_t j = i + 1; j < cand.size(); ++j)
            if (!removed[j] && cand[j].cls == cand[i].cls && iou(cand[i].box, cand[j].box) > 0.45f)
                removed[j] = 1;
    }
}

// ================= face attributes (SCRFD + genderage, on NPU) ============
// M5: per-person gender/age/height/weight. SCRFD det_10g finds faces (+ face
// box); genderage classifies gender + real age per face; height is estimated
// from the person-box / face-box pixel ratio (distance-invariant, ~7.5 heads
// tall), weight from height via BMI. Ported from ../Radxa_camera_age (NPU).
static const int SCRFD_SIZE = 640;
static const float SCRFD_IOU = 0.4f;
static const int   GA_SIZE   = 96;

struct Face { cv::Rect2f box; float score; };

static const float* anchor_centers(int stride) {
    static std::vector<float> cache[3];
    int idx = (stride == 8) ? 0 : (stride == 16) ? 1 : 2;
    if (!cache[idx].empty()) return cache[idx].data();
    int hw = SCRFD_SIZE / stride;
    std::vector<float>& c = cache[idx];
    c.resize((size_t)hw * hw * 2 * 2);
    size_t p = 0;
    for (int y = 0; y < hw; ++y)
        for (int x = 0; x < hw; ++x)
            for (int a = 0; a < 2; ++a) { c[p++] = (float)(x * stride); c[p++] = (float)(y * stride); }
    return c.data();
}

// Map each output blob to (stride, kind) via its dims (order isn't guaranteed).
struct OutMap { int score = -1, bbox = -1, kps = -1; };
static void build_outmap(const vector<TensorInfo>& info, OutMap m[3]) {
    for (int i = 0; i < (int)info.size(); ++i) {
        if (info[i].dims.empty()) continue;
        uint32_t rows = info[i].dims[0];
        uint32_t cols = info[i].dims.size() > 1 ? info[i].dims[1] : 1;
        int s = (rows == 12800) ? 0 : (rows == 3200) ? 1 : (rows == 800) ? 2 : -1;
        if (s < 0) continue;
        if (cols == 1)       m[s].score = i;
        else if (cols == 4)  m[s].bbox  = i;
        else if (cols == 10) m[s].kps   = i;
    }
}

static void scrfd_detect(QnnRuntime& det, const vector<TensorInfo>& out_info,
                         const OutMap outmap[3], const cv::Mat& frame,
                         float conf, vector<Face>& out) {
    out.clear();
    const int W = frame.cols, H = frame.rows;
    float im_ratio = (float)H / (float)W;
    int new_w, new_h;
    if (im_ratio > 1.0f) { new_h = SCRFD_SIZE; new_w = (int)(new_h / im_ratio); }
    else                 { new_w = SCRFD_SIZE; new_h = (int)(new_w * im_ratio); }
    float det_scale = (float)new_h / (float)H;

    cv::Mat resized; cv::resize(frame, resized, cv::Size(new_w, new_h));
    cv::Mat canvas = cv::Mat::zeros(SCRFD_SIZE, SCRFD_SIZE, frame.type());
    resized.copyTo(canvas(cv::Rect(0, 0, new_w, new_h)));

    // det_10g native input = raw letterboxed pixels (0-255, NCHW, RGB); the
    // (x-127.5)/128 normalisation is baked into the quantised graph.
    cv::Mat blob = cv::dnn::blobFromImage(canvas, 1.0, cv::Size(SCRFD_SIZE, SCRFD_SIZE),
                                          cv::Scalar(0, 0, 0), /*swapRB=*/true, false);
    vector<uint8_t> qin(blob.total());
    { const float* bp = blob.ptr<float>();
      for (size_t i = 0; i < qin.size(); ++i)
          qin[i] = (uint8_t)std::min(255.f, std::max(0.f, std::round(bp[i]))); }
    vector<const uint8_t*> in_ptrs{ qin.data() };
    vector<size_t> in_sizes{ qin.size() };
    vector<vector<uint8_t>> outs = det.runRawPtrs(in_ptrs, in_sizes, 0);
    if (outs.size() < out_info.size()) return;

    vector<vector<float>> fout(outs.size());
    for (size_t i = 0; i < outs.size(); ++i) fout[i] = read_output(outs[i], out_info[i]);

    const int strides[3] = {8, 16, 32};
    vector<Face> cand;
    for (int si = 0; si < 3; ++si) {
        const OutMap& om = outmap[si];
        if (om.score < 0 || om.bbox < 0) continue;
        int stride = strides[si];
        const float* sc = fout[om.score].data();
        const float* bb = fout[om.bbox].data();
        const float* ac = anchor_centers(stride);
        int hw = SCRFD_SIZE / stride, n = hw * hw * 2;
        for (int i = 0; i < n; ++i) {
            if (sc[i] < conf) continue;
            float cx = ac[i * 2], cy = ac[i * 2 + 1];
            float l = bb[i*4+0]*stride, t = bb[i*4+1]*stride, r = bb[i*4+2]*stride, d = bb[i*4+3]*stride;
            float x1 = (cx - l) / det_scale, y1 = (cy - t) / det_scale;
            float x2 = (cx + r) / det_scale, y2 = (cy + d) / det_scale;
            cand.push_back({ cv::Rect2f(x1, y1, x2 - x1, y2 - y1), sc[i] });
        }
    }
    std::sort(cand.begin(), cand.end(), [](const Face& a, const Face& b){ return a.score > b.score; });
    vector<char> removed(cand.size(), 0);
    for (size_t i = 0; i < cand.size(); ++i) {
        if (removed[i]) continue;
        out.push_back(cand[i]);
        for (size_t j = i + 1; j < cand.size(); ++j)
            if (!removed[j] && iou(cand[i].box, cand[j].box) > SCRFD_IOU) removed[j] = 1;
    }
}

// genderage: input [1,3,96,96] raw pixels; output fc1[1,3] -> gender=argmax(0:2),
// age=round(fc1[2]*100). Returns false if inference gave nothing.
static bool genderage_infer(QnnRuntime& ga, const TensorInfo& out_info,
                            const cv::Mat& frame, const cv::Rect2f& facebox,
                            int& gender, int& age) {
    float w = facebox.width, h = facebox.height;
    float cx = facebox.x + w * 0.5f, cy = facebox.y + h * 0.5f;
    float scale = (float)GA_SIZE / (std::max(w, h) * 1.5f);
    cv::Matx23f M(scale, 0.f, GA_SIZE * 0.5f - cx * scale,
                  0.f, scale, GA_SIZE * 0.5f - cy * scale);
    cv::Mat aligned;
    cv::warpAffine(frame, aligned, M, cv::Size(GA_SIZE, GA_SIZE),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    cv::Mat blob = cv::dnn::blobFromImage(aligned, 1.0, cv::Size(GA_SIZE, GA_SIZE),
                                          cv::Scalar(0, 0, 0), /*swapRB=*/true, false);
    vector<uint8_t> qin(blob.total());
    { const float* bp = blob.ptr<float>();
      for (size_t i = 0; i < qin.size(); ++i)
          qin[i] = (uint8_t)std::min(255.f, std::max(0.f, std::round(bp[i]))); }
    vector<const uint8_t*> in_ptrs{ qin.data() };
    vector<size_t> in_sizes{ qin.size() };
    vector<vector<uint8_t>> outs = ga.runRawPtrs(in_ptrs, in_sizes, 0);
    if (outs.empty()) return false;
    vector<float> p = read_output(outs[0], out_info);
    if (p.size() < 3) return false;
    gender = (p[0] >= p[1]) ? 0 : 1;
    age = (int)std::lround(p[2] * 100.0f);
    return true;
}

// Robust distance (mm) for a person: median of valid depth over the central
// torso window of the box (ignoring 0 = no-return). 0 if nothing valid.
static int sample_depth_mm(const cv::Mat& depth, const cv::Rect2f& box) {
    if (depth.empty()) return 0;
    int x0 = std::max(0, (int)(box.x + box.width * 0.30f));
    int x1 = std::min(depth.cols, (int)(box.x + box.width * 0.70f));
    int y0 = std::max(0, (int)(box.y + box.height * 0.15f));
    int y1 = std::min(depth.rows, (int)(box.y + box.height * 0.60f));
    if (x1 <= x0 || y1 <= y0) return 0;
    std::vector<uint16_t> v;
    for (int y = y0; y < y1; ++y) {
        const uint16_t* row = depth.ptr<uint16_t>(y);
        for (int x = x0; x < x1; ++x) if (row[x] > 0) v.push_back(row[x]);
    }
    if (v.empty()) return 0;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

// Fill per-person gender/age/height/weight. HEIGHT is real: pinhole projection
// with the true depth distance (object_height = pixel_height * Z / focal) — the
// face-ratio (~7.5 heads) is only a fallback when depth has no return. Gender/age
// come from the genderage NPU on the matched face. Knobs: focal (rgb fy px),
// head_cm, bmi, age_offset (see dh_config).
static void attach_attributes(QnnRuntime& ga, const TensorInfo& ga_out,
                              const vector<Face>& faces, vector<Det>& dets,
                              const cv::Mat& frame, const cv::Mat& depth,
                              double head_cm, double bmi, double focal,
                              int age_offset) {
    for (Det& d : dets) {
        if (d.cls != 0) continue;                 // person only
        // --- height/weight: prefer real depth (pinhole) ---
        int zmm = sample_depth_mm(depth, d.box);
        double cm = (zmm > 0 && focal > 1) ? d.box.height * (double)zmm / focal / 10.0 : 0.0;
        // --- gender/age: largest face whose centre is inside the person box ---
        const Face* best = nullptr;
        for (const Face& f : faces) {
            cv::Point2f c(f.box.x + f.box.width * 0.5f, f.box.y + f.box.height * 0.5f);
            if (!d.box.contains(c)) continue;
            if (!best || f.box.area() > best->box.area()) best = &f;
        }
        if (best) {
            int g = -1, a = -1;
            if (genderage_infer(ga, ga_out, frame, best->box, g, a)) {
                d.gender = g; d.age = std::max(1, a + age_offset);
            }
            if (cm <= 0 && best->box.height > 1.f)      // depth failed -> ratio
                cm = (d.box.height / best->box.height) * head_cm;
        }
        if (cm > 0) {
            d.height = (int)std::lround(std::min(210.0, std::max(120.0, cm)));
            double m = d.height / 100.0;
            d.weight = (int)std::lround(std::min(150.0, std::max(30.0, bmi * m * m)));
        }
    }
}

// ================= motion detection (CPU, MOG2) ===========================
// Fixed camera -> background subtraction. Returns bounding boxes of moving
// blobs above min_area. Camera is fixed so this is cheap and robust.
static bool motion_regions(const cv::Mat& frame, int min_area, vector<cv::Rect>& out) {
    static cv::Ptr<cv::BackgroundSubtractorMOG2> mog =
        cv::createBackgroundSubtractorMOG2(500, 16, false);
    out.clear();
    if (min_area <= 0) return false;
    cv::Mat fg;
    mog->apply(frame, fg);
    cv::threshold(fg, fg, 200, 255, cv::THRESH_BINARY);   // drop MOG2 shadow greys
    cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(fg, fg, cv::MORPH_OPEN, k);
    cv::dilate(fg, fg, k, cv::Point(-1, -1), 2);
    vector<vector<cv::Point>> cont;
    cv::findContours(fg, cont, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (auto& c : cont)
        if (cv::contourArea(c) >= min_area) out.push_back(cv::boundingRect(c));
    return !out.empty();
}

// Append "unknown" dets for motion blobs that don't overlap any YOLO box.
static void add_unknown_motion(const vector<cv::Rect>& motion, vector<Det>& dets) {
    for (const auto& m : motion) {
        cv::Rect2f mb(m.x, m.y, m.width, m.height);
        float marea = mb.width * mb.height;
        bool covered = false;
        for (const auto& d : dets) {
            if (d.cls == CLS_UNKNOWN) continue;
            // suppress if the motion blob is mostly INSIDE a known detection
            float x1 = std::max(mb.x, d.box.x), y1 = std::max(mb.y, d.box.y);
            float x2 = std::min(mb.x + mb.width, d.box.x + d.box.width);
            float y2 = std::min(mb.y + mb.height, d.box.y + d.box.height);
            float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
            if (marea > 0 && inter / marea > 0.5f) { covered = true; break; }
        }
        if (!covered) dets.push_back({ mb, 0.f, CLS_UNKNOWN });
    }
}

// ================= overlay ================================================
static void draw(cv::Mat& frame, const vector<Det>& dets, double fps, bool motion) {
    int people = 0;
    for (const auto& d : dets) {
        // person=green, animal=orange, object=blue, unknown-motion=yellow
        cv::Scalar col = (d.cls == CLS_UNKNOWN) ? cv::Scalar(0, 255, 255)
                       : (d.cls == 0) ? cv::Scalar(0, 220, 0)
                       : is_animal(d.cls) ? cv::Scalar(0, 165, 255)
                       : cv::Scalar(255, 160, 0);
        if (d.cls == 0) people++;
        cv::Rect rc((int)d.box.x, (int)d.box.y, (int)d.box.width, (int)d.box.height);
        cv::rectangle(frame, rc, col, 2);
        char lbl[64];
        if (d.cls == CLS_UNKNOWN) snprintf(lbl, sizeof(lbl), "unknown");
        else snprintf(lbl, sizeof(lbl), "%s %.0f%%", COCO[d.cls], d.score * 100.f);
        int base = 0;
        cv::Size ts = cv::getTextSize(lbl, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
        cv::Point org(rc.x, std::max(ts.height + 4, rc.y - 4));
        cv::rectangle(frame, org + cv::Point(0, base),
                      org + cv::Point(ts.width, -ts.height - 2), col, cv::FILLED);
        cv::putText(frame, lbl, org, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0), 1);
    }
    char hud[112];
    snprintf(hud, sizeof(hud), "%.1f fps  det:%zu  person:%d  %s  NPU",
             fps, dets.size(), people, motion ? "MOTION" : "-");
    cv::putText(frame, hud, cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 255), 2);
}

// ================= helpers ================================================
static string join(const string& a, const string& b) {
    if (!a.empty() && a.back() == '/') return a + b;
    return a + "/" + b;
}

// ================= entry ==================================================
extern "C" int dh_run(const dh_config* cfg) {
    string qairt = cfg->qairt_root ? cfg->qairt_root : string();
    if (qairt.empty()) {
        const char* home = getenv("HOME");
        qairt = string(home ? home : "/home/radxa") + "/qairt/2.42.0.251225";
    }
    string syslib = join(qairt, "lib/aarch64-oe-linux-gcc11.2/libQnnSystem.so");
    string wdir   = cfg->weights_dir;
    string backend = join(wdir, "libQnnHtp.so");
    string binpath = join(wdir, "yolov8n_qcs6490.bin");

    QnnRuntime net(backend, syslib, binpath);
    try { net.init(); }
    catch (const std::exception& e) { fprintf(stderr, "QNN init failed: %s\n", e.what()); return 2; }

    vector<TensorInfo> in_info = net.inputInfo(0);
    vector<TensorInfo> out_info = net.outputInfo(0);
    if (in_info.empty() || out_info.empty()) { fprintf(stderr, "no model I/O info\n"); return 2; }
    printf("NPU model loaded (YOLOv8n COCO) from %s\n", wdir.c_str());
    printf("  input %s bytes=%zu  output %s elems=%zu\n",
           in_info[0].name.c_str(), in_info[0].byte_size,
           out_info[0].name.c_str(), elems(out_info[0]));

    // M5: optional per-person attribute models (SCRFD det_10g + genderage).
    // Loaded best-effort — if the bins are missing or the HTP can't fit all
    // three graphs, attributes are skipped and detection keeps running.
    std::unique_ptr<QnnRuntime> det10g, genderage;
    vector<TensorInfo> ga_out_info;
    OutMap scrfd_map[3];
    bool attrib = cfg->attrib != 0;
    if (attrib) {
        try {
            det10g.reset(new QnnRuntime(backend, syslib, join(wdir, "det_10g_qcs6490.bin")));
            det10g->init();
            genderage.reset(new QnnRuntime(backend, syslib, join(wdir, "genderage_qcs6490.bin")));
            genderage->init();
            build_outmap(det10g->outputInfo(0), scrfd_map);
            ga_out_info = genderage->outputInfo(0);
            printf("NPU attribute models loaded (SCRFD det_10g + genderage)\n");
        } catch (const std::exception& e) {
            fprintf(stderr, "attribute models unavailable (%s); running without M5\n", e.what());
            det10g.reset(); genderage.reset(); attrib = false;
        }
    }

    // Frame source: the Angstrong ascamera service owns the camera (the depth
    // SDK needs exclusive USB access) and writes RGB + depth to /dev/shm. We
    // poll for a new frame_id below. This replaces the old direct v4l2 capture.
    printf("frame source: /dev/shm/ascamera_* (ascamera service)\n");

    int lfd = make_listen_socket(cfg->port);
    if (lfd < 0) { fprintf(stderr, "cannot bind port %d\n", cfg->port); return 4; }
    std::thread acceptor([lfd]{
        while (!g_stop.load()) {
            int cfd = accept(lfd, nullptr, nullptr);
            if (cfd < 0) break;
            std::thread(serve_client, cfd).detach();
        }
    });
    printf("MJPEG stream: http://<board-ip>:%d/   (Ctrl+C to stop)\n", cfg->port);

    vector<Det> dets;
    vector<uchar> rgbbuf, depthbuf;
    long frameNo = 0, last_fid = -1;
    int stale = 0;
    auto t0 = std::chrono::steady_clock::now();
    double fps = 0;

    while (!g_stop.load()) {
        ShmMeta m;
        if (!read_shm_meta(m) || m.frame_id == last_fid) {
            usleep(10000);                        // no new frame yet
            if (++stale % 500 == 0)
                fprintf(stderr, "[cap] waiting for ascamera frames (stale %d)\n", stale);
            continue;
        }
        stale = 0; last_fid = m.frame_id;

        if (m.rgb_size != (size_t)m.rgb_w * m.rgb_h * 3) continue;
        if (!read_shm_blob("/dev/shm/ascamera_rgb.raw", m.rgb_size, rgbbuf)) continue;
        cv::Mat frame(m.rgb_h, m.rgb_w, CV_8UC3, rgbbuf.data());
        frame = frame.clone();                    // detach from the shm buffer

        if (cfg->rotate == 90)  cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
        else if (cfg->rotate == 180) cv::rotate(frame, frame, cv::ROTATE_180);
        else if (cfg->rotate == 270) cv::rotate(frame, frame, cv::ROTATE_90_COUNTERCLOCKWISE);

        // Depth (uint16 mm), read once — used for real height + the depth views.
        cv::Mat d16;
        if (m.depth_size == (size_t)m.depth_w * m.depth_h * 2 &&
            read_shm_blob("/dev/shm/ascamera_depth.raw", m.depth_size, depthbuf)) {
            d16 = cv::Mat(m.depth_h, m.depth_w, CV_16U, depthbuf.data()).clone();
            if (cfg->rotate == 90)  cv::rotate(d16, d16, cv::ROTATE_90_CLOCKWISE);
            else if (cfg->rotate == 180) cv::rotate(d16, d16, cv::ROTATE_180);
            else if (cfg->rotate == 270) cv::rotate(d16, d16, cv::ROTATE_90_COUNTERCLOCKWISE);
        }

        if (frameNo % cfg->detect_every == 0) {
            yolo_detect(net, in_info[0], out_info, frame, cfg->conf, dets);
            // M5: faces -> gender/age/height/weight on any person boxes.
            // Skip the SCRFD pass entirely when no person is in frame.
            bool any_person = false;
            for (const Det& d : dets) if (d.cls == 0) { any_person = true; break; }
            if (attrib && genderage && any_person) {
                vector<Face> faces;
                scrfd_detect(*det10g, det10g->outputInfo(0), scrfd_map, frame, 0.5f, faces);
                attach_attributes(*genderage, ga_out_info[0], faces, dets, frame,
                                  d16, cfg->head_cm, cfg->bmi, cfg->focal, cfg->age_offset);
            }
        }

        // Motion runs every frame (cheap, fixed camera). Moving blobs not
        // covered by a YOLO box become "unknown" detections.
        vector<cv::Rect> motion;
        bool motion_active = motion_regions(frame, cfg->motion_min_area, motion);
        vector<Det> shown = dets;
        add_unknown_motion(motion, shown);

        auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(now - t0).count();
        t0 = now;
        if (dt > 0) fps = 0.9 * fps + 0.1 * (1.0 / dt);

        // Publish detection snapshot as JSON for the Orange Pi webview.
        {
            char head[96];
            snprintf(head, sizeof(head),
                     "{\"fps\":%.1f,\"w\":%d,\"h\":%d,\"motion\":%s,\"dets\":[",
                     fps, frame.cols, frame.rows, motion_active ? "true" : "false");
            string j = head;
            for (size_t i = 0; i < shown.size(); ++i) {
                const Det& d = shown[i];
                const char* nm = (d.cls == CLS_UNKNOWN) ? "unknown" : COCO[d.cls];
                char e[160];
                snprintf(e, sizeof(e),
                    "%s{\"x\":%d,\"y\":%d,\"w\":%d,\"h\":%d,\"cls\":%d,\"name\":\"%s\",\"conf\":%.2f",
                    i ? "," : "", (int)d.box.x, (int)d.box.y, (int)d.box.width,
                    (int)d.box.height, d.cls, nm, d.score);
                j += e;
                if (d.cls == 0 && d.gender >= 0) {   // M5 attributes (person)
                    char at[96];
                    snprintf(at, sizeof(at),
                        ",\"gender\":\"%s\",\"age\":%d,\"height\":%d,\"weight\":%d",
                        d.gender == 1 ? "Nam" : "Nữ", d.age, d.height, d.weight);
                    j += at;
                }
                j += "}";
            }
            j += "]}";
            publish_data(j);
        }

        if (cfg->annotate) draw(frame, shown, fps, motion_active);
        vector<uchar> jpg;
        cv::imencode(".jpg", frame, jpg, {cv::IMWRITE_JPEG_QUALITY, 80});
        publish(jpg);

        // LiDAR view (d16 already rotated): colormap the uint16 mm depth
        // (0..4m -> JET, no-return = black).
        if (!d16.empty()) {
            cv::Mat d8, cmap;
            d16.convertTo(d8, CV_8U, 255.0 / 4000.0);
            cv::applyColorMap(d8, cmap, cv::COLORMAP_JET);
            cmap.setTo(cv::Scalar(0, 0, 0), d16 == 0);
            vector<uchar> djpg;
            cv::imencode(".jpg", cmap, djpg, {cv::IMWRITE_JPEG_QUALITY, 80});
            publish_to(g_depth_hub, djpg);
        }

        frameNo++;
        if (frameNo % 30 == 1) {
            fprintf(stderr, "[cap] frame %ld  %dx%d  %.1ffps  det=%zu\n",
                    frameNo, frame.cols, frame.rows, fps, dets.size());
            fflush(stderr);
        }
    }

    g_stop.store(true);
    shutdown(lfd, SHUT_RDWR);
    close(lfd);
    g_hub.cv.notify_all();
    g_depth_hub.cv.notify_all();
    if (acceptor.joinable()) acceptor.join();
    try { net.close(); } catch (...) {}

    if (frameNo == 0) {
        fprintf(stderr,
            "\nERROR: no frames from /dev/shm/ascamera_*. Is the ascamera\n"
            "  service running?  systemctl status ascamera ; ls -l /dev/shm/ascamera_*\n");
        return 5;
    }
    printf("\nstopped after %ld frames\n", frameNo);
    return 0;
}
