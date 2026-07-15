#!/usr/bin/env python3
# server.py — Orange Pi host for the detect_human webview.
# Proxies the Radxa detector's video + JSON so the browser only talks to the
# Orange Pi (which is reachable over Tailscale; the Radxa is LAN-only), and
# serves the single-page UI: live view, HTML FPS readout, 3-view switch, and
# per-attribute toggle buttons.
import os, sys, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Radxa detector base URL: env RADXA_URL, else argv[1], else this LAN default.
RADXA = os.environ.get("RADXA_URL", "http://192.168.1.10:8092")
PORT  = 8090                          # 8080 is taken by another project on this host

PAGE = r"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Detect Human — Radxa</title>
<style>
:root{--bg:#0d1117;--fg:#e6edf3;--mut:#8b949e;--acc:#2ea043;--card:#161b22;--bd:#30363d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.4 system-ui,Segoe UI,Roboto,sans-serif}
header{display:flex;align-items:center;gap:16px;padding:10px 16px;border-bottom:1px solid var(--bd)}
header h1{font-size:16px;margin:0;font-weight:600}
.badge{margin-left:auto;display:flex;gap:14px;align-items:center}
.badge b{font-variant-numeric:tabular-nums}
.fps{color:var(--acc);font-weight:700;font-size:18px}
.wrap{max-width:1000px;margin:0 auto;padding:16px}
.stage{position:relative;background:#000;border:1px solid var(--bd);border-radius:8px;
overflow:hidden;line-height:0}
.stage img{width:100%;display:block}
.stage img.night{filter:brightness(2.3) contrast(1.35) saturate(.35) sepia(.6) hue-rotate(55deg)}
.stage canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.placeholder{color:var(--mut);padding:60px;text-align:center;line-height:1.4}
.row{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}
.row .lbl{color:var(--mut);align-self:center;margin-right:4px;font-size:13px}
button{background:var(--card);color:var(--fg);border:1px solid var(--bd);border-radius:6px;
padding:7px 13px;cursor:pointer;font-size:14px}
button:hover{border-color:var(--mut)}
button.on{background:var(--acc);border-color:var(--acc);color:#03130a;font-weight:600}
button:disabled{opacity:.45;cursor:not-allowed}
</style></head><body>
<header>
  <h1>🎥 Detect Human</h1>
  <div class="badge">
    <span style="color:#3fb950">sống: <b id="nl">0</b></span>
    <span style="color:#58a6ff">thiết bị: <b id="nd">0</b></span>
    <span id="mo" style="color:var(--mut)">motion</span>
    <span class="fps">FPS <b id="fps">0</b></span>
  </div>
</header>
<div class="wrap">
  <div class="stage">
    <img id="cam" alt="camera">
    <canvas id="ov"></canvas>
    <div id="ph" class="placeholder" hidden></div>
  </div>

  <div class="row">
    <span class="lbl">Màn hình:</span>
    <button class="view on" data-view="rgb">RGB thường</button>
    <button class="view" data-view="ir">Nhìn đêm</button>
    <button class="view" data-view="lidar">LiDAR (độ sâu)</button>
  </div>

  <div class="row">
    <span class="lbl">Hiện thêm:</span>
    <button id="btnAnimals">Động vật</button>
    <button id="btnUnknown">Chưa rõ</button>
    <button id="btnDevices">Thiết bị xung quanh</button>
  </div>

  <div class="row">
    <span class="lbl">Thuộc tính người:</span>
    <button class="attr" data-attr="gender">Giới tính</button>
    <button class="attr" data-attr="age">Tuổi</button>
    <button class="attr" data-attr="height">Chiều cao</button>
    <button class="attr" data-attr="weight">Cân nặng</button>
  </div>
</div>

<script>
const cam=document.getElementById('cam'), ov=document.getElementById('ov'),
      ph=document.getElementById('ph'), ctx=ov.getContext('2d');
let view='rgb', showAnimals=false, showUnknown=false, showDevices=false;
const attrs={gender:false,age:false,height:false,weight:false};
let last={fps:0,w:640,h:480,dets:[],motion:false};
// Only person (cls 0) is always shown; animals (COCO 14..23), unknown-motion
// (cls -1, "chưa rõ") and devices/objects each hide behind their own toggle.
const isPerson=d=>d.cls===0;
const isAnimal=d=>d.cls>=14&&d.cls<=23;
const isLiving=d=>isPerson(d)||isAnimal(d);
const isUnknown=d=>d.cls===-1;
const isDevice=d=>!isLiving(d)&&!isUnknown(d);

// view switch
document.querySelectorAll('.view').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('.view').forEach(x=>x.classList.remove('on'));
  b.classList.add('on'); view=b.dataset.view; applyView();
});
function applyView(){
  // rgb + ir both show the RGB stream (ir = software low-light filter); lidar
  // shows the real colormapped depth stream from the Angstrong sensor.
  ph.hidden=true; cam.hidden=false;
  cam.classList.toggle('night', view==='ir');
  cam.src = (view==='lidar' ? '/depth?' : '/video?') + Date.now();
}
// attribute toggles
document.querySelectorAll('.attr').forEach(b=>b.onclick=()=>{
  const k=b.dataset.attr; attrs[k]=!attrs[k]; b.classList.toggle('on',attrs[k]);
});
// category toggles — each hidden until sticked on (person always shows)
document.getElementById('btnAnimals').onclick=function(){
  showAnimals=!showAnimals; this.classList.toggle('on',showAnimals);
};
document.getElementById('btnUnknown').onclick=function(){
  showUnknown=!showUnknown; this.classList.toggle('on',showUnknown);
};
document.getElementById('btnDevices').onclick=function(){
  showDevices=!showDevices; this.classList.toggle('on',showDevices);
};

// poll detection JSON for FPS + counts + (future) attribute overlays
async function poll(){
  try{
    const r=await fetch('/data',{cache:'no-store'}); last=await r.json();
    document.getElementById('fps').textContent=last.fps.toFixed(1);
    document.getElementById('nl').textContent=last.dets.filter(isLiving).length;
    document.getElementById('nd').textContent=
      last.dets.filter(d=>!isLiving(d)&&!isUnknown(d)).length;
    document.getElementById('mo').style.color=last.motion?'#e3b341':'#8b949e';
    drawOverlay();
  }catch(e){}
}
function drawOverlay(){
  // The Radxa serves RAW video; all boxes are drawn here from /data so toggles
  // control visibility. Living beings (người/động vật) always show; surrounding
  // devices only when the "Thiết bị xung quanh" button is sticked on.
  // overlay boxes over the RGB-based views (rgb/ir); depth has its own map
  if(view==='lidar'||!cam.clientWidth){ ctx.clearRect(0,0,ov.width,ov.height); return; }
  ov.width=cam.clientWidth; ov.height=cam.clientHeight;
  const sx=cam.clientWidth/last.w, sy=cam.clientHeight/last.h;
  ctx.clearRect(0,0,ov.width,ov.height);
  ctx.font='13px system-ui'; ctx.lineWidth=2; ctx.textBaseline='bottom';
  for(const d of last.dets){
    if(isAnimal(d)  && !showAnimals)  continue;        // each category hidden
    if(isUnknown(d) && !showUnknown)  continue;        // until its toggle is
    if(isDevice(d)  && !showDevices)  continue;        // sticked on (person always)
    const col = isUnknown(d) ? '#e3b341'
              : d.cls===0 ? '#3fb950'
              : isLiving(d) ? '#f0883e' : '#58a6ff';
    const x=d.x*sx, y=d.y*sy, w=d.w*sx, h=d.h*sy;
    ctx.strokeStyle=col; ctx.strokeRect(x,y,w,h);
    let t = isUnknown(d) ? 'chưa rõ'
          : d.name + (d.conf ? ' '+Math.round(d.conf*100)+'%' : '');
    if(d.cls===0){                                     // per-person attributes (M5)
      const p=[];
      if(attrs.gender&&d.gender) p.push(d.gender);
      if(attrs.age&&d.age!=null) p.push(d.age+'t');
      if(attrs.height&&d.height!=null) p.push(d.height+'cm');
      if(attrs.weight&&d.weight!=null) p.push(d.weight+'kg');
      if(p.length) t += ' · ' + p.join(' ');
    }
    const tw=ctx.measureText(t).width+8, ty=Math.max(15,y);
    ctx.fillStyle=col; ctx.fillRect(x, ty-15, tw, 15);
    ctx.fillStyle='#000'; ctx.fillText(t, x+4, ty-2);
  }
}
applyView(); setInterval(poll,400); poll();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        if self.path.startswith("/data"):
            try:
                data = urllib.request.urlopen(RADXA + "/data", timeout=3).read()
            except Exception:
                data = b'{"fps":0,"w":640,"h":480,"dets":[],"motion":false}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path.startswith("/video"):
            self._proxy_stream("/")
        elif self.path.startswith("/depth"):
            self._proxy_stream("/depth")
        else:
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def _proxy_stream(self, path):
        try:
            up = urllib.request.urlopen(RADXA + path, timeout=5)
        except Exception:
            self.send_error(502, "detector offline")
            return
        self.send_response(200)
        self.send_header("Content-Type", up.headers.get(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while True:
                chunk = up.read(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            up.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        RADXA = sys.argv[1]
    print(f"webview host on :{PORT}  (detector: {RADXA})")
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
