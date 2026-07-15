#!/usr/bin/env python3
# grab.py — pull ONE annotated JPEG from the local MJPEG stream and save it.
# Runs on the board over localhost (no WiFi), so it works even while the camera
# saturates the USB/WiFi uplink. Usage: grab.py [port] [out.jpg]
import socket, sys, time
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8092
out  = sys.argv[2] if len(sys.argv) > 2 else "/tmp/shot.jpg"
# The server only starts listening after QNN init (~15s), so retry on refusal.
s = None
t0 = time.time()
while time.time() - t0 < 25:
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=5); break
    except (ConnectionRefusedError, OSError):
        time.sleep(1)
if s is None:
    print("could not connect to stream"); sys.exit(1)
s.sendall(b"GET / HTTP/1.0\r\n\r\n")
buf = b""; t = time.time()
while time.time() - t < 10:
    d = s.recv(65536)
    if not d: break
    buf += d
    a = buf.find(b"\xff\xd8"); b = buf.find(b"\xff\xd9", a + 2)
    if a >= 0 and b >= 0:
        open(out, "wb").write(buf[a:b + 2]); print("saved", b + 2 - a, "bytes ->", out); break
s.close()
