import cv2
import os
import threading
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# ── Backend Sound Controller (Bypass Browser Policies) ───────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIREN_SOUND_FILE = os.path.join(BASE_DIR, "static", "sound", "siren.mp3")
MACHINEGUN_SOUND_FILE = os.path.join(BASE_DIR, "static", "sound", "mechine_gun.mp3")

_siren_played_once = False
_machinegun_playing = False

import ctypes

def play_siren_sound():
    global _siren_played_once
    if _siren_played_once:
        return
    
    if os.path.exists(SIREN_SOUND_FILE):
        _siren_played_once = True
        try:
            ctypes.windll.winmm.mciSendStringW('close siren', None, 0, None)
            ctypes.windll.winmm.mciSendStringW(f'open "{SIREN_SOUND_FILE}" type mpegvideo alias siren', None, 0, None)
            ctypes.windll.winmm.mciSendStringW('play siren', None, 0, None)
        except Exception as e:
            print(f"[ERROR] Siren play failed: {e}")

def play_machinegun_sound():
    global _machinegun_playing
    if _machinegun_playing:
        return
    
    if os.path.exists(MACHINEGUN_SOUND_FILE):
        _machinegun_playing = True
        try:
            ctypes.windll.winmm.mciSendStringW('close mg', None, 0, None)
            ctypes.windll.winmm.mciSendStringW(f'open "{MACHINEGUN_SOUND_FILE}" type mpegvideo alias mg', None, 0, None)
            ctypes.windll.winmm.mciSendStringW('play mg repeat', None, 0, None)
        except Exception as e:
            print(f"[ERROR] Machinegun play failed: {e}")

def stop_machinegun_sound():
    global _machinegun_playing
    if _machinegun_playing:
        try:
            ctypes.windll.winmm.mciSendStringW('stop mg', None, 0, None)
            ctypes.windll.winmm.mciSendStringW('close mg', None, 0, None)
        except Exception:
            pass
        _machinegun_playing = False


# ── Load YOLO model (auto-downloads yolov8n.pt on first run) ─────────────────
print("[INFO] Loading YOLO model …")
model = YOLO("yolov8n.pt")
print("[INFO] Model loaded successfully.")

# ── Global state ─────────────────────────────────────────────────────────────
jet_data = {"detected": False, "x": 0, "y": 0, "conf": 0}


def gen_frames():
    cap = cv2.VideoCapture(0)
    
    # Push for extreme precision, framerate, and zero latency
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Zero out latency buffer

    if not cap.isOpened():
        print("[ERROR] Cannot open camera – check that a webcam is connected.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("[WARNING] Failed to read frame – retrying …")
            continue

        # Run YOLO detection (imgsz=320 massive speedup for CPU, processing 4x fewer pixels)
        results = model(frame, stream=True, verbose=False, imgsz=320)
        detected_this_frame = False

        highest_conf = 0.0
        for r in results:
            for box in r.boxes:
                # Class 4 = 'airplane' in COCO dataset
                if int(box.cls[0]) == 4:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf_percent = round(conf * 100, 1)

                    if conf_percent > highest_conf:
                        highest_conf = conf_percent
                        jet_data["x"]        = (x1 + x2) // 2
                        jet_data["y"]        = (y1 + y2) // 2
                        jet_data["conf"]     = conf_percent
                    
                    jet_data["detected"] = True
                    detected_this_frame  = True

                    # Draw bounding box & label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        f"JET {conf_percent}%",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                    )

        if detected_this_frame:
            # First time detection
            play_siren_sound()
            
            # Fire mode
            if jet_data["conf"] >= 50:
                play_machinegun_sound()
            else:
                stop_machinegun_sound()
        else:
            jet_data["detected"] = False
            stop_machinegun_sound()

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

    cap.release()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/get_coords')
def get_coords():
    return jsonify(jet_data)


if __name__ == "__main__":
    import os
    # Use the port Render provides, or default to 5000 locally
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
