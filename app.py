import os, sys, traceback, threading, time
from pathlib import Path
from datetime import datetime
from collections import deque
import cv2
import numpy as np
from flask import (Flask, render_template, request, jsonify,
                   send_from_directory, Response, send_file)
from flask_cors import CORS
from werkzeug.utils import secure_filename

sys.path.insert(0, str(Path(__file__).parent))
try:
    from surveillance import SurveillanceSystem, Config
    SURVEILLANCE_AVAILABLE = True
except ImportError:
    SURVEILLANCE_AVAILABLE = False
    print("WARNING: surveillance.py not found – demo mode active.")

BASE_DIR   = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads";  UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = BASE_DIR / "outputs";  OUTPUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"),
            static_folder=str(BASE_DIR / "static"))
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['SECRET_KEY'] = 'ppe-surveillance-2026'

ALLOWED_IMG = {'png','jpg','jpeg','bmp','webp'}
ALLOWED_VID = {'mp4','avi','mov','mkv'}

system = None
proc_status = dict(is_processing=False, progress=0,
                   message='', error=None, result_file=None, stats=None)

def ext(f):    return f.rsplit('.',1)[-1].lower() if '.' in f else ''
def is_img(f): return ext(f) in ALLOWED_IMG
def is_vid(f): return ext(f) in ALLOWED_VID

def get_stats():
    if system is None: return {}
    return dict(frames=system.frame_count,
                ppe_violations=system.total_violations,
                fire=system.total_fire, smoke=system.total_smoke,
                total_alerts=len(system.alerts.history),
                proximity_critical=getattr(system, 'total_proximity_crit', 0),
                proximity_high=getattr(system, 'total_proximity_high', 0))

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index(): return render_template('index.html')


@app.route('/history')
def history_page(): return render_template('history.html')
@app.route('/webcam')
def webcam_page(): return render_template('webcam.html')

@app.route('/api/init', methods=['POST'])
def api_init():
    global system
    data = request.json or {}
    if not SURVEILLANCE_AVAILABLE:
        return jsonify({'success':False,'error':'surveillance.py not found'})
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        system = SurveillanceSystem(
            ppe_model_path=str(BASE_DIR / data.get('ppe_model', 'models/best.pt')),
            fire_model_path=str(BASE_DIR / data.get('fire_model', 'models/fire_smoke_detection.pt')),
            device=device, enable_audio=False,
            enable_logging=True, show_dashboard=True)
        return jsonify({'success':True,'message':'System ready'})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)})

@app.route('/api/status')
def api_status():
    return jsonify({'system_ready':system is not None,
                    'processing':proc_status,'stats':get_stats()})

@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files:
        return jsonify({'success':False,'error':'No file'}),400
    f  = request.files['file']
    fn = secure_filename(f.filename)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved = UPLOAD_DIR / f"{ts}_{fn}"
    f.save(str(saved))
    ftype = 'image' if is_img(fn) else ('video' if is_vid(fn) else None)
    if not ftype:
        saved.unlink()
        return jsonify({'success':False,'error':'Unsupported type'}),400
    return jsonify({'success':True,'filename':saved.name,'file_type':ftype})

@app.route('/api/process', methods=['POST'])
def api_process():
    global proc_status
    if system is None:
        return jsonify({'success':False,'error':'System not initialised'}),400
    if proc_status['is_processing']:
        return jsonify({'success':False,'error':'Already processing'}),400

    data     = request.json or {}
    filename = data.get('filename')
    ftype    = data.get('file_type')
    in_path  = UPLOAD_DIR / filename

    out_name = (f"processed_{filename}" if ftype=='image'
                else f"processed_{Path(filename).stem}.mp4")
    out_path = OUTPUT_DIR / out_name

    def task():
        global proc_status
        system.reset_stats()
        proc_status.update(is_processing=True,progress=5,
                           message='Starting…',error=None,result_file=None)
        try:
            if ftype == 'image':
                frame  = cv2.imread(str(in_path))
                proc_status['progress'] = 30
                result = system.process_frame(frame)
                cv2.imwrite(str(out_path), result)
                proc_status.update(progress=100,message='Done',
                                   result_file=out_name,stats=get_stats())
            else:
                cap    = cv2.VideoCapture(str(in_path))
                total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                fps_v  = cap.get(cv2.CAP_PROP_FPS) or 25
                W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*'mp4v'),fps_v,(W,H))
                idx = 0
                while True:
                    ret,frame = cap.read()
                    if not ret: break
                    writer.write(system.process_frame(frame))
                    idx += 1
                    proc_status['progress'] = int(10+85*idx/total)
                    proc_status['message']  = f"Frame {idx}/{total}"
                cap.release(); writer.release()
                proc_status.update(progress=100,message='Done',
                                   result_file=out_name,stats=get_stats())
        except Exception as e:
            proc_status.update(error=str(e),message='Failed')
            traceback.print_exc()
        finally:
            proc_status['is_processing'] = False

    threading.Thread(target=task,daemon=True).start()
    return jsonify({'success':True,'output':out_name})

@app.route('/api/result/<filename>')
def api_result(filename):
    p = OUTPUT_DIR / filename
    if not p.exists(): return jsonify({'error':'Not found'}),404
    return send_file(str(p))

@app.route('/api/preview/<filename>')
def api_preview(filename):
    p = OUTPUT_DIR / filename
    if not p.exists(): return jsonify({'error':'Not found'}),404
    if is_vid(filename):
        cap = cv2.VideoCapture(str(p))
        ret,frame = cap.read(); cap.release()
        if ret:
            _,buf = cv2.imencode('.jpg',frame)
            return Response(buf.tobytes(),mimetype='image/jpeg')
    return send_file(str(p))

@app.route('/api/alerts')
def api_alerts():
    if system is None: return jsonify({'alerts':[],'count':0})
    return jsonify({'alerts':list(system.alerts.history)[-50:],
                    'count':len(system.alerts.history)})

@app.route('/api/webcam/feed')
def webcam_feed():
    def gen():
        if system: system.reset_stats()
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        try:
            while True:
                ret,frame = cap.read()
                if not ret: break
                out = system.process_frame(frame) if system else frame
                _,buf = cv2.imencode('.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,80])
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       +buf.tobytes()+b'\r\n')
        finally:
            cap.release()
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/clear', methods=['POST'])
def api_clear():
    for d in [UPLOAD_DIR,OUTPUT_DIR]:
        for f in d.glob('*'):
            try: f.unlink()
            except: pass
    return jsonify({'success':True})

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ppe-model',  default='models/best.pt')
    ap.add_argument('--fire-model', default='models/fire_smoke_detection.pt')
    ap.add_argument('--host',  default='127.0.0.1')
    ap.add_argument('--port',  type=int, default=5001)
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    if SURVEILLANCE_AVAILABLE:
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            system = SurveillanceSystem(
                ppe_model_path=str(BASE_DIR / args.ppe_model),
                fire_model_path=str(BASE_DIR / args.fire_model),
                device=device,enable_audio=False,
                enable_logging=True,show_dashboard=True)
            print(f"✓ Surveillance system ready (Device: {device})")
        except Exception as e:
            print(f"✗ Could not auto-init: {e}")

    print(f"\n{'='*50}\n  http://{args.host}:{args.port}\n{'='*50}\n")
    app.run(host=args.host,port=args.port,debug=args.debug,threaded=True)