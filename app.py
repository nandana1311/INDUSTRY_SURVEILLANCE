"""
Integrated Safety Surveillance Web App
Flask backend — connects surveillance.py with the web dashboard
Run: python app.py --ppe-model ppe_detection.pt --fire-model fire.pt
"""

import os, sys, uuid, json, traceback, threading, base64, time
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

def frame_to_b64(frame):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode()

def get_stats():
    if system is None: return {}
    return dict(frames=system.frame_count,
                ppe_violations=system.total_violations,
                fire=system.total_fire, smoke=system.total_smoke,
                total_alerts=len(system.alerts.history))

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    'yolov8n': {'name':'YOLOv8n (Main PPE)',       'backbone':'CSPDarkNet',    'runner':'yolo',    'path':'models/best.pt',        'color':'#16a34a'},
    'yolov5':  {'name':'YOLOv5',                   'backbone':'CSP-DarkNet53', 'runner':'yolo',    'path':'models/YOLOv5.pt',      'color':'#7c3aed'},
    'yolov3':  {'name':'YOLOv3-spp',               'backbone':'DarkNet53',     'runner':'yolo',    'path':'models/YOLOv3spp-19.pt','color':'#db2777'},
    'frcnn_r': {'name':'Faster R-CNN (ResNet50)',  'backbone':'ResNet50+FPN',  'runner':'frcnn_r', 'path':'models/resNetFpn-model-19.pth', 'color':'#2563eb'},
    'frcnn_m': {'name':'Faster R-CNN (MobileNet)','backbone':'MobileNetV3',   'runner':'frcnn_m', 'path':'models/mobile-model.pth',      'color':'#0891b2'},
    'ssd':     {'name':'SSD',                      'backbone':'VGG16',         'runner':'ssd',     'path':'models/SSD_19.pth',            'color':'#d97706'},
}

import glob
for pt_file in glob.glob("models/*.pt"):
    fname = Path(pt_file).name
    if not any(Path(v['path']).name == fname for v in MODEL_REGISTRY.values()):
        idx = fname.split('.')[0]
        MODEL_REGISTRY[idx] = {
            'name': f"YOLO ({fname})",
            'backbone': 'Custom',
            'runner': 'yolo',
            'path': pt_file.replace('\\', '/'),
            'color': '#3b82f6'
        }
BENCHMARK = {
    'yolov8n': {'ap50':81.0,'precision':83.2,'recall':78.6,'fps':45},
    'yolov5':  {'ap50':79.3,'precision':81.5,'recall':76.8,'fps':52},
    'yolov3':  {'ap50':74.8,'precision':77.6,'recall':72.0,'fps':38},
    'frcnn_r': {'ap50':78.4,'precision':81.2,'recall':76.8,'fps':12},
    'frcnn_m': {'ap50':72.1,'precision':75.4,'recall':69.3,'fps':22},
    'ssd':     {'ap50':65.7,'precision':68.9,'recall':62.1,'fps':58},
}

def _run_yolo(model_path, img_path, conf=0.25):
    try:
        from ultralytics import YOLO
        import torch
        m = YOLO(str(model_path))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        half = device == 'cuda'
        res = m.predict(str(img_path), conf=conf, device=device, half=half, verbose=False)
        annotated = res[0].plot()
        dets = [{'label':m.names[int(b.cls[0])],
                 'confidence':round(float(b.conf[0]),3),
                 'bbox':b.xyxy[0].cpu().numpy().tolist()}
                for b in res[0].boxes]
        return annotated, dets
    except Exception as e:
        print(f"YOLO error: {e}"); return None, []

def _run_frcnn(model_path, img_path, backbone='resnet'):
    try:
        import torch
        from torchvision import transforms
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn)
        from PIL import Image as PILImage
        LABELS = ['__bg__','Hardhat','Mask','NO-Hardhat','NO-Mask',
                  'NO-Safety Vest','Person','Safety Cone','Safety Vest','machinery','vehicle']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model  = (fasterrcnn_mobilenet_v3_large_fpn(num_classes=11)
                  if backbone=='mobilenet' else fasterrcnn_resnet50_fpn(num_classes=11))
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        model.to(device).eval()
        img    = PILImage.open(str(img_path)).convert('RGB')
        tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
        with torch.no_grad(): preds = model(tensor)[0]
        frame = cv2.imread(str(img_path)); dets = []
        for box, score, lbl in zip(preds['boxes'],preds['scores'],preds['labels']):
            if score < 0.5: continue
            x1,y1,x2,y2 = map(int,box.tolist())
            label = LABELS[lbl] if lbl < len(LABELS) else str(int(lbl))
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,200,80),2)
            cv2.putText(frame,f"{label} {score:.2f}",(x1,y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            dets.append({'label':label,'confidence':round(float(score),3),'bbox':[x1,y1,x2,y2]})
        return frame, dets
    except Exception as e:
        print(f"FRCNN error: {e}"); return None, []

def _run_ssd(model_path, img_path):
    try:
        import torch
        from torchvision.models.detection import ssd300_vgg16
        from torchvision import transforms
        from PIL import Image as PILImage
        LABELS = ['__bg__','Hardhat','Mask','NO-Hardhat','NO-Mask',
                  'NO-Safety Vest','Person','Safety Cone','Safety Vest','machinery','vehicle']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model  = ssd300_vgg16(num_classes=11)
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        model.to(device).eval()
        img    = PILImage.open(str(img_path)).convert('RGB')
        tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
        with torch.no_grad(): preds = model(tensor)[0]
        frame = cv2.imread(str(img_path)); dets = []
        for box, score, lbl in zip(preds['boxes'],preds['scores'],preds['labels']):
            if score < 0.5: continue
            x1,y1,x2,y2 = map(int,box.tolist())
            label = LABELS[lbl] if lbl < len(LABELS) else str(int(lbl))
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,140,0),2)
            cv2.putText(frame,f"{label} {score:.2f}",(x1,y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            dets.append({'label':label,'confidence':round(float(score),3),'bbox':[x1,y1,x2,y2]})
        return frame, dets
    except Exception as e:
        print(f"SSD error: {e}"); return None, []

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index(): return render_template('index.html')

@app.route('/comparison')
def comparison(): return render_template('comparison.html')

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
            ppe_model_path=data.get('ppe_model','models/best.pt'),
            fire_model_path=data.get('fire_model','models/fire_smoke_detection.pt'),
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

@app.route('/api/compare', methods=['POST'])
def api_compare():
    if 'image' not in request.files:
        return jsonify({'error':'No image'}),400
    f  = request.files['image']
    fn = f"{uuid.uuid4().hex}{Path(secure_filename(f.filename)).suffix}"
    img_path = UPLOAD_DIR / fn
    f.save(str(img_path))
    results = []
    for mid, meta in MODEL_REGISTRY.items():
        t0    = time.time()
        mpath = Path(meta['path'])
        frame, dets = None, []
        if mpath.exists():
            r = meta['runner']
            if r == 'yolo':             frame,dets = _run_yolo(mpath,img_path)
            elif r == 'frcnn_r':        frame,dets = _run_frcnn(mpath,img_path,'resnet')
            elif r == 'frcnn_m':        frame,dets = _run_frcnn(mpath,img_path,'mobilenet')
            elif r == 'ssd':            frame,dets = _run_ssd(mpath,img_path)
        elapsed = round((time.time()-t0)*1000)
        if frame is None:
            orig = cv2.imread(str(img_path))
            if orig is not None:
                cv2.putText(orig,f"Model not found: {meta['path']}",(20,40),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                frame = orig
        bm = BENCHMARK.get(mid,{})
        results.append({
            'id':mid,'name':meta['name'],'backbone':meta['backbone'],
            'color':meta['color'],
            'ap50':bm.get('ap50',0),'precision':bm.get('precision',0),
            'recall':bm.get('recall',0),'fps':bm.get('fps',0),
            'inference_ms':elapsed,'detections':dets,'det_count':len(dets),
            'image_b64':frame_to_b64(frame) if frame is not None else None,
            'model_found':mpath.exists(),
        })
    try: img_path.unlink()
    except: pass
    return jsonify({'results':results})

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
    ap.add_argument('--port',  type=int, default=5000)
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()
    if SURVEILLANCE_AVAILABLE:
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            system = SurveillanceSystem(
                ppe_model_path=args.ppe_model,
                fire_model_path=args.fire_model,
                device=device,enable_audio=False,
                enable_logging=True,show_dashboard=True)
            print(f"✓ Surveillance system ready (Device: {device})")
        except Exception as e:
            print(f"✗ Could not auto-init: {e}")
    print(f"\n{'='*50}\n  http://{args.host}:{args.port}\n{'='*50}\n")
    app.run(host=args.host,port=args.port,debug=args.debug,threaded=True)