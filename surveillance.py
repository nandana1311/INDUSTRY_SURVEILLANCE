import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import threading
from collections import deque
import time
import sys

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: pip install ultralytics")
    sys.exit(1)

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    # ── PPE model classes  ───────────────
    PERSON_CLASS    = 'Person'
    PPE_CLASSES     = ['Hardhat', 'Mask', 'Safety Vest']
    REQUIRED_PPE    = ['Hardhat', 'Mask', 'Safety Vest']
    VIOLATION_CLASSES = {
        'NO-Hardhat':     'Hardhat',
        'NO-Mask':        'Mask',
        'NO-Safety Vest': 'Safety Vest',
    }
    IGNORE_CLASSES  = ['Safety Cone', 'machinery', 'vehicle']

    # ── Hazard objects for proximity reasoning ────────────────
    # These are the classes whose proximity to a non-compliant
    # worker escalates the alert to CRITICAL.
    PROXIMITY_HAZARD_CLASSES = ['machinery', 'vehicle']

    # ── Proximity thresholds (pixels) ────────────────────────
    # Distance from person centre to hazard object centre.
    # Tune these based on your typical camera distance.
    PROXIMITY_CRITICAL_PX = 250   # within 250px → CRITICAL
    PROXIMITY_HIGH_PX     = 450   # within 450px → HIGH

    # ── Fire model ───────────────────────────────────────────
    HAZARD_CLASSES  = ['fire', 'smoke']

    # ── Severity levels ──────────────────────────────────────
    SEV_CRITICAL = 3
    SEV_HIGH     = 2
    SEV_MEDIUM   = 1
    SEV_LOW      = 0

    # ── model.overrides ──────────────────────────────────────
    PPE_CONF        = 0.25
    PPE_IOU         = 0.45
    PPE_MAX_DET     = 300

    FIRE_CONF       = 0.25
    FIRE_IOU        = 0.45
    FIRE_MAX_DET    = 100

    # ── Per-person association thresholds ────────────────────
    IOU_ASSIGN      = 0.10
    DIST_ASSIGN     = 250   # pixels

    # ── Alert cooldown (seconds) ─────────────────────────────
    ALERT_COOLDOWN  = 5.0
    STATS_WINDOW    = 60

    # ── Drawing colours (BGR) ───────────────────────────────
    C_SAFE      = (0, 200, 80)
    C_WARNING   = (0, 165, 255)
    C_DANGER    = (0, 0, 220)
    C_VIOLATION = (180, 0, 220)
    C_FIRE      = (0, 0, 255)
    C_SMOKE     = (160, 160, 160)
    C_WHITE     = (255, 255, 255)
    C_BLACK     = (0, 0, 0)
    C_PROXIMITY = (0, 100, 255)   # orange — proximity critical alert


# ============================================================
# GEOMETRY HELPERS
# ============================================================
def iou(b1, b2):
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def centre_dist(b1, b2):
    c1 = ((b1[0]+b1[2])/2, (b1[1]+b1[3])/2)
    c2 = ((b2[0]+b2[2])/2, (b2[1]+b2[3])/2)
    return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)


def box_centre(b):
    return (int((b[0]+b[2])/2), int((b[1]+b[3])/2))


def box_contains(outer, inner):
    cx = (inner[0]+inner[2])/2
    cy = (inner[1]+inner[3])/2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


# ============================================================
# PROXIMITY HAZARD ANALYSIS
# ============================================================
def analyse_proximity(compliance_list, hazard_objects):
    """
    For each non-compliant person, compute the distance to every
    detected machinery/vehicle object.

    Returns a list of proximity events:
      {
        'person_id'   : int,
        'person_bbox' : [x1,y1,x2,y2],
        'hazard_class': str,
        'hazard_bbox' : [x1,y1,x2,y2],
        'distance_px' : float,
        'severity'    : int   (CRITICAL or HIGH)
      }

    Logic:
      - Only non-compliant workers (status != 'compliant') are checked.
        A fully-compliant worker near machinery is NOT flagged because
        they are properly protected.
      - Distance is centre-to-centre between person bbox and hazard bbox.
      - severity = CRITICAL if distance < PROXIMITY_CRITICAL_PX
                 = HIGH     if distance < PROXIMITY_HIGH_PX
    """
    events = []
    for person in compliance_list:
        if person['status'] == 'compliant':
            continue  # compliant worker near machinery is fine

        # Find the single closest hazard object within threshold.
        # One person → one proximity event max. Prevents label pile-up
        # when multiple machinery detections exist in the same frame.
        closest = None
        closest_dist = float('inf')

        for haz in hazard_objects:
            dist = centre_dist(person['bbox'], haz['bbox'])
            if dist < Config.PROXIMITY_HIGH_PX and dist < closest_dist:
                closest_dist = dist
                closest = haz

        if closest is None:
            continue  # no hazard within threshold for this person

        severity = (Config.SEV_CRITICAL
                    if closest_dist < Config.PROXIMITY_CRITICAL_PX
                    else Config.SEV_HIGH)

        events.append({
            'person_id':    person['id'],
            'person_bbox':  person['bbox'],
            'hazard_class': closest['class'],
            'hazard_bbox':  closest['bbox'],
            'distance_px':  round(closest_dist, 1),
            'severity':     severity,
            'missing':      person['missing'],
        })

    return events


def draw_proximity_alert(frame, event):
    """
    Draws:
      - A danger-coloured box around the hazard object
      - A line connecting the non-compliant person to the hazard
      - A compact label anchored just above the person bounding box
    """
    h_frame, w_frame = frame.shape[:2]
    px, py   = box_centre(event['person_bbox'])
    hx, hy   = box_centre(event['hazard_bbox'])
    dist_px  = event['distance_px']
    is_crit  = event['severity'] == Config.SEV_CRITICAL
    color    = Config.C_FIRE if is_crit else Config.C_PROXIMITY

    # Draw hazard bounding box
    x1, y1, x2, y2 = [int(v) for v in event['hazard_bbox']]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    # Draw line from person centre to hazard centre
    cv2.line(frame, (px, py), (hx, hy), color, 2, cv2.LINE_AA)

    # Label anchored above the PERSON box — keeps each person's label
    # attached to that person, preventing stacking in the frame centre.
    tag   = "CRIT" if is_crit else "HIGH"
    miss  = ", ".join(event['missing']) if event['missing'] else "PPE"
    label = f"{tag} | {event['hazard_class']} {dist_px:.0f}px | -{miss}"

    font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1
    (tw, th), bl = cv2.getTextSize(label, font, fs, ft)
    pad = 4

    lx = max(0, int(event['person_bbox'][0]))
    ly = max(th + pad * 2 + 2, int(event['person_bbox'][1]) - 4)
    lx = min(lx, w_frame - tw - pad * 2 - 2)

    cv2.rectangle(frame,
                  (lx, ly - th - pad),
                  (lx + tw + pad * 2, ly + bl),
                  color, -1)
    cv2.putText(frame, label, (lx + pad, ly),
                font, fs, Config.C_WHITE, ft, cv2.LINE_AA)


# ============================================================
# PER-PERSON PPE ASSOCIATION
# ============================================================
def associate_ppe_to_persons(persons, ppe_items, violation_items):
    results = []
    for pidx, pbbox in enumerate(persons):
        present    = set()
        violations = set()

        for item in ppe_items:
            if (box_contains(pbbox, item['bbox'])
                    or iou(pbbox, item['bbox']) > Config.IOU_ASSIGN
                    or centre_dist(pbbox, item['bbox']) < Config.DIST_ASSIGN):
                present.add(item['class'])

        for item in violation_items:
            if (box_contains(pbbox, item['bbox'])
                    or iou(pbbox, item['bbox']) > Config.IOU_ASSIGN
                    or centre_dist(pbbox, item['bbox']) < Config.DIST_ASSIGN):
                missing_ppe = Config.VIOLATION_CLASSES.get(item['class'])
                if missing_ppe:
                    violations.add(missing_ppe)

        missing = []
        for req in Config.REQUIRED_PPE:
            if req not in present or req in violations:
                missing.append(req)

        if not missing:
            status = 'compliant'
        elif len(missing) < len(Config.REQUIRED_PPE):
            status = 'partial'
        else:
            status = 'none'

        results.append({
            'id':         pidx,
            'bbox':       pbbox,
            'present':    present,
            'violations': violations,
            'missing':    missing,
            'status':     status,
        })
    return results


# ============================================================
# DRAWING
# ============================================================
def draw_box(frame, bbox, label, color, conf=None, thickness=2):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [
        max(0, min(int(v), (w if i % 2 == 0 else h) - 1))
        for i, v in enumerate(bbox)
    ]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    text = f"{label} {conf:.2f}" if conf is not None else label
    font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    (tw, th), bl = cv2.getTextSize(text, font, fs, ft)
    pad = 6
    lh  = th + bl + pad * 2
    if y1 > lh + 4:
        ly1, ly2, ty = y1 - lh, y1 - 2, y1 - bl - pad
    else:
        ly1, ly2, ty = y1 + 2, y1 + lh + 2, y1 + lh - bl
    lx2 = min(x1 + tw + pad * 2, w - 2)
    cv2.rectangle(frame, (x1, ly1), (lx2, ly2), color, -1)
    cv2.putText(frame, text, (x1 + pad, ty),
                font, fs, Config.C_WHITE, ft, cv2.LINE_AA)


def draw_person_compliance(frame, person):
    status = person['status']
    if status == 'compliant':
        color = Config.C_SAFE
        label = f"Person #{person['id']+1} | OK"
    elif status == 'partial':
        ms    = ", ".join(person['missing'])
        label = f"Person #{person['id']+1} | MISSING: {ms}"
        color = Config.C_WARNING
    else:
        color = Config.C_DANGER
        label = f"Person #{person['id']+1} | NO PPE"
    draw_box(frame, person['bbox'], label, color, thickness=3)


def draw_dashboard(frame, stats, fps, timings):
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (440, 215), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, "SAFETY SURVEILLANCE", (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, Config.C_WHITE, 2)
    lines = [
        f"Frame {stats['frames']}   FPS {fps:.1f}",
        f"Persons: {stats['persons']}   Violations: {stats['violations']}",
        f"Fire: {stats['fire']}   Smoke: {stats['smoke']}",
        f"Proximity critical: {stats['proximity_critical']}   "
        f"high: {stats['proximity_high']}",
        f"Session violations: {stats['total_violations']}",
        f"PPE:{timings.get('ppe',0)*1000:.0f}ms  "
        f"Fire:{timings.get('fire',0)*1000:.0f}ms",
    ]
    y = 58
    for line in lines:
        cv2.putText(frame, line, (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, Config.C_WHITE, 1)
        y += 24

    if stats['fire'] > 0:
        sc, st = Config.C_FIRE,     "STATUS: CRITICAL - FIRE/SMOKE"
    elif stats['proximity_critical'] > 0:
        sc, st = Config.C_FIRE,     "STATUS: CRITICAL - WORKER NEAR MACHINERY"
    elif stats['violations'] > 0 or stats['proximity_high'] > 0:
        sc, st = Config.C_WARNING,  "STATUS: WARNING - PPE VIOLATION"
    else:
        sc, st = Config.C_SAFE,     "STATUS: ALL CLEAR"
    cv2.rectangle(frame, (18, 183), (430, 206), sc, -1)
    cv2.putText(frame, st, (28, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, Config.C_BLACK, 2)


# ============================================================
# ALERT SYSTEM
# ============================================================
class AlertSystem:
    def __init__(self, enable_audio=False, enable_logging=True):
        self.enable_audio   = enable_audio and PYGAME_AVAILABLE
        self.enable_logging = enable_logging
        self._cooldown      = {}
        self.history        = deque(maxlen=1000)

        if self.enable_audio:
            try:
                pygame.mixer.init()
                self._sounds = {
                    'critical': self._beep(880, 0.4, 3),
                    'high':     self._beep(660, 0.3, 2),
                    'medium':   self._beep(440, 0.3, 1),
                }
            except Exception:
                self.enable_audio = False

        if self.enable_logging:
            log_dir = Path("surveillance_logs")
            log_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = log_dir / f"alerts_{ts}.json"
            self.snap_dir = log_dir / "snapshots"
            self.snap_dir.mkdir(exist_ok=True)

    def _beep(self, freq, dur, n):
        sr = 22050
        t  = np.arange(int(dur * sr)) / sr
        b  = np.sin(2 * np.pi * freq * t)
        g  = np.zeros(int(0.08 * sr))
        w  = np.concatenate([np.concatenate([b, g]) for _ in range(n)])
        w  = (w * 32767).astype(np.int16)
        return pygame.sndarray.make_sound(np.column_stack([w, w]))

    def alert(self, kind, severity, msg, frame=None, frame_no=None):
        now = datetime.now()
        key = f"{kind}_{severity}"
        if key in self._cooldown:
            if (now - self._cooldown[key]).total_seconds() < Config.ALERT_COOLDOWN:
                return
        self._cooldown[key] = now
        if self.enable_audio:
            lvl = {3: 'critical', 2: 'high', 1: 'medium'}.get(severity)
            if lvl and hasattr(self, '_sounds'):
                self._sounds[lvl].play()
        snap = None
        if frame is not None and self.enable_logging:
            snap = str(self.snap_dir /
                       f"{kind}_{now.strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(snap, frame)
        rec = dict(timestamp=now.isoformat(), frame=frame_no,
                   kind=kind, severity=severity, msg=msg, snapshot=snap)
        self.history.append(rec)
        tag = ['INFO','WARNING','HIGH','CRITICAL'][severity]
        print(f"[{tag}] {msg}")

    def save(self):
        if not self.enable_logging:
            return
        with open(self.log_file, 'w') as f:
            json.dump({'total': len(self.history),
                       'alerts': list(self.history)}, f, indent=2)
        print(f"Log saved: {self.log_file}")


# ============================================================
# MAIN SURVEILLANCE SYSTEM
# ============================================================
class SurveillanceSystem:

    def __init__(self, ppe_model_path, fire_model_path=None,
                 device='cpu', enable_audio=False,
                 enable_logging=True, show_dashboard=True):

        print("\n" + "="*55)
        print("  INTEGRATED SAFETY SURVEILLANCE SYSTEM")
        print("="*55)

        self.device         = device
        self.show_dashboard = show_dashboard
        self._lock          = threading.Lock()

        print(f"\n[PPE]  Loading: {ppe_model_path}")
        self.ppe_model = YOLO(ppe_model_path)
        self.ppe_model.overrides['conf']         = Config.PPE_CONF
        self.ppe_model.overrides['iou']          = Config.PPE_IOU
        self.ppe_model.overrides['max_det']      = Config.PPE_MAX_DET
        self.ppe_model.overrides['agnostic_nms'] = False
        print(f"       Classes: {list(self.ppe_model.names.values())}")
        print(f"       conf={Config.PPE_CONF}  iou={Config.PPE_IOU}")

        self.fire_model = None
        if fire_model_path and Path(fire_model_path).exists():
            print(f"\n[FIRE] Loading: {fire_model_path}")
            self.fire_model = YOLO(fire_model_path)
            self.fire_model.overrides['conf']         = Config.FIRE_CONF
            self.fire_model.overrides['iou']          = Config.FIRE_IOU
            self.fire_model.overrides['max_det']      = Config.FIRE_MAX_DET
            self.fire_model.overrides['agnostic_nms'] = False
            print(f"       Classes: {list(self.fire_model.names.values())}")
        else:
            print("\n[FIRE] No fire model — fire detection disabled.")

        self.alerts = AlertSystem(enable_audio, enable_logging)

        self.frame_count         = 0
        self.total_violations    = 0
        self.total_fire          = 0
        self.total_smoke         = 0
        self.total_proximity_crit = 0   # NEW: proximity critical events
        self.total_proximity_high = 0   # NEW: proximity high events
        self._fps_buf            = deque(maxlen=Config.STATS_WINDOW)
        self._last_t             = time.time()

        print("\n[OK]   System ready.\n" + "="*55 + "\n")

    def reset_stats(self):
        self.frame_count          = 0
        self.total_violations     = 0
        self.total_fire           = 0
        self.total_smoke          = 0
        self.total_proximity_crit = 0
        self.total_proximity_high = 0
        self.alerts.history.clear()
        self._fps_buf.clear()
        self._last_t = time.time()

    def _infer(self, frame):
        results = {}
        timings = {}
        half    = 'cuda' in str(self.device)

        def run_ppe():
            t0 = time.time()
            r  = self.ppe_model.predict(
                    source=frame, device=self.device, half=half, verbose=False)
            timings['ppe'] = time.time() - t0
            with self._lock: results['ppe'] = r

        def run_fire():
            if self.fire_model is None:
                results['fire'] = None
                return
            t0 = time.time()
            r  = self.fire_model.predict(
                    source=frame, device=self.device, half=half, verbose=False)
            timings['fire'] = time.time() - t0
            with self._lock: results['fire'] = r

        t1 = threading.Thread(target=run_ppe,  daemon=True)
        t2 = threading.Thread(target=run_fire, daemon=True)
        t1.start(); t2.start(); t1.join(); t2.join()
        return results, timings

    def _parse(self, result, names):
        persons    = []
        ppe_items  = []
        violations = []
        hazards    = []
        # NEW: collect machinery/vehicle as proximity hazard objects
        proximity_objects = []

        for box in result[0].boxes:
            bbox  = box.xyxy[0].cpu().numpy().tolist()
            conf  = float(box.conf[0])
            cname = names[int(box.cls[0])]

            if cname == Config.PERSON_CLASS:
                persons.append(bbox)
            elif cname in Config.PPE_CLASSES:
                ppe_items.append({'class': cname, 'bbox': bbox, 'conf': conf})
            elif cname in Config.VIOLATION_CLASSES:
                violations.append({'class': cname, 'bbox': bbox, 'conf': conf})
            elif cname in Config.HAZARD_CLASSES:
                hazards.append({'class': cname, 'bbox': bbox, 'conf': conf})
            elif cname in Config.PROXIMITY_HAZARD_CLASSES:
                # Previously just ignored — now captured for proximity analysis
                proximity_objects.append({'class': cname, 'bbox': bbox, 'conf': conf})

        return persons, ppe_items, violations, hazards, proximity_objects

    def process_frame(self, frame):
        self.frame_count += 1
        now = time.time()
        self._fps_buf.append(1.0 / max(now - self._last_t, 1e-6))
        self._last_t = now
        fps = float(np.mean(self._fps_buf))

        results, timings = self._infer(frame)

        # Unpack — now includes proximity_objects (machinery, vehicle)
        persons, ppe_items, violation_items, _, proximity_objects = self._parse(
            results['ppe'], self.ppe_model.names)

        fire_hazards = []
        if results.get('fire'):
            _, _, _, fire_hazards, _ = self._parse(
                results['fire'], self.fire_model.names)

        annotated = frame.copy()  # clean frame — we draw only what we need

        # ── Per-person compliance ─────────────────────────────────────────────
        compliance_list = []
        if persons:
            compliance_list = associate_ppe_to_persons(
                persons, ppe_items, violation_items)

            frame_violations = sum(1 for p in compliance_list
                                   if p['status'] != 'compliant')
            self.total_violations += frame_violations

            for person in compliance_list:
                draw_person_compliance(annotated, person)
                if person['status'] == 'none':
                    self.alerts.alert(
                        'ppe', Config.SEV_HIGH,
                        f"Person #{person['id']+1}: NO PPE detected!",
                        annotated, self.frame_count)
                elif person['status'] == 'partial':
                    self.alerts.alert(
                        'ppe', Config.SEV_MEDIUM,
                        f"Person #{person['id']+1} missing: "
                        f"{', '.join(person['missing'])}",
                        None, self.frame_count)

        # ── PROXIMITY HAZARD ANALYSIS ─────────────────────────────────────────
        proximity_events = []
        if compliance_list and proximity_objects:
            proximity_events = analyse_proximity(compliance_list, proximity_objects)

            frame_prox_crit = 0
            frame_prox_high = 0

            for event in proximity_events:
                draw_proximity_alert(annotated, event)

                if event['severity'] == Config.SEV_CRITICAL:
                    frame_prox_crit += 1
                    self.total_proximity_crit += 1
                    miss = ", ".join(event['missing']) or "PPE"
                    self.alerts.alert(
                        'proximity', Config.SEV_CRITICAL,
                        f"CRITICAL: Person #{event['person_id']+1} without {miss} "
                        f"is {event['distance_px']:.0f}px from {event['hazard_class']}!",
                        annotated, self.frame_count)
                else:
                    frame_prox_high += 1
                    self.total_proximity_high += 1
                    miss = ", ".join(event['missing']) or "PPE"
                    self.alerts.alert(
                        'proximity', Config.SEV_HIGH,
                        f"HIGH: Person #{event['person_id']+1} without {miss} "
                        f"near {event['hazard_class']} ({event['distance_px']:.0f}px)",
                        None, self.frame_count)

        # ── Fire/smoke overlay ────────────────────────────────────────────────
        # Draw fire/smoke boxes manually for consistency (no plot() clutter)

        for h in fire_hazards:
            if h['class'] == 'fire':
                self.total_fire += 1
                draw_box(annotated, h['bbox'], 'FIRE', Config.C_FIRE, h['conf'], 3)
                self.alerts.alert('fire', Config.SEV_CRITICAL,
                                  f"FIRE detected! conf={h['conf']:.2f}",
                                  annotated, self.frame_count)
            else:
                self.total_smoke += 1
                draw_box(annotated, h['bbox'], 'SMOKE', Config.C_SMOKE, h['conf'], 3)
                self.alerts.alert('smoke', Config.SEV_HIGH,
                                  f"SMOKE detected! conf={h['conf']:.2f}",
                                  annotated, self.frame_count)

        if self.show_dashboard:
            prox_crit = sum(1 for e in proximity_events
                            if e['severity'] == Config.SEV_CRITICAL)
            prox_high = sum(1 for e in proximity_events
                            if e['severity'] == Config.SEV_HIGH)
            stats = {
                'frames':            self.frame_count,
                'persons':           len(persons),
                'violations':        sum(1 for p in compliance_list
                                         if p['status'] != 'compliant'),
                'fire':              len([h for h in fire_hazards
                                          if h['class'] == 'fire']),
                'smoke':             len([h for h in fire_hazards
                                          if h['class'] == 'smoke']),
                'total_violations':  self.total_violations,
                'proximity_critical': prox_crit,
                'proximity_high':    prox_high,
            }
            draw_dashboard(annotated, stats, fps, timings)

        return annotated

    def run_image(self, image_path, output_path=None, display=True):
        print(f"Processing image: {image_path}")
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Cannot read: {image_path}")
        result = self.process_frame(frame)
        if output_path:
            cv2.imwrite(str(output_path), result)
            print(f"Saved: {output_path}")
        if display:
            cv2.imshow("Surveillance", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        self.alerts.save()
        return result

    def run_video(self, video_path, output_path=None,
                  display=True, skip_frames=0):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {video_path}")
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {W}x{H} @ {fps_src:.0f}fps | {total} frames")
        writer = None
        if output_path:
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps_src, (W, H))
        idx = 0
        t0  = time.time()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if skip_frames and idx % (skip_frames + 1) != 0:
                    idx += 1
                    continue
                result = self.process_frame(frame)
                if writer:
                    writer.write(result)
                if display:
                    cv2.imshow("Surveillance – Q quit", result)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                idx += 1
                if idx % 30 == 0:
                    el  = time.time() - t0
                    avg = idx / el if el else 0
                    eta = (total - idx) / avg if avg else 0
                    print(f"  {idx}/{total} | {avg:.1f} FPS | ETA {eta:.0f}s",
                          end='\r')
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            self.alerts.save()
            el = time.time() - t0
            print(f"\nDone: {idx} frames in {el:.1f}s "
                  f"({idx/el:.1f} FPS avg)")
            self._print_summary()

    def run_realtime(self, camera_id=0):
        print(f"Starting webcam {camera_id}  |  Q=quit  S=snapshot")
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                result = self.process_frame(frame)
                cv2.imshow("Surveillance – Q quit  S snapshot", result)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('s'):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fn = f"snapshot_{ts}.jpg"
                    cv2.imwrite(fn, result)
                    print(f"Snapshot saved: {fn}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.alerts.save()
            self._print_summary()

    def _print_summary(self):
        print("\n" + "="*45)
        print("  SESSION SUMMARY")
        print("="*45)
        print(f"  Frames processed   : {self.frame_count}")
        print(f"  PPE violations     : {self.total_violations}")
        print(f"  Proximity critical : {self.total_proximity_crit}")
        print(f"  Proximity high     : {self.total_proximity_high}")
        print(f"  Fire detections    : {self.total_fire}")
        print(f"  Smoke detections   : {self.total_smoke}")
        print(f"  Total alerts       : {len(self.alerts.history)}")
        print("="*45)


# ============================================================
# ENTRY POINT
# ============================================================
def main():
    import argparse
    p = argparse.ArgumentParser(description='Integrated Safety Surveillance')
    p.add_argument('--ppe-model',  required=True)
    p.add_argument('--fire-model', default=None)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument('--image',  type=str)
    grp.add_argument('--video',  type=str)
    grp.add_argument('--camera', type=int)
    p.add_argument('--output',       type=str,  default=None)
    p.add_argument('--device',       type=str,  default=None)
    p.add_argument('--skip-frames',  type=int,  default=0)
    p.add_argument('--no-display',   action='store_true')
    p.add_argument('--no-dashboard', action='store_true')
    p.add_argument('--audio',        action='store_true')
    p.add_argument('--no-logging',   action='store_true')
    args = p.parse_args()

    if not Path(args.ppe_model).exists():
        print(f"PPE model not found: {args.ppe_model}"); sys.exit(1)

    device = args.device
    if device is None:
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            device = 'cpu'

    sys_ = SurveillanceSystem(
        ppe_model_path  = args.ppe_model,
        fire_model_path = args.fire_model,
        device          = device,
        enable_audio    = args.audio,
        enable_logging  = not args.no_logging,
        show_dashboard  = not args.no_dashboard,
    )

    try:
        if args.image:
            sys_.run_image(args.image, args.output, not args.no_display)
        elif args.video:
            sys_.run_video(args.video, args.output,
                           not args.no_display, args.skip_frames)
        else:
            sys_.run_realtime(args.camera)
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        import traceback
        print(f"\nError: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()