import sys, os, glob, torch, cv2
sys.path.insert(0, '.')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

mpath = 'models/YOLOv3spp-19.pt'
hub_dir = torch.hub.get_dir()
yv5_dirs = sorted(glob.glob(os.path.join(hub_dir, '*yolov5*')))
yv5_path = yv5_dirs[-1]
sys.path.insert(0, yv5_path)

from models.yolo import Model
cfg = os.path.join(yv5_path, 'models', 'hub', 'yolov3-spp.yaml')
new_model = Model(cfg, nc=6)
new_model.eval()

ckpt = torch.load(mpath, map_location='cpu')
old_sd = ckpt['model']

old_tensors = [v for k,v in old_sd.items()
               if 'total_ops' not in k and 'total_params' not in k
               and isinstance(v, torch.Tensor)]

new_sd   = new_model.state_dict()
new_keys = list(new_sd.keys())
mapped   = {}
old_ptr  = 0
for nk in new_keys:
    ns = new_sd[nk].shape
    while old_ptr < len(old_tensors):
        ov = old_tensors[old_ptr]
        old_ptr += 1
        if ov.shape == ns:
            mapped[nk] = ov
            break

# Correct detect head mapping by in_channels
_detect_map = {}
for k, v in old_sd.items():
    if 'Conv2d.weight' in k and v.dim() == 4 and v.shape[0] == 33:
        in_ch = v.shape[1]
        bias_k = k.replace('.weight', '.bias')
        bias_v = old_sd.get(bias_k)
        _detect_map[in_ch] = (v, bias_v)

for wk, bk in [('model.28.m.0.weight','model.28.m.0.bias'),
                ('model.28.m.1.weight','model.28.m.1.bias'),
                ('model.28.m.2.weight','model.28.m.2.bias')]:
    in_ch = new_sd[wk].shape[1]
    if in_ch in _detect_map:
        w, b = _detect_map[in_ch]
        mapped[wk] = w
        if b is not None:
            mapped[bk] = b
        print('Detect head mapped: wk=%s in_ch=%d' % (wk, in_ch))

new_model.load_state_dict(mapped, strict=False)
new_model.nc = 6
new_model.names = {0:'Hardhat',1:'Mask',2:'NO-Hardhat',3:'NO-Mask',4:'NO-Safety Vest',5:'Safety Vest'}
print('Total remapped: %d/%d' % (len(mapped), len(new_keys)))

# Use any existing jpg in the project
import glob as g2
imgs = g2.glob('outputs/*.jpg') + g2.glob('outputs/*.png') + g2.glob('*.jpg')
if not imgs:
    import numpy as np, urllib.request
    urllib.request.urlretrieve('https://ultralytics.com/images/zidane.jpg', 'test_zidane.jpg')
    imgs = ['test_zidane.jpg']

from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

img_orig = cv2.imread(imgs[0])
print('Testing on:', imgs[0], img_orig.shape)
img_lb, ratio, pad = letterbox(img_orig, new_shape=640)
img_t = img_lb[:,:,::-1].transpose(2,0,1).copy()
img_t = torch.from_numpy(img_t).float().unsqueeze(0) / 255.0

with torch.no_grad():
    out = new_model(img_t)
    preds = out[0] if isinstance(out, tuple) else out

preds = non_max_suppression(preds, conf_thres=0.05, iou_thres=0.45)[0]
n = len(preds) if preds is not None else 0
print('Detections (conf>0.05):', n)
if preds is not None and len(preds):
    for row in preds.cpu().numpy()[:5]:
        box = row[:4]; score = row[4]; cls = int(row[5])
        name = new_model.names.get(cls, str(cls))
        print('  class=%d (%s) conf=%.3f' % (cls, name, score))
