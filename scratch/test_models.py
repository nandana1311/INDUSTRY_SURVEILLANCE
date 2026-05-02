
import torch
from pathlib import Path
import sys

def test_load():
    models = [
        ('yolov3', 'models/YOLOv3spp-19.pt'),
        ('ssd', 'models/SSD_19.pth'),
        ('frcnn_r', 'models/resNetFpn-model-19.pth'),
        ('frcnn_m', 'models/mobile-model.pth'),
    ]
    
    for name, path in models:
        print(f"\nTesting {name} at {path}...")
        p = Path(path)
        if not p.exists():
            print(f"  FAILED: File does not exist")
            continue
            
        try:
            if 'yolo' in name:
                from ultralytics import YOLO
                m = YOLO(path)
                print(f"  SUCCESS: YOLO loaded")
            elif 'ssd' in name:
                from torchvision.models.detection import ssd300_vgg16
                model = ssd300_vgg16(num_classes=7)
                ckpt = torch.load(path, map_location='cpu')
                if 'model' in ckpt: model.load_state_dict(ckpt['model'])
                else: model.load_state_dict(ckpt)
                print(f"  SUCCESS: SSD loaded")
            elif 'frcnn' in name:
                from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn
                backbone = 'mobilenet' if 'm' in name else 'resnet'
                model = (fasterrcnn_mobilenet_v3_large_fpn(num_classes=7) 
                         if backbone=='mobilenet' else fasterrcnn_resnet50_fpn(num_classes=7))
                ckpt = torch.load(path, map_location='cpu')
                if 'model' in ckpt: model.load_state_dict(ckpt['model'])
                else: model.load_state_dict(ckpt)
                print(f"  SUCCESS: FRCNN {backbone} loaded")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_load()
