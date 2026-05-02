
import torch
from pathlib import Path
import sys

def test_load():
    models = [
        ('ssd', 'models/SSD_19.pth', 'ssd'),
        ('frcnn_m', 'models/mobile-model.pth', 'frcnn_m'),
    ]
    
    for name, path, runner in models:
        print(f"\nTesting {name} at {path}...")
        try:
            if runner == 'ssd':
                from torchvision.models.detection import ssd300_vgg16
                model = ssd300_vgg16(num_classes=7)
                ckpt = torch.load(path, map_location='cpu')
                if 'model' in ckpt: ckpt = ckpt['model']
                model.load_state_dict(ckpt, strict=False)
                print(f"  SUCCESS: SSD loaded with strict=False")
            elif runner == 'frcnn_m':
                from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
                model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=7)
                ckpt = torch.load(path, map_location='cpu')
                if 'model' in ckpt: ckpt = ckpt['model']
                model.load_state_dict(ckpt, strict=False)
                print(f"  SUCCESS: FRCNN mobilenet loaded with strict=False")
        except Exception as e:
            print(f"  FAILED: {e}")

if __name__ == "__main__":
    test_load()
