
import torch
from pathlib import Path
import sys

def test_load():
    models = [
        ('yolov3', 'models/YOLOv3spp-19.pt', 'yolo'),
        ('ssd', 'models/SSD_19.pth', 'ssd'),
        ('frcnn_r', 'models/resNetFpn-model-19.pth', 'frcnn_r'),
        ('frcnn_m', 'models/mobile-model.pth', 'frcnn_m'),
    ]
    
    for name, path, runner in models:
        print(f"\n{'='*20}\nTesting {name} at {path}...")
        p = Path(path)
        if not p.exists():
            print(f"  FAILED: File does not exist")
            continue
            
        try:
            if runner == 'yolo':
                try:
                    from ultralytics import YOLO
                    m = YOLO(path)
                    print(f"  SUCCESS: YOLO loaded via ultralytics")
                except Exception as e:
                    print(f"  Ultralytics failed: {e}. Trying hub...")
                    model = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=False)
                    print(f"  SUCCESS: YOLO loaded via hub")
                    
            elif runner == 'ssd':
                for nc in [7, 11, 12, 21]:
                    try:
                        from torchvision.models.detection import ssd300_vgg16
                        model = ssd300_vgg16(num_classes=nc)
                        ckpt = torch.load(path, map_location='cpu')
                        if 'model' in ckpt: model.load_state_dict(ckpt['model'])
                        else: model.load_state_dict(ckpt)
                        print(f"  SUCCESS: SSD loaded with num_classes={nc}")
                        break
                    except Exception as e:
                        print(f"  SSD failed with nc={nc}: {str(e)[:100]}...")
                        
            elif 'frcnn' in runner:
                backbone = 'mobilenet' if 'm' in runner else 'resnet'
                from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn
                for nc in [7, 11, 12, 21]:
                    try:
                        model = (fasterrcnn_mobilenet_v3_large_fpn(num_classes=nc) 
                                 if backbone=='mobilenet' else fasterrcnn_resnet50_fpn(num_classes=nc))
                        ckpt = torch.load(path, map_location='cpu')
                        if 'model' in ckpt: model.load_state_dict(ckpt['model'])
                        else: model.load_state_dict(ckpt)
                        print(f"  SUCCESS: FRCNN {backbone} loaded with num_classes={nc}")
                        break
                    except Exception as e:
                        print(f"  FRCNN {backbone} failed with nc={nc}: {str(e)[:100]}...")
        except Exception as e:
            print(f"  FAILED: {e}")

if __name__ == "__main__":
    test_load()
