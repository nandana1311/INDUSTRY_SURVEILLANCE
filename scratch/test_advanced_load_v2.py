
import torch
from pathlib import Path
import sys

def test_yolov3_load():
    path = 'models/YOLOv3spp-19.pt'
    print(f"Testing YOLOv3 load with state_dict...")
    try:
        from ultralytics import YOLO
        # Initialize model with yaml
        model = YOLO('yolov3-spp.yaml')
        ckpt = torch.load(path, map_location='cpu')
        # YOLOv8 models store the actual torch model in .model
        model.model.load_state_dict(ckpt, strict=False)
        print("SUCCESS: YOLOv3-spp loaded via load_state_dict")
    except Exception as e:
        print(f"FAILED YOLOv3: {e}")

def test_mobilenet_no_fpn():
    path = 'models/mobile-model.pth'
    print(f"\nTesting MobileNet FasterRCNN without FPN...")
    try:
        from torchvision.models.detection import FasterRCNN
        from torchvision.models.mobilenetv3 import mobilenet_v3_large
        
        backbone = mobilenet_v3_large().features
        # The checkpoint has 1280 channels in RPN head, which matches MobileNetV3 large output
        backbone.out_channels = 1280
        
        model = FasterRCNN(backbone, num_classes=7)
        ckpt = torch.load(path, map_location='cpu')
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print("SUCCESS: MobileNet FasterRCNN (No FPN) loaded")
        
    except Exception as e:
        print(f"FAILED MobileNet: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_yolov3_load()
    test_mobilenet_no_fpn()
