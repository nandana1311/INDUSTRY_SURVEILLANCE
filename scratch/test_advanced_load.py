
import torch
from pathlib import Path
import sys

def test_yolov3_load():
    path = 'models/YOLOv3spp-19.pt'
    print(f"Testing YOLOv3 load with architecture...")
    try:
        from ultralytics import YOLO
        # Try loading as a YOLOv3-spp model
        model = YOLO('yolov3-spp.yaml') # This loads the architecture
        model.load(path) # This loads the weights
        print("SUCCESS: YOLOv3-spp loaded via ultralytics architecture + weights")
    except Exception as e:
        print(f"FAILED YOLOv3: {e}")

def test_mobilenet_no_fpn():
    path = 'models/mobile-model.pth'
    print(f"\nTesting MobileNet FasterRCNN without FPN...")
    try:
        from torchvision.models.detection import FasterRCNN
        from torchvision.models.detection.backbone_utils import mobilenet_backbone_scaler
        from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
        
        # Create backbone without FPN
        backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).features
        backbone.out_channels = 1280
        
        model = FasterRCNN(backbone, num_classes=7)
        ckpt = torch.load(path, map_location='cpu')
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print("SUCCESS: MobileNet FasterRCNN (No FPN) loaded with strict=False")
        
        # Check specific layers
        print(f"  RPN head weight shape in model: {model.rpn.head.conv[0].weight.shape}")
        
    except Exception as e:
        print(f"FAILED MobileNet: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_yolov3_load()
    test_mobilenet_no_fpn()
