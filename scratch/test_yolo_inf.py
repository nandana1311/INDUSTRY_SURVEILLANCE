
import torch
from pathlib import Path
import cv2
import numpy as np

def test_inference():
    path = 'models/YOLOv3spp-19.pt'
    img_path = 'uploads/20260502_125831_Gemini_Generated_Image_ptx4cptx4cptx4cp.png' # Use an existing one
    
    print(f"Testing YOLOv3 inference...")
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=False)
        print("Model loaded.")
        res = model(img_path)
        print("Inference done.")
        ann = res.render()[0]
        print(f"Rendered. Shape: {ann.shape}")
        cv2.imwrite('scratch/test_yolo_out.jpg', ann)
        print("Saved to scratch/test_yolo_out.jpg")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inference()
