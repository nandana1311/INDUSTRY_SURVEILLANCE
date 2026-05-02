
import torch
from pathlib import Path

def inspect_ckpt(path):
    print(f"\nInspecting {path}...")
    try:
        ckpt = torch.load(path, map_location='cpu')
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        print("Keys count:", len(state_dict.keys()))
        
        # Look for class counts
        for k in state_dict.keys():
            if 'cls_score' in k or 'class_predict' in k or 'cls_logits' in k or 'nc' in k:
                print(f"  {k}: {state_dict[k].shape if hasattr(state_dict[k], 'shape') else state_dict[k]}")
                
    except Exception as e:
        print(f"  FAILED: {e}")

if __name__ == "__main__":
    inspect_ckpt('models/YOLOv3spp-19.pt')
    inspect_ckpt('models/SSD_19.pth')
    inspect_ckpt('models/mobile-model.pth')
