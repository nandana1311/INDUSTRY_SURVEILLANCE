
import torch
from torchvision.models.detection import ssd300_vgg16

def compare_keys():
    path = 'models/SSD_19.pth'
    model = ssd300_vgg16(num_classes=7)
    model_keys = set(model.state_dict().keys())
    
    ckpt = torch.load(path, map_location='cpu')
    ckpt_keys = set((ckpt['model'] if 'model' in ckpt else ckpt).keys())
    
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    
    print(f"Missing keys ({len(missing)}):", list(missing)[:10])
    print(f"Unexpected keys ({len(unexpected)}):", list(unexpected)[:10])

if __name__ == "__main__":
    compare_keys()
