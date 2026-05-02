
import torch
from torchvision.models.detection import ssd300_vgg16
from torchvision import transforms
from PIL import Image as PILImage
import numpy as np

def test_ssd_scores():
    path = 'models/SSD_19.pth'
    img_path = 'uploads/20260502_125831_Gemini_Generated_Image_ptx4cptx4cptx4cp.png'
    
    device = torch.device('cpu')
    model = ssd300_vgg16(num_classes=7)
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    img = PILImage.open(img_path).convert('RGB')
    tensor = transforms.ToTensor()(img).unsqueeze(0)
    
    with torch.no_grad():
        preds = model(tensor)[0]
    
    scores = preds['scores'].numpy()
    print(f"Total detections: {len(scores)}")
    print(f"Scores (first 20): {scores[:20]}")
    print(f"Scores > 0.5: {np.sum(scores > 0.5)}")
    print(f"Scores > 0.9: {np.sum(scores > 0.9)}")

if __name__ == "__main__":
    test_ssd_scores()
