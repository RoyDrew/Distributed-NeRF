import torch
import torchvision.transforms as transforms
from PIL import Image
import models

# def image_to_vector(image_path, transform, model):
#     """Convert image to vector."""
#     with Image.open(image_path) as img:
#         tensor = transform(img).unsqueeze(0).to(models.device)
#         with torch.no_grad():
#             features = model(tensor)
#             return features.cpu().numpy()
#             # return features.squeeze().cpu().numpy()

def image_to_vector(image_path, transform, model):
    """Convert image to vector."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        tensor = transform(img).unsqueeze(0).to(models.device)
        tensor = tensor
        with torch.no_grad():
            features = model(tensor)
            return features.squeeze(-1).squeeze(-1).cpu().numpy()  # Adding squeeze to reshape the output


# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
