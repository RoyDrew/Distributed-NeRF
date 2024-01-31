import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import numpy as np

# Load pre-trained model and remove last fully connected layer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
model = model.eval()

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# def image_to_vector(image_path):
#     """Convert image to vector."""
#     with Image.open(image_path) as img:
#         tensor = transform(img).unsqueeze(0).to(device)
#         with torch.no_grad():
#             features = model(tensor)
#             return features.squeeze(-1).squeeze(-1).cpu().numpy()
def image_to_vector(image_path):
    """Convert image to vector."""
    with Image.open(image_path) as img:
        tensor = transform(img).unsqueeze(0).to(device)
        # .unsqueeze(0): This adds an extra dimension to the tensor. 
        # This is done to mimic a batch of images, 
        # even though we're only processing one image. 
        # Neural network models usually expect a batch of inputs, 
        # so we need to adjust the shape accordingly.
        with torch.no_grad():
            features = model(tensor)
            return features.squeeze(-1).squeeze(-1).cpu().numpy().squeeze()  # Added another squeeze here
        # features.squeeze(-1).squeeze(-1): This removes the spatial dimensions 
        # (height and width) of the feature tensor.
        # .squeeze(): Ensures that the numpy array is one-dimensional 
        # by removing any dimensions of size 1.

def compute_similarity(vector_a, vector_b):
    """Compute similarity between two vectors using dot product."""
    return np.dot(vector_a, vector_b)

class1_paths = [os.path.join('class1', f'1_{i}.JPG') for i in range(1, 5)]
class2_paths = [os.path.join('class2', f'2_{i}.JPG') for i in range(1, 5)]

# Extract features for all images
class1_vectors = np.array([image_to_vector(path) for path in class1_paths])
class2_vectors = np.array([image_to_vector(path) for path in class2_paths])

# Compute similarity between class1 and class2 images
results = {}
for idx, class1_image in enumerate(class1_paths):
    best_similarity = -float('inf')
    # Set the initial best similarity to negative infinity. 
    # This ensures that any similarity computed will be larger than this initial value.
    best_match = None
    for jdx, class2_image in enumerate(class2_paths):
        similarity = compute_similarity(class1_vectors[idx], class2_vectors[jdx])
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = class2_image
    results[class1_image] = (best_match, best_similarity)

# Display results
for class1_image, (class2_image, similarity) in results.items():
    print(f"{class1_image} is most similar to {class2_image} with a similarity score of {similarity:.4f}")

if __name__ == "__main__":
    pass  # The code is already executed in the global scope
