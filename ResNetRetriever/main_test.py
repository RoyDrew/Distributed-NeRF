import torch
import torchvision.transforms as transforms
import torchvision.models as models
import faiss
from PIL import Image
import numpy as np
import os

# 初始化GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1. 加载预训练模型
model = models.resnet50(pretrained=True).to(device)
model = model.eval()

# 2. 图像处理: 定义预处理流程
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def image_to_vector(image_path):
    """将图像转换为向量"""
    with Image.open(image_path) as img:
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(tensor)
            return features.cpu().numpy()

# 构建图像路径列表
image_folder = 'airplanes'
image_paths = [os.path.join(image_folder, f'image_{str(i).zfill(4)}.jpg') for i in range(1, 801)]

# 3. 特征提取
def extract_features(image_paths):
    """对一组图像提取特征"""
    return np.vstack([image_to_vector(path) for path in image_paths])

# 4. 建立索引 (这里使用HNSW作为示例)
def create_hnsw_index(vectors, dim):
    index = faiss.IndexHNSWFlat(dim, 16)
    index.hnsw.efConstruction = 40
    index.add(vectors)
    return index

# 5. 检索
def search_similar_image(index, query_image_path, k=1):
    """在索引中搜索与查询图像最相似的图像"""
    query_vector = image_to_vector(query_image_path)
    _, I = index.search(query_vector, k)
    return I[0]

# 示例
vectors = extract_features(image_paths)
dim = vectors.shape[1]
index = create_hnsw_index(vectors, dim)

query_image_path = "/home/yuhai/ResNetRetriever/image_0012.jpg"  # 将此路径替换为你的查询图像路径
most_similar_image_index = search_similar_image(index, query_image_path)
print(f"Most similar image is at path: {image_paths[most_similar_image_index[0]]}")
