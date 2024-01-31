import os
import glob
from models import model
from image_processing import image_to_vector, transform
from indexing import create_hnsw_index, search_similar_image
import numpy as np

folder_1 = '/home/air/multinerf/dataset/distributed_nerf/NewYork/1/images'
#folder_1 = '/home/air/multinerf/output/render_ours_sim/ours_NewYork2_test_preds_step_250000'
folder_2 = '/home/air/multinerf/dataset/distributed_nerf/NewYork/2/images'
class1_paths = glob.glob(os.path.join(folder_1, '*.JPG')) + glob.glob(os.path.join(folder_1, '*.png'))
class2_paths = glob.glob(os.path.join(folder_2, '*.jpg')) + glob.glob(os.path.join(folder_2, '*.png'))

def extract_features(image_paths):
    """Extract features for a list of images."""
    return np.vstack([image_to_vector(path, transform, model) for path in image_paths])

if __name__ == "__main__":
    class1_vectors = extract_features(class1_paths)
    class2_vectors = extract_features(class2_paths)
    print(class2_vectors.shape)
    dim = class1_vectors.shape[1]
    index = create_hnsw_index(class2_vectors, dim)

    # Finding the most similar image from class2 for each image in class1
    results = {}
    for idx, image_path in enumerate(class1_paths):
        query_vector = image_to_vector(image_path, transform, model)
        most_similar_image_index, distance = search_similar_image(index, query_vector)
        results[image_path] = (class2_paths[most_similar_image_index[0]], distance[0])

    # Displaying the most similar pairs with distances
    for class1_img, (class2_img, distance) in results.items():
        print(f"{class1_img} is most similar to {class2_img} with a distance of {distance:.4f}")
## 100 iamges times
## 文字
