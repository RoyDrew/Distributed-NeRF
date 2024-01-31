import faiss
import numpy as np

# def create_hnsw_index(vectors, dim):
#     """Create HNSW index using FAISS."""
#     index = faiss.IndexHNSWFlat(dim, 16)
#     index.hnsw.efConstruction = 40
#     index.add(vectors)
#     return index

# def search_similar_image(index, query_vector, k=1):
#     """Search for similar images in the index."""
#     _, I = index.search(query_vector, k)
#     return I[0]


def create_hnsw_index(vectors, dim):
    """Create HNSW index using FAISS."""
    index = faiss.IndexHNSWFlat(dim, 16)
    index.hnsw.efConstruction = 40
    index.add(vectors)
    return index

def search_similar_image(index, query_vector, k=1):
    """Search for similar images in the index and return distances as well."""
    D, I = index.search(query_vector, k)
    return I[0], D[0]  # returning both indices and distances
