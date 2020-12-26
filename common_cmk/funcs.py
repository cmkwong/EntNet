import numpy as np

def get_most_similar_vectors_pos(reference_vectors, vector, k):
    """
    :param reference_vectors: np.array(size=(m,n))
    :param vector: np.array(size=(n,1))
    :param k: int
    :return: [int] * k
    """
    magnitudes = np.linalg.norm(reference_vectors, ord=2, axis=1).reshape(-1,1) # (m, )
    similarities = (np.dot(reference_vectors, vector) / magnitudes).reshape(-1,) # (m, )
    sorted_similarities_list = list(np.argsort(similarities)[-k:]) # [float] * k
    sorted_similarities_list.reverse()
    return sorted_similarities_list
