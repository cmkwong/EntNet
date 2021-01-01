import numpy as np
import torch
import torch.nn as nn

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

def unitVector_2d(tensors, dim=0):
    """
    :param tensors: torch.Tensor(size=(n,m))
    :return: return normalized torch.Tensor(size=(n,m))
    """
    magnitude = tensors.pow(2).sum(dim=dim).sqrt().unsqueeze(dim)
    unit_tensors = tensors / magnitude
    return unit_tensors

def unitVector_3d(tensors, dim=1):
    """
    :param tensors: torch.Tensor(size=(b,n,m))
    :return: return normalized torch.Tensor(size=(b,n,m))
    """
    magnitude = tensors.pow(2).sum(dim=dim).sqrt().unsqueeze(dim)
    unit_tensors = tensors / magnitude
    return unit_tensors