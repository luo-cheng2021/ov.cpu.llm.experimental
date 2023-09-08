import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    summation = e_x.sum(axis=-1, keepdims=True)
    return e_x / summation

def get_top_k_logits(scores, top_k):
    """
    perform top-k sampling

    Parameters:
      scores - model output logits
      top_k - number of elements with highest probability to select
    """
    filter_value = -float("inf")
    top_k = min(max(top_k, 1), scores.shape[-1])
    top_k_scores = -np.sort(-scores)[:, :top_k]
    indices_to_remove = scores < np.min(top_k_scores)
    filtred_scores = np.ma.array(scores, mask=indices_to_remove,
                                 fill_value=filter_value).filled()
    return filtred_scores


def create_sinusoidal_positions(num_pos: int, dim: int):
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    sinusoid_inp = np.einsum("i , j -> i j", np.arange(num_pos, dtype=np.float), inv_freq).astype("float32")
    return np.sin(sinusoid_inp), np.cos(sinusoid_inp)