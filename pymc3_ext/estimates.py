import numpy as np

#single trace
#TODO weights can be aquired by modelcontext
def estimate_between(score_trace, weights, a, b):
    """
    Function to estimate probability like P(a <= score < b)

    :param score_trace:
    :param weights:
    :param a:
    :param b:
    :return:
    """
    w = np.array([weights[s] for s in score_trace])
    mask = interval_mask(score_trace, a, b)
    return _estimate_numenator(score_trace, mask, w) / _estimate_denominator(score_trace, mask, w)

def interval_mask(score_trace, left, right):
    mask = np.array((score_trace >= left) & (score_trace < right))
    return mask

def bool_func_mask(score_trace, weights, bool_func):
    pass

def _estimate_numenator(score_trace, mask, weights):

    return np.mean(mask / weights)

def _estimate_denominator(score_trace, mask, weights):
    return np.mean(1 / weights)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def varianceOBM(score_trace, weights, left, right):
    mask = interval_mask(score_trace, left, right)

    h1 = _estimate_numenator(score_trace, mask, weights)
    h2 = _estimate_denominator(score_trace, mask, weights)

    num_arr = mask / weights[score_trace]
    denom_arr = 1/ weights[score_trace]

    N = score_trace.size
    batch_size = int(np.sqrt(N))

    batches_num = moving_average(num_arr, batch_size)
    batches_denom = moving_average(denom_arr, batch_size)

    C = (N * batch_size) / (N - batch_size)

    obm_num = C * np.mean((batches_num - h1) ** 2)
    obm_denom = C * np.mean((batches_denom - h2) ** 2)
    obm_cov = C * np.mean((batches_num - h1) * (batches_denom - h2))

    grad = np.array([1/h2, -h1/(h2 ** 2)]).reshape(2)
    cov_matrix = np.array([obm_num, obm_cov, obm_cov, obm_denom]).reshape(2, 2)

    return np.dot(grad, np.matmul(cov_matrix, grad))

