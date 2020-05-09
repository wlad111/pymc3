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
    mask = interval_mask(score_trace, a, b)
    return _estimate_numenator(score_trace, mask, weights) / _estimate_denominator(score_trace, mask, weights)

def interval_mask(score_trace, left, right):
    mask = np.array((score_trace >= left) & (score_trace < right))
    return mask

def bool_func_mask(score_trace, weights, bool_func):
    pass

def _estimate_numenator(score_trace, mask, weights):
    return np.mean(mask / weights[score_trace])

def _estimate_denominator(score_trace, mask, weights):
    return np.mean(1 / weights[score_trace])
