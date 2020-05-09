import sys; print('Python %s on %s' % (sys.version, sys.platform))
import os
sys.path.insert(0, '../../')
sys.path.append('../')
print(sys.path)

import numpy as np
from scipy.spatial.distance import hamming

import pymc3 as pm
import theano.tensor as tt
from theano import *
import estimates



class dna_state:
    alphabet = frozenset("ATGC")

    state_fixed = ["A"] * 10
    n_letters = 10
    def __init__(self, length):
        dna_state.n_letters = length
        dna_state.state_fixed = ["A"] * dna_state.n_letters

    def score(state):
        state = state[0]
        return dna_state.n_letters - dna_state.n_letters * hamming(dna_state.state_fixed, list(state))

    def proposal(state):
        pos = np.random.choice(a=dna_state.n_letters, size=1)
        state_new_l = list(np.copy(state)[0])
        state_new_l[pos[0]] = np.random.choice(a=list(dna_state.alphabet), size=1)[0]
        state_new = np.array(''.join(state_new_l)).reshape(1)
        return state_new


Dna = dna_state(10)

A = ["A"] * 10


state = A


def weightingFunc(score):
    return 100 if score == 10 else 2

#def logProb(state):
#    return tt.log(weightingFunc(dna_state.score(state)))

#val = logProb(A)

x = np.array(['AAAA'])
y = shared(x)

with pm.Model() as model:
    #x = pm.Normal('x', mu=100500, sigma=42)
    #trace = pm.sample(1000)
    #weights = wang_landau(...)
    s = pm.WeightedScoreDistribution('S', scorer=dna_state.score, weighting=np.array([2]*11), cat=True, default_val='AAAAAAAAAA')
    trace = pm.sample(100000, cores=1, start={'S':['AAAAAAAAAA']},
                      step=pm.GenericCatMetropolis(vars=[s], proposal=dna_state.proposal),
                    compute_convergence_checks=False, chains=1, wl_weights=True)
    weights = np.exp(s.distribution.weights.get_value())


#add probability estimation
#TODO optimize it
score_trace = np.array([dna_state.score(x) for x in trace['S']]).astype('int32')
print(estimates.estimate_between(score_trace,  weights, 9, 11))

