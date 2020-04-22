import numpy as np
from scipy.spatial.distance import hamming

import pymc3 as pm
import theano.tensor as tt
from theano import *

class dna_state:
    alphabet = frozenset("ATGC")

    state_fixed = ["A"] * 4
    n_letters = 4
    def __init__(self, length):
        dna_state.n_letters = length
        dna_state.state_fixed = ["A"] * dna_state.n_letters

    def score(state):
        return dna_state.n_letters - dna_state.n_letters * hamming(dna_state.state_fixed, list(state))

    def proposal(state):
        pos = np.random.choice(a=dna_state.n_letters, size=1)
        state_new = np.copy(state)
        state_new[pos] = np.random.choice(a=list(dna_state.alphabet), size=1)
        return state_new

Dna = dna_state(4)

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
    s = pm.WeightedScoreDistribution('S', scorer=dna_state.score, weighting=np.array([2]*5), cat=True, default_val='AAAA')
    trace = pm.sample(2000, cores=1, start={'S':['AAAA']},
                      step=pm.GenericCatMetropolis(vars=[s], proposal=dna_state.proposal))
