import sys;

print('Python %s on %s' % (sys.version, sys.platform))
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

class string2:


    def __init__(self, length):
        self.n_letters = length
        self.state_fixed = np.array(["A"] * dna_state.n_letters)
        alphabet = frozenset("ATGC")
        self.letters_list = list(alphabet)
        self.proposed = 0
        self.propose_letters = np.random.choice(a=self.letters_list, size=100000)
        self.propose_positions = np.random.choice(a=self.n_letters, size=100000)

    def score(self, state):
        return self.n_letters - np.sum(state != self.state_fixed)

    def proposal(self, state):
        if (self.proposed == 100000):
            self.propose_letters = np.random.choice(a=self.letters_list, size=100000)
            self.propose_positions = np.random.choice(a=self.n_letters, size=100000)
            self.proposed = 0
        state[self.propose_positions[self.proposed]] = self.propose_letters[self.proposed]
        self.proposed += 1
        return state

Dna = dna_state(10)

string_fixed = np.array(["A"] * 10)

def score(state):
    return


def weightingFunc(score):
    return 100 if score == 10 else 2


# def logProb(state):
#    return tt.log(weightingFunc(dna_state.score(state)))

# val = logProb(A)

x = np.array(['AAAA'])
y = shared(x)

s2 = string2(10)

with pm.Model() as model:
    # x = pm.Normal('x', mu=100500, sigma=42)
    # trace = pm.sample(1000)
    # weights = wang_landau(...)
    s = pm.WeightedScoreDistribution('S', scorer=s2.score, weighting=np.array([2] * 11), cat=True,
                                     default_val=string_fixed)
    trace = pm.sample(1000000, cores=1, start={'S': string_fixed},
                      step=pm.GenericCatMetropolis(vars=[s], proposal=s2.proposal),
                      compute_convergence_checks=False, chains=1, wl_weights=True)
    weights = np.exp(s.distribution.weights.get_value())

# add probability estimation
# TODO optimize it
score_trace = np.array([s2.score(x) for x in trace['S']]).astype('int32')
print("Estimated: ", estimates.estimate_between(score_trace, weights, 10, 11))

print("True answer is: ", 1/(4 ** 10))
print("variance estimation: ", estimates.varianceOBM(score_trace, weights, 10, 11))
