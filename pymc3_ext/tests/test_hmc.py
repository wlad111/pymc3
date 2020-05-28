import numpy as np
import numpy.testing as npt

from . import models
from pymc3_ext.step_methods.hmc.base_hmc import BaseHMC
from pymc3_ext.exceptions import SamplingError
import pymc3_ext
import pytest
import logging
from pymc3_ext.theanof import floatX
logger = logging.getLogger('pymc3_ext')

def test_leapfrog_reversible():
    n = 3
    np.random.seed(42)
    start, model, _ = models.non_normal(n)
    size = model.ndim
    scaling = floatX(np.random.rand(size))
    step = BaseHMC(vars=model.vars, model=model, scaling=scaling)
    step.integrator._logp_dlogp_func.set_extra_values({})
    p = floatX(step.potential.random())
    q = floatX(np.random.randn(size))
    start = step.integrator.compute_state(p, q)
    for epsilon in [.01, .1]:
        for n_steps in [1, 2, 3, 4, 20]:
            state = start
            for _ in range(n_steps):
                state = step.integrator.step(epsilon, state)
            for _ in range(n_steps):
                state = step.integrator.step(-epsilon, state)
            npt.assert_allclose(state.q, start.q, rtol=1e-5)
            npt.assert_allclose(state.p, start.p, rtol=1e-5)


def test_nuts_tuning():
    model = pymc3_ext.Model()
    with model:
        pymc3_ext.Normal("mu", mu=0, sigma=1)
        step = pymc3_ext.NUTS()
        trace = pymc3_ext.sample(10, step=step, tune=5, progressbar=False, chains=1)

    assert not step.tune
    assert np.all(trace['step_size'][5:] == trace['step_size'][5])

def test_nuts_error_reporting(caplog):
    model = pymc3_ext.Model()
    with caplog.at_level(logging.CRITICAL) and pytest.raises(SamplingError):
        with model:
            pymc3_ext.HalfNormal('a', sigma=1, transform=None, testval=-1)
            pymc3_ext.HalfNormal('b', sigma=1, transform=None)
            trace = pymc3_ext.sample(init='adapt_diag', chains=1)
        assert "Bad initial energy, check any log  probabilities that are inf or -inf: a        -inf\nb" in caplog.text

