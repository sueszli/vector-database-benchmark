from typing import Tuple
import numpy as np
import numpy.random as nr
from pymc.blocking import RaveledVars, StatsType
from pymc.model import modelcontext
from pymc.pytensorf import compile_pymc, join_nonshared_inputs, make_shared_replacements
from pymc.step_methods.arraystep import ArrayStepShared
from pymc.step_methods.compound import Competence
from pymc.util import get_value_vars_from_user_vars
from pymc.vartypes import continuous_types
__all__ = ['Slice']
LOOP_ERR_MSG = 'max slicer iters %d exceeded'

class Slice(ArrayStepShared):
    """
    Univariate slice sampler step method.

    Parameters
    ----------
    vars: list
        List of value variables for sampler.
    w: float
        Initial width of slice (Defaults to 1).
    tune: bool
        Flag for tuning (Defaults to True).
    model: PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).

    """
    name = 'slice'
    default_blocked = False
    stats_dtypes_shapes = {'tune': (bool, []), 'nstep_out': (int, []), 'nstep_in': (int, [])}

    def __init__(self, vars=None, w=1.0, tune=True, model=None, iter_limit=np.inf, **kwargs):
        if False:
            while True:
                i = 10
        model = modelcontext(model)
        self.w = np.asarray(w).copy()
        self.tune = tune
        self.n_tunes = 0.0
        self.iter_limit = iter_limit
        if vars is None:
            vars = model.continuous_value_vars
        else:
            vars = get_value_vars_from_user_vars(vars, model)
        point = model.initial_point()
        shared = make_shared_replacements(point, vars, model)
        ([logp], raveled_inp) = join_nonshared_inputs(point=point, outputs=[model.logp()], inputs=vars, shared_inputs=shared)
        self.logp = compile_pymc([raveled_inp], logp)
        self.logp.trust_input = True
        super().__init__(vars, shared)

    def astep(self, apoint: RaveledVars) -> Tuple[RaveledVars, StatsType]:
        if False:
            while True:
                i = 10
        q0_val = apoint.data
        if q0_val.shape != self.w.shape:
            self.w = np.resize(self.w, len(q0_val))
        nstep_out = nstep_in = 0
        q = np.copy(q0_val)
        ql = np.copy(q0_val)
        qr = np.copy(q0_val)
        logp = self.logp
        for (i, wi) in enumerate(self.w):
            y = logp(q) - nr.standard_exponential()
            ql[i] = q[i] - nr.uniform() * wi
            qr[i] = ql[i] + wi
            cnt = 0
            while y <= logp(ql):
                ql[i] -= wi
                cnt += 1
                if cnt > self.iter_limit:
                    raise RuntimeError(LOOP_ERR_MSG % self.iter_limit)
            nstep_out += cnt
            cnt = 0
            while y <= logp(qr):
                qr[i] += wi
                cnt += 1
                if cnt > self.iter_limit:
                    raise RuntimeError(LOOP_ERR_MSG % self.iter_limit)
            nstep_out += cnt
            cnt = 0
            q[i] = nr.uniform(ql[i], qr[i])
            while y > logp(q):
                if q[i] > q0_val[i]:
                    qr[i] = q[i]
                elif q[i] < q0_val[i]:
                    ql[i] = q[i]
                q[i] = nr.uniform(ql[i], qr[i])
                cnt += 1
                if cnt > self.iter_limit:
                    raise RuntimeError(LOOP_ERR_MSG % self.iter_limit)
            nstep_in += cnt
            if self.tune:
                self.w[i] = wi * (self.n_tunes / (self.n_tunes + 1)) + (qr[i] - ql[i]) / (self.n_tunes + 1)
            qr[i] = ql[i] = q[i]
        if self.tune:
            self.n_tunes += 1
        stats = {'tune': self.tune, 'nstep_out': nstep_out, 'nstep_in': nstep_in}
        return (RaveledVars(q, apoint.point_map_info), [stats])

    @staticmethod
    def competence(var, has_grad):
        if False:
            i = 10
            return i + 15
        if var.dtype in continuous_types:
            if not has_grad and var.ndim == 0:
                return Competence.PREFERRED
            return Competence.COMPATIBLE
        return Competence.INCOMPATIBLE