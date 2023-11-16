import base64
import logging
import pickle
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast
import hagelkorn
import mcbackend as mcb
import numpy as np
from mcbackend.npproto.utils import ndarray_from_numpy
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.basic import Constant
from pymc.backends.base import IBaseTrace
from pymc.model import Model
from pymc.pytensorf import PointFunc
from pymc.step_methods.compound import BlockedStep, CompoundStep, StatsBijection, check_step_emits_tune, flat_statname, flatten_steps
_log = logging.getLogger(__name__)

def find_data(pmodel: Model) -> List[mcb.DataVariable]:
    if False:
        i = 10
        return i + 15
    'Extracts data variables from a model.'
    observed_rvs = {pmodel.rvs_to_values[rv] for rv in pmodel.observed_RVs}
    dvars = []
    for (name, var) in pmodel.named_vars.items():
        dv = mcb.DataVariable(name)
        if isinstance(var, Constant):
            dv.value = ndarray_from_numpy(var.data)
        elif isinstance(var, SharedVariable):
            dv.value = ndarray_from_numpy(var.get_value())
        else:
            continue
        dv.dims = list(pmodel.named_vars_to_dims.get(name, []))
        dv.is_observed = var in observed_rvs
        dvars.append(dv)
    return dvars

def get_variables_and_point_fn(model: Model, initial_point: Mapping[str, np.ndarray]) -> Tuple[List[mcb.Variable], PointFunc]:
    if False:
        return 10
    'Get metadata on free, value and deterministic model variables.'
    vvars = model.value_vars
    vars = model.unobserved_value_vars
    point_fn = model.compile_fn(vars, inputs=vvars, on_unused_input='ignore', point_fn=True)
    point_fn = cast(PointFunc, point_fn)
    point = point_fn(initial_point)
    names = [v.name for v in vars]
    dtypes = [v.dtype for v in vars]
    shapes = [v.shape for v in point]
    deterministics = {d.name for d in model.deterministics}
    variables = [mcb.Variable(name=name, dtype=str(dtype), shape=list(shape), dims=list(model.named_vars_to_dims.get(name, [])), is_deterministic=name in deterministics) for (name, dtype, shape) in zip(names, dtypes, shapes)]
    return (variables, point_fn)

class ChainRecordAdapter(IBaseTrace):
    """Wraps an McBackend ``Chain`` as an ``IBaseTrace``."""

    def __init__(self, chain: mcb.Chain, point_fn: PointFunc, stats_bijection: StatsBijection) -> None:
        if False:
            print('Hello World!')
        self.chain = chain.cmeta.chain_number
        self.varnames = [v.name for v in chain.rmeta.variables]
        stats_dtypes = {s.name: np.dtype(s.dtype) for s in chain.rmeta.sample_stats}
        self.sampler_vars = [{sname: stats_dtypes[fname] for (fname, sname, is_obj) in sstats} for sstats in stats_bijection._stat_groups]
        self._chain = chain
        self._point_fn = point_fn
        self._statsbj = stats_bijection
        super().__init__()

    def record(self, draw: Mapping[str, np.ndarray], stats: Sequence[Mapping[str, Any]]):
        if False:
            print('Hello World!')
        values = self._point_fn(draw)
        value_dict = {n: v for (n, v) in zip(self.varnames, values)}
        stats_dict = self._statsbj.map(stats)
        for fname in self._statsbj.object_stats.keys():
            val_bytes = pickle.dumps(stats_dict[fname])
            val = base64.encodebytes(val_bytes).decode('ascii')
            stats_dict[fname] = np.array(val, dtype=str)
        return self._chain.append(value_dict, stats_dict)

    def __len__(self):
        if False:
            return 10
        return len(self._chain)

    def get_values(self, varname: str, burn=0, thin=1) -> np.ndarray:
        if False:
            while True:
                i = 10
        return self._chain.get_draws(varname, slice(burn, None, thin))

    def _get_stats(self, fname: str, slc: slice) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        'Wraps `self._chain.get_stats` but unpickles automatically.'
        values = self._chain.get_stats(fname, slc)
        if fname in self._statsbj.object_stats:
            objs = []
            for v in values:
                enc = str(v).encode('ascii')
                str_ = base64.decodebytes(enc)
                obj = pickle.loads(str_)
                objs.append(obj)
            return np.array(objs, dtype=object)
        return values

    def get_sampler_stats(self, stat_name: str, sampler_idx: Optional[int]=None, burn=0, thin=1) -> np.ndarray:
        if False:
            print('Hello World!')
        slc = slice(burn, None, thin)
        if sampler_idx is None and self._statsbj.n_samplers == 1:
            sampler_idx = 0
        if sampler_idx is not None:
            return self._get_stats(flat_statname(sampler_idx, stat_name), slc)
        stats_dict = {stat.name: self._get_stats(stat.name, slc) for stat in self._chain.rmeta.sample_stats if stat_name in stat.name}
        if not stats_dict:
            raise KeyError(f"No stat '{stat_name}' was recorded.")
        stats_list = self._statsbj.rmap(stats_dict)
        stats_arrays = []
        is_ragged = False
        for sd in stats_list:
            if not sd:
                is_ragged = True
                continue
            else:
                stats_arrays.append(tuple(sd.values())[0])
        if is_ragged:
            _log.debug("Stat '%s' was not recorded by all samplers.", stat_name)
        if len(stats_arrays) == 1:
            return stats_arrays[0]
        return np.array(stats_arrays).T

    def _slice(self, idx: slice) -> 'IBaseTrace':
        if False:
            i = 10
            return i + 15
        (start, stop, step) = idx.indices(len(self))
        indices = np.arange(start, stop, step)
        nchain = mcb.backends.numpy.NumPyChain(self._chain.cmeta, self._chain.rmeta, preallocate=len(indices))
        vnames = [v.name for v in nchain.variables.values()]
        snames = [s.name for s in nchain.sample_stats.values()]
        for i in indices:
            draw = self._chain.get_draws_at(i, var_names=vnames)
            stats = self._chain.get_stats_at(i, stat_names=snames)
            nchain.append(draw, stats)
        return ChainRecordAdapter(nchain, self._point_fn, self._statsbj)

    def point(self, idx: int) -> Dict[str, np.ndarray]:
        if False:
            print('Hello World!')
        return self._chain.get_draws_at(idx, [v.name for v in self._chain.variables.values()])

def make_runmeta_and_point_fn(*, initial_point: Mapping[str, np.ndarray], step: Union[CompoundStep, BlockedStep], model: Model) -> Tuple[mcb.RunMeta, PointFunc]:
    if False:
        while True:
            i = 10
    (variables, point_fn) = get_variables_and_point_fn(model, initial_point)
    check_step_emits_tune(step)
    sample_stats = []
    steps = flatten_steps(step)
    for (s, sm) in enumerate(steps):
        for (statname, (dtype, shape)) in sm.stats_dtypes_shapes.items():
            sname = flat_statname(s, statname)
            sshape = [-1 if s is None else s for s in shape or []]
            dt = np.dtype(dtype).name
            if dt == 'object':
                dt = 'str'
            svar = mcb.Variable(name=sname, dtype=dt, shape=sshape, undefined_ndim=shape is None)
            sample_stats.append(svar)
    coordinates = [mcb.Coordinate(dname, mcb.npproto.utils.ndarray_from_numpy(np.array(cvals))) for (dname, cvals) in model.coords.items() if cvals is not None]
    meta = mcb.RunMeta(rid=hagelkorn.random(), variables=variables, coordinates=coordinates, sample_stats=sample_stats, data=find_data(model))
    return (meta, point_fn)

def init_chain_adapters(*, backend: mcb.Backend, chains: int, initial_point: Mapping[str, np.ndarray], step: Union[CompoundStep, BlockedStep], model: Model) -> Tuple[mcb.Run, List[ChainRecordAdapter]]:
    if False:
        print('Hello World!')
    'Create an McBackend metadata description for the MCMC run.\n\n    Parameters\n    ----------\n    backend\n        An McBackend `Backend` instance.\n    chains\n        Number of chains to initialize.\n    initial_point\n        Dictionary mapping value variable names to initial values.\n    step : CompoundStep or BlockedStep\n        The step method that iterates the MCMC.\n    model : pm.Model\n        The current PyMC model.\n\n    Returns\n    -------\n    adapters\n        Chain recording adapters that wrap McBackend Chains in the PyMC IBaseTrace interface.\n    '
    (meta, point_fn) = make_runmeta_and_point_fn(initial_point=initial_point, step=step, model=model)
    run = backend.init_run(meta)
    statsbj = StatsBijection(step.stats_dtypes)
    adapters = [ChainRecordAdapter(chain=run.init_chain(chain_number=chain_number), point_fn=point_fn, stats_bijection=statsbj) for chain_number in range(chains)]
    return (run, adapters)