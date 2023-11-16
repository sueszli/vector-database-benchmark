from typing import List, Optional
import numpy as np
import pytensor.tensor as pt
from pytensor.graph.basic import Node
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.scalar.basic import Ceil, Clip, Floor, RoundHalfToEven
from pytensor.scalar.basic import clip as scalar_clip
from pytensor.tensor.math import ceil, clip, floor, round_half_to_even
from pytensor.tensor.variable import TensorConstant
from pymc.logprob.abstract import MeasurableElemwise, _logcdf, _logprob
from pymc.logprob.rewriting import PreserveRVMappings, measurable_ir_rewrites_db
from pymc.logprob.utils import CheckParameterValue

class MeasurableClip(MeasurableElemwise):
    """A placeholder used to specify a log-likelihood for a clipped RV sub-graph."""
    valid_scalar_types = (Clip,)
measurable_clip = MeasurableClip(scalar_clip)

@node_rewriter(tracks=[clip])
def find_measurable_clips(fgraph: FunctionGraph, node: Node) -> Optional[List[MeasurableClip]]:
    if False:
        i = 10
        return i + 15
    rv_map_feature: Optional[PreserveRVMappings] = getattr(fgraph, 'preserve_rv_mappings', None)
    if rv_map_feature is None:
        return None
    if not rv_map_feature.request_measurable(node.inputs):
        return None
    (base_var, lower_bound, upper_bound) = node.inputs
    lower_bound = lower_bound if lower_bound is not base_var else pt.constant(-np.inf)
    upper_bound = upper_bound if upper_bound is not base_var else pt.constant(np.inf)
    clipped_rv = measurable_clip.make_node(base_var, lower_bound, upper_bound).outputs[0]
    return [clipped_rv]
measurable_ir_rewrites_db.register('find_measurable_clips', find_measurable_clips, 'basic', 'censoring')

@_logprob.register(MeasurableClip)
def clip_logprob(op, values, base_rv, lower_bound, upper_bound, **kwargs):
    if False:
        while True:
            i = 10
    'Logprob of a clipped censored distribution\n\n    The probability is given by\n    .. math::\n        \\begin{cases}\n            0 & \\text{for } x < lower, \\\\\n            \\text{CDF}(lower, dist) & \\text{for } x = lower, \\\\\n            \\text{P}(x, dist) & \\text{for } lower < x < upper, \\\\\n            1-\\text{CDF}(upper, dist) & \\text {for} x = upper, \\\\\n            0 & \\text{for } x > upper,\n        \\end{cases}\n\n    '
    (value,) = values
    base_rv_op = base_rv.owner.op
    base_rv_inputs = base_rv.owner.inputs
    logprob = _logprob(base_rv_op, (value,), *base_rv_inputs, **kwargs)
    logcdf = _logcdf(base_rv_op, value, *base_rv_inputs, **kwargs)
    if base_rv_op.name:
        logprob.name = f'{base_rv_op}_logprob'
        logcdf.name = f'{base_rv_op}_logcdf'
    (is_lower_bounded, is_upper_bounded) = (False, False)
    if not (isinstance(upper_bound, TensorConstant) and np.all(np.isinf(upper_bound.value))):
        is_upper_bounded = True
        logccdf = pt.log1mexp(logcdf)
        if base_rv.dtype.startswith('int'):
            logccdf = pt.logaddexp(logccdf, logprob)
        logprob = pt.switch(pt.eq(value, upper_bound), logccdf, pt.switch(pt.gt(value, upper_bound), -np.inf, logprob))
    if not (isinstance(lower_bound, TensorConstant) and np.all(np.isneginf(lower_bound.value))):
        is_lower_bounded = True
        logprob = pt.switch(pt.eq(value, lower_bound), logcdf, pt.switch(pt.lt(value, lower_bound), -np.inf, logprob))
    if is_lower_bounded and is_upper_bounded:
        logprob = CheckParameterValue('lower_bound <= upper_bound')(logprob, pt.all(pt.le(lower_bound, upper_bound)))
    return logprob

class MeasurableRound(MeasurableElemwise):
    """A placeholder used to specify a log-likelihood for a clipped RV sub-graph."""
    valid_scalar_types = (RoundHalfToEven, Floor, Ceil)

@node_rewriter(tracks=[ceil, floor, round_half_to_even])
def find_measurable_roundings(fgraph: FunctionGraph, node: Node) -> Optional[List[MeasurableRound]]:
    if False:
        i = 10
        return i + 15
    rv_map_feature: Optional[PreserveRVMappings] = getattr(fgraph, 'preserve_rv_mappings', None)
    if rv_map_feature is None:
        return None
    if not rv_map_feature.request_measurable(node.inputs):
        return None
    [base_var] = node.inputs
    rounded_op = MeasurableRound(node.op.scalar_op)
    rounded_rv = rounded_op.make_node(base_var).default_output()
    rounded_rv.name = node.outputs[0].name
    return [rounded_rv]
measurable_ir_rewrites_db.register('find_measurable_roundings', find_measurable_roundings, 'basic', 'censoring')

@_logprob.register(MeasurableRound)
def round_logprob(op, values, base_rv, **kwargs):
    if False:
        print('Hello World!')
    'Logprob of a rounded censored distribution\n\n    The probability of a distribution rounded to the nearest integer is given by\n    .. math::\n        \\begin{cases}\n            \\text{CDF}(x+\\frac{1}{2}, dist) - \\text{CDF}(x-\\frac{1}{2}, dist) & \\text{for } x \\in \\mathbb{Z}, \\\\\n            0 & \\text{otherwise},\n        \\end{cases}\n\n    The probability of a distribution rounded up is given by\n    .. math::\n        \\begin{cases}\n            \\text{CDF}(x, dist) - \\text{CDF}(x-1, dist) & \\text{for } x \\in \\mathbb{Z}, \\\\\n            0 & \\text{otherwise},\n        \\end{cases}\n\n    The probability of a distribution rounded down is given by\n    .. math::\n        \\begin{cases}\n            \\text{CDF}(x+1, dist) - \\text{CDF}(x, dist) & \\text{for } x \\in \\mathbb{Z}, \\\\\n            0 & \\text{otherwise},\n        \\end{cases}\n\n    '
    (value,) = values
    if isinstance(op.scalar_op, RoundHalfToEven):
        value = pt.round(value)
        value_upper = value + 0.5
        value_lower = value - 0.5
    elif isinstance(op.scalar_op, Floor):
        value = pt.floor(value)
        value_upper = value + 1.0
        value_lower = value
    elif isinstance(op.scalar_op, Ceil):
        value = pt.ceil(value)
        value_upper = value
        value_lower = value - 1.0
    else:
        raise TypeError(f'Unsupported scalar_op {op.scalar_op}')
    base_rv_op = base_rv.owner.op
    base_rv_inputs = base_rv.owner.inputs
    logcdf_upper = _logcdf(base_rv_op, value_upper, *base_rv_inputs, **kwargs)
    logcdf_lower = _logcdf(base_rv_op, value_lower, *base_rv_inputs, **kwargs)
    if base_rv_op.name:
        logcdf_upper.name = f'{base_rv_op}_logcdf_upper'
        logcdf_lower.name = f'{base_rv_op}_logcdf_lower'
    from pymc.math import logdiffexp
    return logdiffexp(logcdf_upper, logcdf_lower)