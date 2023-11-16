from typing import Callable, List, Tuple
from thinc.api import Model, to_numpy
from thinc.types import Ints1d, Ragged
from ..util import registry

@registry.layers('spacy.extract_spans.v1')
def extract_spans() -> Model[Tuple[Ragged, Ragged], Ragged]:
    if False:
        while True:
            i = 10
    'Extract spans from a sequence of source arrays, as specified by an array\n    of (start, end) indices. The output is a ragged array of the\n    extracted spans.\n    '
    return Model('extract_spans', forward, layers=[], refs={}, attrs={}, dims={}, init=init)

def init(model, X=None, Y=None):
    if False:
        while True:
            i = 10
    pass

def forward(model: Model, source_spans: Tuple[Ragged, Ragged], is_train: bool) -> Tuple[Ragged, Callable]:
    if False:
        print('Hello World!')
    'Get subsequences from source vectors.'
    ops = model.ops
    (X, spans) = source_spans
    assert spans.dataXd.ndim == 2
    indices = _get_span_indices(ops, spans, X.lengths)
    if len(indices) > 0:
        Y = Ragged(X.dataXd[indices], spans.dataXd[:, 1] - spans.dataXd[:, 0])
    else:
        Y = Ragged(ops.xp.zeros(X.dataXd.shape, dtype=X.dataXd.dtype), ops.xp.zeros((len(X.lengths),), dtype='i'))
    x_shape = X.dataXd.shape
    x_lengths = X.lengths

    def backprop_windows(dY: Ragged) -> Tuple[Ragged, Ragged]:
        if False:
            while True:
                i = 10
        dX = Ragged(ops.alloc2f(*x_shape), x_lengths)
        ops.scatter_add(dX.dataXd, indices, dY.dataXd)
        return (dX, spans)
    return (Y, backprop_windows)

def _get_span_indices(ops, spans: Ragged, lengths: Ints1d) -> Ints1d:
    if False:
        while True:
            i = 10
    'Construct a flat array that has the indices we want to extract from the\n    source data. For instance, if we want the spans (5, 9), (8, 10) the\n    indices will be [5, 6, 7, 8, 8, 9].\n    '
    (spans, lengths) = _ensure_cpu(spans, lengths)
    indices: List[int] = []
    offset = 0
    for (i, length) in enumerate(lengths):
        spans_i = spans[i].dataXd + offset
        for j in range(spans_i.shape[0]):
            indices.extend(range(spans_i[j, 0], spans_i[j, 1]))
        offset += length
    return ops.asarray1i(indices)

def _ensure_cpu(spans: Ragged, lengths: Ints1d) -> Tuple[Ragged, Ints1d]:
    if False:
        return 10
    return (Ragged(to_numpy(spans.dataXd), to_numpy(spans.lengths)), to_numpy(lengths))