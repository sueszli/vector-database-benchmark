from functools import partial, singledispatch
from typing import Tuple
import torch
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

class ProvenanceTensor(torch.Tensor):
    """
    Provenance tracking implementation in Pytorch.

    This class wraps a :class:`torch.Tensor` to track provenance through
    PyTorch ops, where provenance is a user-defined frozenset of objects. The
    provenance of the output tensors of any op is the union of provenances of
    input tensors.

    -   To start tracking provenance, wrap a :class:`torch.Tensor` in a
        :class:`ProvenanceTensor` with user-defined initial provenance.
    -   To read the provenance of a tensor use :meth:`get_provenance` .
    -   To detach provenance during a computation (similar to
        :meth:`~torch.Tensor.detach` to detach gradients during Pytorch
        computations), use the :meth:`detach_provenance` . This is useful to
        distinguish direct vs indirect provenance.

    Example::

        >>> a = ProvenanceTensor(torch.randn(3), frozenset({"a"}))
        >>> b = ProvenanceTensor(torch.randn(3), frozenset({"b"}))
        >>> c = torch.randn(3)
        >>> assert get_provenance(a + b + c) == frozenset({"a", "b"})
        >>> assert get_provenance(a + detach_provenance(b) + c) == frozenset({"a"})

    **References**

    [1] David Wingate, Noah Goodman, Andreas Stuhlmüller, Jeffrey Siskind (2011)
        Nonstandard Interpretations of Probabilistic Programs for Efficient Inference
        http://papers.neurips.cc/paper/4309-nonstandard-interpretations-of-probabilistic-programs-for-efficient-inference.pdf

    :param torch.Tensor data: An initial tensor to start tracking.
    :param frozenset provenance: An initial provenance set.
    """
    _t: torch.Tensor
    _provenance: frozenset

    def __new__(cls, data: torch.Tensor, provenance=frozenset(), **kwargs):
        if False:
            i = 10
            return i + 15
        assert not isinstance(data, ProvenanceTensor)
        if not provenance:
            return data
        ret = data.as_subclass(cls)
        ret._t = data
        ret._provenance = provenance
        return ret

    def __repr__(self):
        if False:
            return 10
        return 'Provenance:\n{}\nTensor:\n{}'.format(self._provenance, self._t)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if False:
            return 10
        (_args, _kwargs) = detach_provenance([args, kwargs or {}])
        ret = func(*_args, **_kwargs)
        return track_provenance(ret, get_provenance([args, kwargs]))

@singledispatch
def track_provenance(x, provenance: frozenset):
    if False:
        print('Hello World!')
    '\n    Adds provenance info to the :class:`torch.Tensor` leaves of a data structure.\n\n    :param x: an object to add provenence info to.\n    :param frozenset provenance: A provenence set.\n    :returns: A provenence-tracking version of ``x``.\n    '
    return x
track_provenance.register(torch.Tensor)(ProvenanceTensor)

@track_provenance.register(frozenset)
@track_provenance.register(set)
def _track_provenance_set(x, provenance: frozenset):
    if False:
        print('Hello World!')
    return type(x)((track_provenance(part, provenance) for part in x))

@track_provenance.register(list)
@track_provenance.register(tuple)
@track_provenance.register(dict)
def _track_provenance_pytree(x, provenance: frozenset):
    if False:
        print('Hello World!')
    return tree_map(partial(track_provenance, provenance=provenance), x)

@track_provenance.register
def _track_provenance_provenancetensor(x: ProvenanceTensor, provenance: frozenset):
    if False:
        print('Hello World!')
    (x_value, old_provenance) = extract_provenance(x)
    return track_provenance(x_value, old_provenance | provenance)

@singledispatch
def extract_provenance(x) -> Tuple[object, frozenset]:
    if False:
        print('Hello World!')
    '\n    Extracts the provenance of a data structure possibly containing\n    :class:`torch.Tensor` s as leaves, and separates into a detached object and\n    provenance.\n\n    :param x: An input data structure.\n    :returns: a tuple ``(detached_value, provenance)``\n    :rtype: tuple\n    '
    return (x, frozenset())

@extract_provenance.register(ProvenanceTensor)
def _extract_provenance_tensor(x):
    if False:
        while True:
            i = 10
    return (x._t, x._provenance)

@extract_provenance.register(frozenset)
@extract_provenance.register(set)
def _extract_provenance_set(x):
    if False:
        for i in range(10):
            print('nop')
    provenance = frozenset()
    values = []
    for part in x:
        (v, p) = extract_provenance(part)
        values.append(v)
        provenance |= p
    value = type(x)(values)
    return (value, provenance)

@extract_provenance.register(list)
@extract_provenance.register(tuple)
@extract_provenance.register(dict)
def _extract_provenance_pytree(x):
    if False:
        print('Hello World!')
    (flat_args, spec) = tree_flatten(x)
    xs = []
    provenance = frozenset()
    for (x, p) in map(extract_provenance, flat_args):
        xs.append(x)
        provenance |= p
    return (tree_unflatten(xs, spec), provenance)

def get_provenance(x) -> frozenset:
    if False:
        i = 10
        return i + 15
    '\n    Reads the provenance of a recursive datastructure possibly containing\n    :class:`torch.Tensor` s.\n\n    :param torch.Tensor tensor: An input tensor.\n    :returns: A provenance frozenset.\n    :rtype: frozenset\n    '
    (_, provenance) = extract_provenance(x)
    return provenance

def detach_provenance(x):
    if False:
        for i in range(10):
            print('nop')
    '\n    Blocks provenance tracking through a tensor, similar to :meth:`torch.Tensor.detach`.\n\n    :param torch.Tensor tensor: An input tensor.\n    :returns: A tensor sharing the same data but with no provenance.\n    :rtype: torch.Tensor\n    '
    (value, _) = extract_provenance(x)
    return value