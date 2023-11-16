import re
import weakref
from abc import ABCMeta, abstractmethod
from contextlib import ExitStack
import torch
from torch.distributions import biject_to
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import pyro.poutine.runtime as runtime
from pyro.distributions.util import broadcast_shape, sum_rightmost
from pyro.infer.autoguide.guides import prototype_hide_fn
from pyro.infer.autoguide.initialization import InitMessenger
from pyro.nn.module import PyroModule, PyroParam

class _EasyGuideMeta(type(PyroModule), ABCMeta):
    pass

class EasyGuide(PyroModule, metaclass=_EasyGuideMeta):
    """
    Base class for "easy guides", which are more flexible than
    :class:`~pyro.infer.AutoGuide` s, but are easier to write than raw Pyro guides.

    Derived classes should define a :meth:`guide` method. This :meth:`guide`
    method can combine ordinary guide statements (e.g. ``pyro.sample`` and
    ``pyro.param``) with the following special statements:

    - ``group = self.group(...)`` selects multiple ``pyro.sample`` sites in the
      model. See :class:`Group` for subsequent methods.
    - ``with self.plate(...): ...`` should be used instead of ``pyro.plate``.
    - ``self.map_estimate(...)`` uses a ``Delta`` guide for a single site.

    Derived classes may also override the :meth:`init` method to provide custom
    initialization for models sites.

    :param callable model: A Pyro model.
    """

    def __init__(self, model):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._pyro_name = type(self).__name__
        self._model = (model,)
        self.prototype_trace = None
        self.frames = {}
        self.groups = {}
        self.plates = {}

    @property
    def model(self):
        if False:
            for i in range(10):
                print('nop')
        return self._model[0]

    def _setup_prototype(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        model = poutine.block(InitMessenger(self.init)(self.model), prototype_hide_fn)
        self.prototype_trace = poutine.block(poutine.trace(model).get_trace)(*args, **kwargs)
        for (name, site) in self.prototype_trace.iter_stochastic_nodes():
            for frame in site['cond_indep_stack']:
                if not frame.vectorized:
                    raise NotImplementedError('EasyGuide does not support sequential pyro.plate')
                self.frames[frame.name] = frame

    @abstractmethod
    def guide(self, *args, **kargs):
        if False:
            return 10
        '\n        Guide implementation, to be overridden by user.\n        '
        raise NotImplementedError

    def init(self, site):
        if False:
            print('Hello World!')
        '\n        Model initialization method, may be overridden by user.\n\n        This should input a site and output a valid sample from that site.\n        The default behavior is to draw a random sample::\n\n            return site["fn"]()\n\n        For other possible initialization functions see\n        http://docs.pyro.ai/en/stable/infer.autoguide.html#module-pyro.infer.autoguide.initialization\n        '
        return site['fn']()

    def forward(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Runs the guide. This is typically used by inference algorithms.\n\n        .. note:: This method is used internally by :class:`~torch.nn.Module`.\n            Users should instead use :meth:`~torch.nn.Module.__call__`.\n        '
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)
        result = self.guide(*args, **kwargs)
        self.plates.clear()
        return result

    def plate(self, name, size=None, subsample_size=None, subsample=None, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        A wrapper around :class:`pyro.plate` to allow `EasyGuide` to\n        automatically construct plates. You should use this rather than\n        :class:`pyro.plate` inside your :meth:`guide` implementation.\n        '
        if name not in self.plates:
            self.plates[name] = pyro.plate(name, size, subsample_size, subsample, *args, **kwargs)
        return self.plates[name]

    def group(self, match='.*'):
        if False:
            i = 10
            return i + 15
        '\n        Select a :class:`Group` of model sites for joint guidance.\n\n        :param str match: A regex string matching names of model sample sites.\n        :return: A group of model sites.\n        :rtype: Group\n        '
        if match not in self.groups:
            sites = [site for (name, site) in self.prototype_trace.iter_stochastic_nodes() if re.match(match, name)]
            if not sites:
                raise ValueError('EasyGuide.group() pattern {} matched no model sites'.format(repr(match)))
            self.groups[match] = Group(self, sites)
        return self.groups[match]

    def map_estimate(self, name):
        if False:
            while True:
                i = 10
        '\n        Construct a maximum a posteriori (MAP) guide using Delta distributions.\n\n        :param str name: The name of a model sample site.\n        :return: A sampled value.\n        :rtype: torch.Tensor\n        '
        site = self.prototype_trace.nodes[name]
        fn = site['fn']
        event_dim = fn.event_dim
        init_needed = not hasattr(self, name)
        if init_needed:
            init_value = site['value'].detach()
        with ExitStack() as stack:
            for frame in site['cond_indep_stack']:
                plate = self.plate(frame.name)
                if plate not in runtime._PYRO_STACK:
                    stack.enter_context(plate)
                elif init_needed and plate.subsample_size < plate.size:
                    dim = plate.dim - event_dim
                    assert init_value.size(dim) == plate.subsample_size
                    ind = torch.arange(plate.size, device=init_value.device)
                    ind = ind % plate.subsample_size
                    init_value = init_value.index_select(dim, ind)
            if init_needed:
                setattr(self, name, PyroParam(init_value, fn.support, event_dim))
            value = getattr(self, name)
            return pyro.sample(name, dist.Delta(value, event_dim=event_dim))

class Group:
    """
    An autoguide helper to match a group of model sites.

    :ivar torch.Size event_shape: The total flattened concatenated shape of all
        matching sample sites in the model.
    :ivar list prototype_sites: A list of all matching sample sites in a
        prototype trace of the model.
    :param EasyGuide guide: An easyguide instance.
    :param list sites: A list of model sites.
    """

    def __init__(self, guide, sites):
        if False:
            print('Hello World!')
        assert isinstance(sites, list)
        assert sites
        self._guide = weakref.ref(guide)
        self.prototype_sites = sites
        self._site_sizes = {}
        self._site_batch_shapes = {}
        self.common_frames = frozenset.intersection(*(frozenset((f for f in site['cond_indep_stack'] if f.vectorized)) for site in sites))
        rightmost_common_dim = -float('inf')
        if self.common_frames:
            rightmost_common_dim = max((f.dim for f in self.common_frames))
        for site in sites:
            site_event_numel = torch.Size(site['fn'].event_shape).numel()
            site_batch_shape = list(site['fn'].batch_shape)
            for f in self.common_frames:
                site_batch_shape[f.dim] = 1
            while site_batch_shape and site_batch_shape[0] == 1:
                site_batch_shape = site_batch_shape[1:]
            if len(site_batch_shape) > -rightmost_common_dim:
                raise ValueError('Group expects all per-site plates to be right of all common plates, but found a per-site plate {} on left at site {}'.format(-len(site_batch_shape), repr(site['name'])))
            site_batch_shape = torch.Size(site_batch_shape)
            self._site_batch_shapes[site['name']] = site_batch_shape
            self._site_sizes[site['name']] = site_batch_shape.numel() * site_event_numel
        self.event_shape = torch.Size([sum(self._site_sizes.values())])

    def __getstate__(self):
        if False:
            for i in range(10):
                print('nop')
        state = getattr(super(), '__getstate__', self.__dict__.copy)()
        state['_guide'] = state['_guide']()
        return state

    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        self.__dict__.update(state)
        self._guide = weakref.ref(self._guide)

    @property
    def guide(self):
        if False:
            print('Hello World!')
        return self._guide()

    def sample(self, guide_name, fn, infer=None):
        if False:
            print('Hello World!')
        '\n        Wrapper around ``pyro.sample()`` to create a single auxiliary sample\n        site and then unpack to multiple sample sites for model replay.\n\n        :param str guide_name: The name of the auxiliary guide site.\n        :param callable fn: A distribution with shape ``self.event_shape``.\n        :param dict infer: Optional inference configuration dict.\n        :returns: A pair ``(guide_z, model_zs)`` where ``guide_z`` is the\n            single concatenated blob and ``model_zs`` is a dict mapping\n            site name to constrained model sample.\n        :rtype: tuple\n        '
        if fn.event_shape != self.event_shape:
            raise ValueError('Invalid fn.event_shape for group: expected {}, actual {}'.format(tuple(self.event_shape), tuple(fn.event_shape)))
        if infer is None:
            infer = {}
        infer['is_auxiliary'] = True
        guide_z = pyro.sample(guide_name, fn, infer=infer)
        common_batch_shape = guide_z.shape[:-1]
        model_zs = {}
        pos = 0
        for site in self.prototype_sites:
            name = site['name']
            fn = site['fn']
            size = self._site_sizes[name]
            batch_shape = broadcast_shape(common_batch_shape, self._site_batch_shapes[name])
            unconstrained_z = guide_z[..., pos:pos + size]
            unconstrained_z = unconstrained_z.reshape(batch_shape + fn.event_shape)
            pos += size
            transform = biject_to(fn.support)
            z = transform(unconstrained_z)
            log_density = transform.inv.log_abs_det_jacobian(z, unconstrained_z)
            log_density = sum_rightmost(log_density, log_density.dim() - z.dim() + fn.event_dim)
            delta_dist = dist.Delta(z, log_density=log_density, event_dim=fn.event_dim)
            with ExitStack() as stack:
                for frame in site['cond_indep_stack']:
                    plate = self.guide.plate(frame.name)
                    if plate not in runtime._PYRO_STACK:
                        stack.enter_context(plate)
                model_zs[name] = pyro.sample(name, delta_dist)
        return (guide_z, model_zs)

    def map_estimate(self):
        if False:
            print('Hello World!')
        '\n        Construct a maximum a posteriori (MAP) guide using Delta distributions.\n\n        :return: A dict mapping model site name to sampled value.\n        :rtype: dict\n        '
        return {site['name']: self.guide.map_estimate(site['name']) for site in self.prototype_sites}

def easy_guide(model):
    if False:
        while True:
            i = 10
    '\n    Convenience decorator to create an :class:`EasyGuide` .\n    The following are equivalent::\n\n        # Version 1. Decorate a function.\n        @easy_guide(model)\n        def guide(self, foo, bar):\n            return my_guide(foo, bar)\n\n        # Version 2. Create and instantiate a subclass of EasyGuide.\n        class Guide(EasyGuide):\n            def guide(self, foo, bar):\n                return my_guide(foo, bar)\n        guide = Guide(model)\n\n    Note ``@easy_guide`` wrappers cannot be pickled; to build a guide that can\n    be pickled, instead subclass from :class:`EasyGuide`.\n\n    :param callable model: a Pyro model.\n    '

    def decorator(fn):
        if False:
            while True:
                i = 10
        Guide = type(fn.__name__, (EasyGuide,), {'guide': fn})
        return Guide(model)
    return decorator