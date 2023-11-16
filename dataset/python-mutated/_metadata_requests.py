"""
Metadata Routing Utility

In order to better understand the components implemented in this file, one
needs to understand their relationship to one another.

The only relevant public API for end users are the ``set_{method}_request``,
e.g. ``estimator.set_fit_request(sample_weight=True)``. However, third-party
developers and users who implement custom meta-estimators, need to deal with
the objects implemented in this file.

All estimators (should) implement a ``get_metadata_routing`` method, returning
the routing requests set for the estimator. This method is automatically
implemented via ``BaseEstimator`` for all simple estimators, but needs a custom
implementation for meta-estimators.

In non-routing consumers, i.e. the simplest case, e.g. ``SVM``,
``get_metadata_routing`` returns a ``MetadataRequest`` object.

In routers, e.g. meta-estimators and a multi metric scorer,
``get_metadata_routing`` returns a ``MetadataRouter`` object.

An object which is both a router and a consumer, e.g. a meta-estimator which
consumes ``sample_weight`` and routes ``sample_weight`` to its sub-estimators,
routing information includes both information about the object itself (added
via ``MetadataRouter.add_self_request``), as well as the routing information
for its sub-estimators.

A ``MetadataRequest`` instance includes one ``MethodMetadataRequest`` per
method in ``METHODS``, which includes ``fit``, ``score``, etc.

Request values are added to the routing mechanism by adding them to
``MethodMetadataRequest`` instances, e.g.
``metadatarequest.fit.add(param="sample_weight", alias="my_weights")``. This is
used in ``set_{method}_request`` which are automatically generated, so users
and developers almost never need to directly call methods on a
``MethodMetadataRequest``.

The ``alias`` above in the ``add`` method has to be either a string (an alias),
or a {True (requested), False (unrequested), None (error if passed)}``. There
are some other special values such as ``UNUSED`` and ``WARN`` which are used
for purposes such as warning of removing a metadata in a child class, but not
used by the end users.

``MetadataRouter`` includes information about sub-objects' routing and how
methods are mapped together. For instance, the information about which methods
of a sub-estimator are called in which methods of the meta-estimator are all
stored here. Conceptually, this information looks like:

```
{
    "sub_estimator1": (
        mapping=[(caller="fit", callee="transform"), ...],
        router=MetadataRequest(...),  # or another MetadataRouter
    ),
    ...
}
```

To give the above representation some structure, we use the following objects:

- ``(caller, callee)`` is a namedtuple called ``MethodPair``

- The list of ``MethodPair`` stored in the ``mapping`` field is a
  ``MethodMapping`` object

- ``(mapping=..., router=...)`` is a namedtuple called ``RouterMappingPair``

The ``set_{method}_request`` methods are dynamically generated for estimators
which inherit from the ``BaseEstimator``. This is done by attaching instances
of the ``RequestMethod`` descriptor to classes, which is done in the
``_MetadataRequester`` class, and ``BaseEstimator`` inherits from this mixin.
This mixin also implements the ``get_metadata_routing``, which meta-estimators
need to override, but it works for simple consumers as is.
"""
import inspect
from collections import namedtuple
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union
from warnings import warn
from .. import get_config
from ..exceptions import UnsetMetadataPassedError
from ._bunch import Bunch
SIMPLE_METHODS = ['fit', 'partial_fit', 'predict', 'predict_proba', 'predict_log_proba', 'decision_function', 'score', 'split', 'transform', 'inverse_transform']
COMPOSITE_METHODS = {'fit_transform': ['fit', 'transform'], 'fit_predict': ['fit', 'predict']}
METHODS = SIMPLE_METHODS + list(COMPOSITE_METHODS.keys())

def _routing_enabled():
    if False:
        for i in range(10):
            print('nop')
    'Return whether metadata routing is enabled.\n\n    .. versionadded:: 1.3\n\n    Returns\n    -------\n    enabled : bool\n        Whether metadata routing is enabled. If the config is not set, it\n        defaults to False.\n    '
    return get_config().get('enable_metadata_routing', False)

def _raise_for_params(params, owner, method):
    if False:
        while True:
            i = 10
    'Raise an error if metadata routing is not enabled and params are passed.\n\n    .. versionadded:: 1.4\n\n    Parameters\n    ----------\n    params : dict\n        The metadata passed to a method.\n\n    owner : object\n        The object to which the method belongs.\n\n    method : str\n        The name of the method, e.g. "fit".\n\n    Raises\n    ------\n    ValueError\n        If metadata routing is not enabled and params are passed.\n    '
    caller = f'{owner.__class__.__name__}.{method}' if method else owner.__class__.__name__
    if not _routing_enabled() and params:
        raise ValueError(f'Passing extra keyword arguments to {caller} is only supported if enable_metadata_routing=True, which you can set using `sklearn.set_config`. See the User Guide <https://scikit-learn.org/stable/metadata_routing.html> for more details. Extra parameters passed are: {set(params)}')

def _raise_for_unsupported_routing(obj, method, **kwargs):
    if False:
        print('Hello World!')
    "Raise when metadata routing is enabled and metadata is passed.\n\n    This is used in meta-estimators which have not implemented metadata routing\n    to prevent silent bugs. There is no need to use this function if the\n    meta-estimator is not accepting any metadata, especially in `fit`, since\n    if a meta-estimator accepts any metadata, they would do that in `fit` as\n    well.\n\n    Parameters\n    ----------\n    obj : estimator\n        The estimator for which we're raising the error.\n\n    method : str\n        The method where the error is raised.\n\n    **kwargs : dict\n        The metadata passed to the method.\n    "
    kwargs = {key: value for (key, value) in kwargs.items() if value is not None}
    if _routing_enabled() and kwargs:
        cls_name = obj.__class__.__name__
        raise NotImplementedError(f'{cls_name}.{method} cannot accept given metadata ({set(kwargs.keys())}) since metadata routing is not yet implemented for {cls_name}.')

class _RoutingNotSupportedMixin:
    """A mixin to be used to remove the default `get_metadata_routing`.

    This is used in meta-estimators where metadata routing is not yet
    implemented.

    This also makes it clear in our rendered documentation that this method
    cannot be used.
    """

    def get_metadata_routing(self):
        if False:
            return 10
        'Raise `NotImplementedError`.\n\n        This estimator does not support metadata routing yet.'
        raise NotImplementedError(f'{self.__class__.__name__} has not implemented metadata routing yet.')
UNUSED = '$UNUSED$'
WARN = '$WARN$'
UNCHANGED = '$UNCHANGED$'
VALID_REQUEST_VALUES = [False, True, None, UNUSED, WARN]

def request_is_alias(item):
    if False:
        i = 10
        return i + 15
    'Check if an item is a valid alias.\n\n    Values in ``VALID_REQUEST_VALUES`` are not considered aliases in this\n    context. Only a string which is a valid identifier is.\n\n    Parameters\n    ----------\n    item : object\n        The given item to be checked if it can be an alias.\n\n    Returns\n    -------\n    result : bool\n        Whether the given item is a valid alias.\n    '
    if item in VALID_REQUEST_VALUES:
        return False
    return isinstance(item, str) and item.isidentifier()

def request_is_valid(item):
    if False:
        return 10
    'Check if an item is a valid request value (and not an alias).\n\n    Parameters\n    ----------\n    item : object\n        The given item to be checked.\n\n    Returns\n    -------\n    result : bool\n        Whether the given item is valid.\n    '
    return item in VALID_REQUEST_VALUES

class MethodMetadataRequest:
    """A prescription of how metadata is to be passed to a single method.

    Refer to :class:`MetadataRequest` for how this class is used.

    .. versionadded:: 1.3

    Parameters
    ----------
    owner : str
        A display name for the object owning these requests.

    method : str
        The name of the method to which these requests belong.

    requests : dict of {str: bool, None or str}, default=None
        The initial requests for this method.
    """

    def __init__(self, owner, method, requests=None):
        if False:
            return 10
        self._requests = requests or dict()
        self.owner = owner
        self.method = method

    @property
    def requests(self):
        if False:
            while True:
                i = 10
        'Dictionary of the form: ``{key: alias}``.'
        return self._requests

    def add_request(self, *, param, alias):
        if False:
            i = 10
            return i + 15
        'Add request info for a metadata.\n\n        Parameters\n        ----------\n        param : str\n            The property for which a request is set.\n\n        alias : str, or {True, False, None}\n            Specifies which metadata should be routed to `param`\n\n            - str: the name (or alias) of metadata given to a meta-estimator that\n              should be routed to this parameter.\n\n            - True: requested\n\n            - False: not requested\n\n            - None: error if passed\n        '
        if not request_is_alias(alias) and (not request_is_valid(alias)):
            raise ValueError(f"The alias you're setting for `{param}` should be either a valid identifier or one of {{None, True, False}}, but given value is: `{alias}`")
        if alias == param:
            alias = True
        if alias == UNUSED:
            if param in self._requests:
                del self._requests[param]
            else:
                raise ValueError(f"Trying to remove parameter {param} with UNUSED which doesn't exist.")
        else:
            self._requests[param] = alias
        return self

    def _get_param_names(self, return_alias):
        if False:
            i = 10
            return i + 15
        'Get names of all metadata that can be consumed or routed by this method.\n\n        This method returns the names of all metadata, even the ``False``\n        ones.\n\n        Parameters\n        ----------\n        return_alias : bool\n            Controls whether original or aliased names should be returned. If\n            ``False``, aliases are ignored and original names are returned.\n\n        Returns\n        -------\n        names : set of str\n            A set of strings with the names of all parameters.\n        '
        return set((alias if return_alias and (not request_is_valid(alias)) else prop for (prop, alias) in self._requests.items() if not request_is_valid(alias) or alias is not False))

    def _check_warnings(self, *, params):
        if False:
            return 10
        'Check whether metadata is passed which is marked as WARN.\n\n        If any metadata is passed which is marked as WARN, a warning is raised.\n\n        Parameters\n        ----------\n        params : dict\n            The metadata passed to a method.\n        '
        params = {} if params is None else params
        warn_params = {prop for (prop, alias) in self._requests.items() if alias == WARN and prop in params}
        for param in warn_params:
            warn(f'Support for {param} has recently been added to this class. To maintain backward compatibility, it is ignored now. You can set the request value to False to silence this warning, or to True to consume and use the metadata.')

    def _route_params(self, params):
        if False:
            print('Hello World!')
        'Prepare the given parameters to be passed to the method.\n\n        The output of this method can be used directly as the input to the\n        corresponding method as extra props.\n\n        Parameters\n        ----------\n        params : dict\n            A dictionary of provided metadata.\n\n        Returns\n        -------\n        params : Bunch\n            A :class:`~sklearn.utils.Bunch` of {prop: value} which can be given to the\n            corresponding method.\n        '
        self._check_warnings(params=params)
        unrequested = dict()
        args = {arg: value for (arg, value) in params.items() if value is not None}
        res = Bunch()
        for (prop, alias) in self._requests.items():
            if alias is False or alias == WARN:
                continue
            elif alias is True and prop in args:
                res[prop] = args[prop]
            elif alias is None and prop in args:
                unrequested[prop] = args[prop]
            elif alias in args:
                res[prop] = args[alias]
        if unrequested:
            raise UnsetMetadataPassedError(message=f"[{', '.join([key for key in unrequested])}] are passed but are not explicitly set as requested or not for {self.owner}.{self.method}", unrequested_params=unrequested, routed_params=res)
        return res

    def _consumes(self, params):
        if False:
            return 10
        'Check whether the given parameters are consumed by this method.\n\n        Parameters\n        ----------\n        params : iterable of str\n            An iterable of parameters to check.\n\n        Returns\n        -------\n        consumed : set of str\n            A set of parameters which are consumed by this method.\n        '
        params = set(params)
        res = set()
        for (prop, alias) in self._requests.items():
            if alias is True and prop in params:
                res.add(prop)
            elif isinstance(alias, str) and alias in params:
                res.add(alias)
        return res

    def _serialize(self):
        if False:
            for i in range(10):
                print('nop')
        'Serialize the object.\n\n        Returns\n        -------\n        obj : dict\n            A serialized version of the instance in the form of a dictionary.\n        '
        return self._requests

    def __repr__(self):
        if False:
            return 10
        return str(self._serialize())

    def __str__(self):
        if False:
            print('Hello World!')
        return str(repr(self))

class MetadataRequest:
    """Contains the metadata request info of a consumer.

    Instances of `MethodMetadataRequest` are used in this class for each
    available method under `metadatarequest.{method}`.

    Consumer-only classes such as simple estimators return a serialized
    version of this class as the output of `get_metadata_routing()`.

    .. versionadded:: 1.3

    Parameters
    ----------
    owner : str
        The name of the object to which these requests belong.
    """
    _type = 'metadata_request'

    def __init__(self, owner):
        if False:
            for i in range(10):
                print('nop')
        self.owner = owner
        for method in SIMPLE_METHODS:
            setattr(self, method, MethodMetadataRequest(owner=owner, method=method))

    def consumes(self, method, params):
        if False:
            i = 10
            return i + 15
        'Check whether the given parameters are consumed by the given method.\n\n        .. versionadded:: 1.4\n\n        Parameters\n        ----------\n        method : str\n            The name of the method to check.\n\n        params : iterable of str\n            An iterable of parameters to check.\n\n        Returns\n        -------\n        consumed : set of str\n            A set of parameters which are consumed by the given method.\n        '
        return getattr(self, method)._consumes(params=params)

    def __getattr__(self, name):
        if False:
            for i in range(10):
                print('nop')
        if name not in COMPOSITE_METHODS:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        requests = {}
        for method in COMPOSITE_METHODS[name]:
            mmr = getattr(self, method)
            existing = set(requests.keys())
            upcoming = set(mmr.requests.keys())
            common = existing & upcoming
            conflicts = [key for key in common if requests[key] != mmr._requests[key]]
            if conflicts:
                raise ValueError(f"Conflicting metadata requests for {', '.join(conflicts)} while composing the requests for {name}. Metadata with the same name for methods {', '.join(COMPOSITE_METHODS[name])} should have the same request value.")
            requests.update(mmr._requests)
        return MethodMetadataRequest(owner=self.owner, method=name, requests=requests)

    def _get_param_names(self, method, return_alias, ignore_self_request=None):
        if False:
            for i in range(10):
                print('nop')
        'Get names of all metadata that can be consumed or routed by specified             method.\n\n        This method returns the names of all metadata, even the ``False``\n        ones.\n\n        Parameters\n        ----------\n        method : str\n            The name of the method for which metadata names are requested.\n\n        return_alias : bool\n            Controls whether original or aliased names should be returned. If\n            ``False``, aliases are ignored and original names are returned.\n\n        ignore_self_request : bool\n            Ignored. Present for API compatibility.\n\n        Returns\n        -------\n        names : set of str\n            A set of strings with the names of all parameters.\n        '
        return getattr(self, method)._get_param_names(return_alias=return_alias)

    def _route_params(self, *, method, params):
        if False:
            i = 10
            return i + 15
        'Prepare the given parameters to be passed to the method.\n\n        The output of this method can be used directly as the input to the\n        corresponding method as extra keyword arguments to pass metadata.\n\n        Parameters\n        ----------\n        method : str\n            The name of the method for which the parameters are requested and\n            routed.\n\n        params : dict\n            A dictionary of provided metadata.\n\n        Returns\n        -------\n        params : Bunch\n            A :class:`~sklearn.utils.Bunch` of {prop: value} which can be given to the\n            corresponding method.\n        '
        return getattr(self, method)._route_params(params=params)

    def _check_warnings(self, *, method, params):
        if False:
            i = 10
            return i + 15
        'Check whether metadata is passed which is marked as WARN.\n\n        If any metadata is passed which is marked as WARN, a warning is raised.\n\n        Parameters\n        ----------\n        method : str\n            The name of the method for which the warnings should be checked.\n\n        params : dict\n            The metadata passed to a method.\n        '
        getattr(self, method)._check_warnings(params=params)

    def _serialize(self):
        if False:
            i = 10
            return i + 15
        'Serialize the object.\n\n        Returns\n        -------\n        obj : dict\n            A serialized version of the instance in the form of a dictionary.\n        '
        output = dict()
        for method in SIMPLE_METHODS:
            mmr = getattr(self, method)
            if len(mmr.requests):
                output[method] = mmr._serialize()
        return output

    def __repr__(self):
        if False:
            while True:
                i = 10
        return str(self._serialize())

    def __str__(self):
        if False:
            print('Hello World!')
        return str(repr(self))
RouterMappingPair = namedtuple('RouterMappingPair', ['mapping', 'router'])
MethodPair = namedtuple('MethodPair', ['callee', 'caller'])

class MethodMapping:
    """Stores the mapping between callee and caller methods for a router.

    This class is primarily used in a ``get_metadata_routing()`` of a router
    object when defining the mapping between a sub-object (a sub-estimator or a
    scorer) to the router's methods. It stores a collection of ``Route``
    namedtuples.

    Iterating through an instance of this class will yield named
    ``MethodPair(callee, caller)`` tuples.

    .. versionadded:: 1.3
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._routes = []

    def __iter__(self):
        if False:
            print('Hello World!')
        return iter(self._routes)

    def add(self, *, callee, caller):
        if False:
            while True:
                i = 10
        "Add a method mapping.\n\n        Parameters\n        ----------\n        callee : str\n            Child object's method name. This method is called in ``caller``.\n\n        caller : str\n            Parent estimator's method name in which the ``callee`` is called.\n\n        Returns\n        -------\n        self : MethodMapping\n            Returns self.\n        "
        if callee not in METHODS:
            raise ValueError(f'Given callee:{callee} is not a valid method. Valid methods are: {METHODS}')
        if caller not in METHODS:
            raise ValueError(f'Given caller:{caller} is not a valid method. Valid methods are: {METHODS}')
        self._routes.append(MethodPair(callee=callee, caller=caller))
        return self

    def _serialize(self):
        if False:
            print('Hello World!')
        'Serialize the object.\n\n        Returns\n        -------\n        obj : list\n            A serialized version of the instance in the form of a list.\n        '
        result = list()
        for route in self._routes:
            result.append({'callee': route.callee, 'caller': route.caller})
        return result

    @classmethod
    def from_str(cls, route):
        if False:
            print('Hello World!')
        'Construct an instance from a string.\n\n        Parameters\n        ----------\n        route : str\n            A string representing the mapping, it can be:\n\n              - `"one-to-one"`: a one to one mapping for all methods.\n              - `"method"`: the name of a single method, such as ``fit``,\n                ``transform``, ``score``, etc.\n\n        Returns\n        -------\n        obj : MethodMapping\n            A :class:`~sklearn.utils.metadata_routing.MethodMapping` instance\n            constructed from the given string.\n        '
        routing = cls()
        if route == 'one-to-one':
            for method in METHODS:
                routing.add(callee=method, caller=method)
        elif route in METHODS:
            routing.add(callee=route, caller=route)
        else:
            raise ValueError("route should be 'one-to-one' or a single method!")
        return routing

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return str(self._serialize())

    def __str__(self):
        if False:
            print('Hello World!')
        return str(repr(self))

class MetadataRouter:
    """Stores and handles metadata routing for a router object.

    This class is used by router objects to store and handle metadata routing.
    Routing information is stored as a dictionary of the form ``{"object_name":
    RouteMappingPair(method_mapping, routing_info)}``, where ``method_mapping``
    is an instance of :class:`~sklearn.utils.metadata_routing.MethodMapping` and
    ``routing_info`` is either a
    :class:`~sklearn.utils.metadata_routing.MetadataRequest` or a
    :class:`~sklearn.utils.metadata_routing.MetadataRouter` instance.

    .. versionadded:: 1.3

    Parameters
    ----------
    owner : str
        The name of the object to which these requests belong.
    """
    _type = 'metadata_router'

    def __init__(self, owner):
        if False:
            i = 10
            return i + 15
        self._route_mappings = dict()
        self._self_request = None
        self.owner = owner

    def add_self_request(self, obj):
        if False:
            for i in range(10):
                print('nop')
        'Add `self` (as a consumer) to the routing.\n\n        This method is used if the router is also a consumer, and hence the\n        router itself needs to be included in the routing. The passed object\n        can be an estimator or a\n        :class:`~sklearn.utils.metadata_routing.MetadataRequest`.\n\n        A router should add itself using this method instead of `add` since it\n        should be treated differently than the other objects to which metadata\n        is routed by the router.\n\n        Parameters\n        ----------\n        obj : object\n            This is typically the router instance, i.e. `self` in a\n            ``get_metadata_routing()`` implementation. It can also be a\n            ``MetadataRequest`` instance.\n\n        Returns\n        -------\n        self : MetadataRouter\n            Returns `self`.\n        '
        if getattr(obj, '_type', None) == 'metadata_request':
            self._self_request = deepcopy(obj)
        elif hasattr(obj, '_get_metadata_request'):
            self._self_request = deepcopy(obj._get_metadata_request())
        else:
            raise ValueError('Given `obj` is neither a `MetadataRequest` nor does it implement the required API. Inheriting from `BaseEstimator` implements the required API.')
        return self

    def add(self, *, method_mapping, **objs):
        if False:
            i = 10
            return i + 15
        "Add named objects with their corresponding method mapping.\n\n        Parameters\n        ----------\n        method_mapping : MethodMapping or str\n            The mapping between the child and the parent's methods. If str, the\n            output of :func:`~sklearn.utils.metadata_routing.MethodMapping.from_str`\n            is used.\n\n        **objs : dict\n            A dictionary of objects from which metadata is extracted by calling\n            :func:`~sklearn.utils.metadata_routing.get_routing_for_object` on them.\n\n        Returns\n        -------\n        self : MetadataRouter\n            Returns `self`.\n        "
        if isinstance(method_mapping, str):
            method_mapping = MethodMapping.from_str(method_mapping)
        else:
            method_mapping = deepcopy(method_mapping)
        for (name, obj) in objs.items():
            self._route_mappings[name] = RouterMappingPair(mapping=method_mapping, router=get_routing_for_object(obj))
        return self

    def consumes(self, method, params):
        if False:
            while True:
                i = 10
        'Check whether the given parameters are consumed by the given method.\n\n        .. versionadded:: 1.4\n\n        Parameters\n        ----------\n        method : str\n            The name of the method to check.\n\n        params : iterable of str\n            An iterable of parameters to check.\n\n        Returns\n        -------\n        consumed : set of str\n            A set of parameters which are consumed by the given method.\n        '
        res = set()
        if self._self_request:
            res = res | self._self_request.consumes(method=method, params=params)
        for (_, route_mapping) in self._route_mappings.items():
            for (callee, caller) in route_mapping.mapping:
                if caller == method:
                    res = res | route_mapping.router.consumes(method=callee, params=params)
        return res

    def _get_param_names(self, *, method, return_alias, ignore_self_request):
        if False:
            for i in range(10):
                print('nop')
        'Get names of all metadata that can be consumed or routed by specified             method.\n\n        This method returns the names of all metadata, even the ``False``\n        ones.\n\n        Parameters\n        ----------\n        method : str\n            The name of the method for which metadata names are requested.\n\n        return_alias : bool\n            Controls whether original or aliased names should be returned,\n            which only applies to the stored `self`. If no `self` routing\n            object is stored, this parameter has no effect.\n\n        ignore_self_request : bool\n            If `self._self_request` should be ignored. This is used in `_route_params`.\n            If ``True``, ``return_alias`` has no effect.\n\n        Returns\n        -------\n        names : set of str\n            A set of strings with the names of all parameters.\n        '
        res = set()
        if self._self_request and (not ignore_self_request):
            res = res.union(self._self_request._get_param_names(method=method, return_alias=return_alias))
        for (name, route_mapping) in self._route_mappings.items():
            for (callee, caller) in route_mapping.mapping:
                if caller == method:
                    res = res.union(route_mapping.router._get_param_names(method=callee, return_alias=True, ignore_self_request=False))
        return res

    def _route_params(self, *, params, method):
        if False:
            for i in range(10):
                print('nop')
        'Prepare the given parameters to be passed to the method.\n\n        This is used when a router is used as a child object of another router.\n        The parent router then passes all parameters understood by the child\n        object to it and delegates their validation to the child.\n\n        The output of this method can be used directly as the input to the\n        corresponding method as extra props.\n\n        Parameters\n        ----------\n        method : str\n            The name of the method for which the parameters are requested and\n            routed.\n\n        params : dict\n            A dictionary of provided metadata.\n\n        Returns\n        -------\n        params : Bunch\n            A :class:`~sklearn.utils.Bunch` of {prop: value} which can be given to the\n            corresponding method.\n        '
        res = Bunch()
        if self._self_request:
            res.update(self._self_request._route_params(params=params, method=method))
        param_names = self._get_param_names(method=method, return_alias=True, ignore_self_request=True)
        child_params = {key: value for (key, value) in params.items() if key in param_names}
        for key in set(res.keys()).intersection(child_params.keys()):
            if child_params[key] is not res[key]:
                raise ValueError(f'In {self.owner}, there is a conflict on {key} between what is requested for this estimator and what is requested by its children. You can resolve this conflict by using an alias for the child estimator(s) requested metadata.')
        res.update(child_params)
        return res

    def route_params(self, *, caller, params):
        if False:
            print('Hello World!')
        'Return the input parameters requested by child objects.\n\n        The output of this method is a bunch, which includes the inputs for all\n        methods of each child object that are used in the router\'s `caller`\n        method.\n\n        If the router is also a consumer, it also checks for warnings of\n        `self`\'s/consumer\'s requested metadata.\n\n        Parameters\n        ----------\n        caller : str\n            The name of the method for which the parameters are requested and\n            routed. If called inside the :term:`fit` method of a router, it\n            would be `"fit"`.\n\n        params : dict\n            A dictionary of provided metadata.\n\n        Returns\n        -------\n        params : Bunch\n            A :class:`~sklearn.utils.Bunch` of the form\n            ``{"object_name": {"method_name": {prop: value}}}`` which can be\n            used to pass the required metadata to corresponding methods or\n            corresponding child objects.\n        '
        if self._self_request:
            self._self_request._check_warnings(params=params, method=caller)
        res = Bunch()
        for (name, route_mapping) in self._route_mappings.items():
            (router, mapping) = (route_mapping.router, route_mapping.mapping)
            res[name] = Bunch()
            for (_callee, _caller) in mapping:
                if _caller == caller:
                    res[name][_callee] = router._route_params(params=params, method=_callee)
        return res

    def validate_metadata(self, *, method, params):
        if False:
            while True:
                i = 10
        'Validate given metadata for a method.\n\n        This raises a ``TypeError`` if some of the passed metadata are not\n        understood by child objects.\n\n        Parameters\n        ----------\n        method : str\n            The name of the method for which the parameters are requested and\n            routed. If called inside the :term:`fit` method of a router, it\n            would be `"fit"`.\n\n        params : dict\n            A dictionary of provided metadata.\n        '
        param_names = self._get_param_names(method=method, return_alias=False, ignore_self_request=False)
        if self._self_request:
            self_params = self._self_request._get_param_names(method=method, return_alias=False)
        else:
            self_params = set()
        extra_keys = set(params.keys()) - param_names - self_params
        if extra_keys:
            raise TypeError(f'{self.owner}.{method} got unexpected argument(s) {extra_keys}, which are not requested metadata in any object.')

    def _serialize(self):
        if False:
            return 10
        'Serialize the object.\n\n        Returns\n        -------\n        obj : dict\n            A serialized version of the instance in the form of a dictionary.\n        '
        res = dict()
        if self._self_request:
            res['$self_request'] = self._self_request._serialize()
        for (name, route_mapping) in self._route_mappings.items():
            res[name] = dict()
            res[name]['mapping'] = route_mapping.mapping._serialize()
            res[name]['router'] = route_mapping.router._serialize()
        return res

    def __iter__(self):
        if False:
            while True:
                i = 10
        if self._self_request:
            yield ('$self_request', RouterMappingPair(mapping=MethodMapping.from_str('one-to-one'), router=self._self_request))
        for (name, route_mapping) in self._route_mappings.items():
            yield (name, route_mapping)

    def __repr__(self):
        if False:
            return 10
        return str(self._serialize())

    def __str__(self):
        if False:
            print('Hello World!')
        return str(repr(self))

def get_routing_for_object(obj=None):
    if False:
        i = 10
        return i + 15
    'Get a ``Metadata{Router, Request}`` instance from the given object.\n\n    This function returns a\n    :class:`~sklearn.utils.metadata_routing.MetadataRouter` or a\n    :class:`~sklearn.utils.metadata_routing.MetadataRequest` from the given input.\n\n    This function always returns a copy or an instance constructed from the\n    input, such that changing the output of this function will not change the\n    original object.\n\n    .. versionadded:: 1.3\n\n    Parameters\n    ----------\n    obj : object\n        - If the object is already a\n            :class:`~sklearn.utils.metadata_routing.MetadataRequest` or a\n            :class:`~sklearn.utils.metadata_routing.MetadataRouter`, return a copy\n            of that.\n        - If the object provides a `get_metadata_routing` method, return a copy\n            of the output of that method.\n        - Returns an empty :class:`~sklearn.utils.metadata_routing.MetadataRequest`\n            otherwise.\n\n    Returns\n    -------\n    obj : MetadataRequest or MetadataRouting\n        A ``MetadataRequest`` or a ``MetadataRouting`` taken or created from\n        the given object.\n    '
    if hasattr(obj, 'get_metadata_routing'):
        return deepcopy(obj.get_metadata_routing())
    elif getattr(obj, '_type', None) in ['metadata_request', 'metadata_router']:
        return deepcopy(obj)
    return MetadataRequest(owner=None)
REQUESTER_DOC = '        Request metadata passed to the ``{method}`` method.\n\n        Note that this method is only relevant if\n        ``enable_metadata_routing=True`` (see :func:`sklearn.set_config`).\n        Please see :ref:`User Guide <metadata_routing>` on how the routing\n        mechanism works.\n\n        The options for each parameter are:\n\n        - ``True``: metadata is requested, and passed to ``{method}`` if provided. The request is ignored if metadata is not provided.\n\n        - ``False``: metadata is not requested and the meta-estimator will not pass it to ``{method}``.\n\n        - ``None``: metadata is not requested, and the meta-estimator will raise an error if the user provides it.\n\n        - ``str``: metadata should be passed to the meta-estimator with this given alias instead of the original name.\n\n        The default (``sklearn.utils.metadata_routing.UNCHANGED``) retains the\n        existing request. This allows you to change the request for some\n        parameters and not others.\n\n        .. versionadded:: 1.3\n\n        .. note::\n            This method is only relevant if this estimator is used as a\n            sub-estimator of a meta-estimator, e.g. used inside a\n            :class:`~sklearn.pipeline.Pipeline`. Otherwise it has no effect.\n\n        Parameters\n        ----------\n'
REQUESTER_DOC_PARAM = '        {metadata} : str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED\n            Metadata routing for ``{metadata}`` parameter in ``{method}``.\n\n'
REQUESTER_DOC_RETURN = '        Returns\n        -------\n        self : object\n            The updated object.\n'

class RequestMethod:
    """
    A descriptor for request methods.

    .. versionadded:: 1.3

    Parameters
    ----------
    name : str
        The name of the method for which the request function should be
        created, e.g. ``"fit"`` would create a ``set_fit_request`` function.

    keys : list of str
        A list of strings which are accepted parameters by the created
        function, e.g. ``["sample_weight"]`` if the corresponding method
        accepts it as a metadata.

    validate_keys : bool, default=True
        Whether to check if the requested parameters fit the actual parameters
        of the method.

    Notes
    -----
    This class is a descriptor [1]_ and uses PEP-362 to set the signature of
    the returned function [2]_.

    References
    ----------
    .. [1] https://docs.python.org/3/howto/descriptor.html

    .. [2] https://www.python.org/dev/peps/pep-0362/
    """

    def __init__(self, name, keys, validate_keys=True):
        if False:
            return 10
        self.name = name
        self.keys = keys
        self.validate_keys = validate_keys

    def __get__(self, instance, owner):
        if False:
            for i in range(10):
                print('nop')

        def func(**kw):
            if False:
                print('Hello World!')
            'Updates the request for provided parameters\n\n            This docstring is overwritten below.\n            See REQUESTER_DOC for expected functionality\n            '
            if not _routing_enabled():
                raise RuntimeError('This method is only available when metadata routing is enabled. You can enable it using sklearn.set_config(enable_metadata_routing=True).')
            if self.validate_keys and set(kw) - set(self.keys):
                raise TypeError(f'Unexpected args: {set(kw) - set(self.keys)}. Accepted arguments are: {set(self.keys)}')
            requests = instance._get_metadata_request()
            method_metadata_request = getattr(requests, self.name)
            for (prop, alias) in kw.items():
                if alias is not UNCHANGED:
                    method_metadata_request.add_request(param=prop, alias=alias)
            instance._metadata_request = requests
            return instance
        func.__name__ = f'set_{self.name}_request'
        params = [inspect.Parameter(name='self', kind=inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=owner)]
        params.extend([inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY, default=UNCHANGED, annotation=Optional[Union[bool, None, str]]) for k in self.keys])
        func.__signature__ = inspect.Signature(params, return_annotation=owner)
        doc = REQUESTER_DOC.format(method=self.name)
        for metadata in self.keys:
            doc += REQUESTER_DOC_PARAM.format(metadata=metadata, method=self.name)
        doc += REQUESTER_DOC_RETURN
        func.__doc__ = doc
        return func

class _MetadataRequester:
    """Mixin class for adding metadata request functionality.

    ``BaseEstimator`` inherits from this Mixin.

    .. versionadded:: 1.3
    """
    if TYPE_CHECKING:

        def set_fit_request(self, **kwargs):
            if False:
                i = 10
                return i + 15
            pass

        def set_partial_fit_request(self, **kwargs):
            if False:
                print('Hello World!')
            pass

        def set_predict_request(self, **kwargs):
            if False:
                print('Hello World!')
            pass

        def set_predict_proba_request(self, **kwargs):
            if False:
                while True:
                    i = 10
            pass

        def set_predict_log_proba_request(self, **kwargs):
            if False:
                i = 10
                return i + 15
            pass

        def set_decision_function_request(self, **kwargs):
            if False:
                print('Hello World!')
            pass

        def set_score_request(self, **kwargs):
            if False:
                while True:
                    i = 10
            pass

        def set_split_request(self, **kwargs):
            if False:
                while True:
                    i = 10
            pass

        def set_transform_request(self, **kwargs):
            if False:
                return 10
            pass

        def set_inverse_transform_request(self, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            pass

    def __init_subclass__(cls, **kwargs):
        if False:
            print('Hello World!')
        'Set the ``set_{method}_request`` methods.\n\n        This uses PEP-487 [1]_ to set the ``set_{method}_request`` methods. It\n        looks for the information available in the set default values which are\n        set using ``__metadata_request__*`` class attributes, or inferred\n        from method signatures.\n\n        The ``__metadata_request__*`` class attributes are used when a method\n        does not explicitly accept a metadata through its arguments or if the\n        developer would like to specify a request value for those metadata\n        which are different from the default ``None``.\n\n        References\n        ----------\n        .. [1] https://www.python.org/dev/peps/pep-0487\n        '
        try:
            requests = cls._get_default_requests()
        except Exception:
            super().__init_subclass__(**kwargs)
            return
        for method in SIMPLE_METHODS:
            mmr = getattr(requests, method)
            if not len(mmr.requests):
                continue
            setattr(cls, f'set_{method}_request', RequestMethod(method, sorted(mmr.requests.keys())))
        super().__init_subclass__(**kwargs)

    @classmethod
    def _build_request_for_signature(cls, router, method):
        if False:
            return 10
        "Build the `MethodMetadataRequest` for a method using its signature.\n\n        This method takes all arguments from the method signature and uses\n        ``None`` as their default request value, except ``X``, ``y``, ``Y``,\n        ``Xt``, ``yt``, ``*args``, and ``**kwargs``.\n\n        Parameters\n        ----------\n        router : MetadataRequest\n            The parent object for the created `MethodMetadataRequest`.\n        method : str\n            The name of the method.\n\n        Returns\n        -------\n        method_request : MethodMetadataRequest\n            The prepared request using the method's signature.\n        "
        mmr = MethodMetadataRequest(owner=cls.__name__, method=method)
        if not hasattr(cls, method) or not inspect.isfunction(getattr(cls, method)):
            return mmr
        params = list(inspect.signature(getattr(cls, method)).parameters.items())[1:]
        for (pname, param) in params:
            if pname in {'X', 'y', 'Y', 'Xt', 'yt'}:
                continue
            if param.kind in {param.VAR_POSITIONAL, param.VAR_KEYWORD}:
                continue
            mmr.add_request(param=pname, alias=None)
        return mmr

    @classmethod
    def _get_default_requests(cls):
        if False:
            for i in range(10):
                print('nop')
        'Collect default request values.\n\n        This method combines the information present in ``__metadata_request__*``\n        class attributes, as well as determining request keys from method\n        signatures.\n        '
        requests = MetadataRequest(owner=cls.__name__)
        for method in SIMPLE_METHODS:
            setattr(requests, method, cls._build_request_for_signature(router=requests, method=method))
        defaults = dict()
        for base_class in reversed(inspect.getmro(cls)):
            base_defaults = {attr: value for (attr, value) in vars(base_class).items() if '__metadata_request__' in attr}
            defaults.update(base_defaults)
        defaults = dict(sorted(defaults.items()))
        for (attr, value) in defaults.items():
            substr = '__metadata_request__'
            method = attr[attr.index(substr) + len(substr):]
            for (prop, alias) in value.items():
                getattr(requests, method).add_request(param=prop, alias=alias)
        return requests

    def _get_metadata_request(self):
        if False:
            while True:
                i = 10
        'Get requested data properties.\n\n        Please check :ref:`User Guide <metadata_routing>` on how the routing\n        mechanism works.\n\n        Returns\n        -------\n        request : MetadataRequest\n            A :class:`~sklearn.utils.metadata_routing.MetadataRequest` instance.\n        '
        if hasattr(self, '_metadata_request'):
            requests = get_routing_for_object(self._metadata_request)
        else:
            requests = self._get_default_requests()
        return requests

    def get_metadata_routing(self):
        if False:
            print('Hello World!')
        'Get metadata routing of this object.\n\n        Please check :ref:`User Guide <metadata_routing>` on how the routing\n        mechanism works.\n\n        Returns\n        -------\n        routing : MetadataRequest\n            A :class:`~sklearn.utils.metadata_routing.MetadataRequest` encapsulating\n            routing information.\n        '
        return self._get_metadata_request()

def process_routing(_obj, _method, /, **kwargs):
    if False:
        return 10
    'Validate and route input parameters.\n\n    This function is used inside a router\'s method, e.g. :term:`fit`,\n    to validate the metadata and handle the routing.\n\n    Assuming this signature: ``fit(self, X, y, sample_weight=None, **fit_params)``,\n    a call to this function would be:\n    ``process_routing(self, sample_weight=sample_weight, **fit_params)``.\n\n    Note that if routing is not enabled and ``kwargs`` is empty, then it\n    returns an empty routing where ``process_routing(...).ANYTHING.ANY_METHOD``\n    is always an empty dictionary.\n\n    .. versionadded:: 1.3\n\n    Parameters\n    ----------\n    _obj : object\n        An object implementing ``get_metadata_routing``. Typically a\n        meta-estimator.\n\n    _method : str\n        The name of the router\'s method in which this function is called.\n\n    **kwargs : dict\n        Metadata to be routed.\n\n    Returns\n    -------\n    routed_params : Bunch\n        A :class:`~sklearn.utils.Bunch` of the form ``{"object_name": {"method_name":\n        {prop: value}}}`` which can be used to pass the required metadata to\n        corresponding methods or corresponding child objects. The object names\n        are those defined in `obj.get_metadata_routing()`.\n    '
    if not _routing_enabled() and (not kwargs):

        class EmptyRequest:

            def get(self, name, default=None):
                if False:
                    for i in range(10):
                        print('nop')
                return default if default else {}

            def __getitem__(self, name):
                if False:
                    i = 10
                    return i + 15
                return Bunch(**{method: dict() for method in METHODS})

            def __getattr__(self, name):
                if False:
                    while True:
                        i = 10
                return Bunch(**{method: dict() for method in METHODS})
        return EmptyRequest()
    if not (hasattr(_obj, 'get_metadata_routing') or isinstance(_obj, MetadataRouter)):
        raise AttributeError(f'The given object ({repr(_obj.__class__.__name__)}) needs to either implement the routing method `get_metadata_routing` or be a `MetadataRouter` instance.')
    if _method not in METHODS:
        raise TypeError(f'Can only route and process input on these methods: {METHODS}, while the passed method is: {_method}.')
    request_routing = get_routing_for_object(_obj)
    request_routing.validate_metadata(params=kwargs, method=_method)
    routed_params = request_routing.route_params(params=kwargs, caller=_method)
    return routed_params