import re
import warnings
import weakref
from contextlib import contextmanager
from typing import Callable, Dict, ItemsView, Iterator, KeysView, Optional, Tuple, Union
import torch
from torch.distributions import constraints, transform_to
from torch.serialization import MAP_LOCATION
from typing_extensions import TypedDict

class StateDict(TypedDict):
    params: Dict[str, torch.Tensor]
    constraints: Dict[str, constraints.Constraint]

class ParamStoreDict:
    """
    Global store for parameters in Pyro. This is basically a key-value store.
    The typical user interacts with the ParamStore primarily through the
    primitive `pyro.param`.

    See `Introduction <http://pyro.ai/examples/intro_long.html>`_ for further discussion
    and `SVI Part I <http://pyro.ai/examples/svi_part_i.html>`_ for some examples.

    Some things to bear in mind when using parameters in Pyro:

    - parameters must be assigned unique names
    - the `init_tensor` argument to `pyro.param` is only used the first time that a given (named)
      parameter is registered with Pyro.
    - for this reason, a user may need to use the `clear()` method if working in a REPL in order to
      get the desired behavior. this method can also be invoked with `pyro.clear_param_store()`.
    - the internal name of a parameter within a PyTorch `nn.Module` that has been registered with
      Pyro is prepended with the Pyro name of the module. so nothing prevents the user from having
      two different modules each of which contains a parameter named `weight`. by contrast, a user
      can only have one top-level parameter named `weight` (outside of any module).
    - parameters can be saved and loaded from disk using `save` and `load`.
    - in general parameters are associated with both *constrained* and *unconstrained* values. for
      example, under the hood a parameter that is constrained to be positive is represented as an
      unconstrained tensor in log space.
    """

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        '\n        initialize ParamStore data structures\n        '
        self._params: Dict[str, torch.Tensor] = {}
        self._param_to_name: Dict[torch.Tensor, str] = {}
        self._constraints: Dict[str, constraints.Constraint] = {}

    def clear(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Clear the ParamStore\n        '
        self._params = {}
        self._param_to_name = {}
        self._constraints = {}

    def items(self) -> Iterator[Tuple[str, torch.Tensor]]:
        if False:
            i = 10
            return i + 15
        '\n        Iterate over ``(name, constrained_param)`` pairs. Note that `constrained_param` is\n        in the constrained (i.e. user-facing) space.\n        '
        for name in self._params:
            yield (name, self[name])

    def keys(self) -> KeysView[str]:
        if False:
            print('Hello World!')
        '\n        Iterate over param names.\n        '
        return self._params.keys()

    def values(self) -> Iterator[torch.Tensor]:
        if False:
            i = 10
            return i + 15
        '\n        Iterate over constrained parameter values.\n        '
        for (name, constrained_param) in self.items():
            yield constrained_param

    def __bool__(self) -> bool:
        if False:
            i = 10
            return i + 15
        return bool(self._params)

    def __len__(self) -> int:
        if False:
            print('Hello World!')
        return len(self._params)

    def __contains__(self, name: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return name in self._params

    def __iter__(self) -> Iterator[str]:
        if False:
            print('Hello World!')
        '\n        Iterate over param names.\n        '
        return iter(self.keys())

    def __delitem__(self, name) -> None:
        if False:
            while True:
                i = 10
        '\n        Remove a parameter from the param store.\n        '
        unconstrained_value = self._params.pop(name)
        self._param_to_name.pop(unconstrained_value)
        self._constraints.pop(name)

    def __getitem__(self, name: str) -> torch.Tensor:
        if False:
            return 10
        '\n        Get the *constrained* value of a named parameter.\n        '
        unconstrained_value = self._params[name]
        constraint = self._constraints[name]
        constrained_value: torch.Tensor = transform_to(constraint)(unconstrained_value)
        constrained_value.unconstrained = weakref.ref(unconstrained_value)
        return constrained_value

    def __setitem__(self, name: str, new_constrained_value: torch.Tensor) -> None:
        if False:
            print('Hello World!')
        '\n        Set the constrained value of an existing parameter, or the value of a\n        new *unconstrained* parameter. To declare a new parameter with\n        constraint, use :meth:`setdefault`.\n        '
        constraint = self._constraints.setdefault(name, constraints.real)
        with torch.no_grad():
            unconstrained_value = transform_to(constraint).inv(new_constrained_value)
            unconstrained_value = unconstrained_value.contiguous()
        unconstrained_value.requires_grad_(True)
        self._params[name] = unconstrained_value
        self._param_to_name[unconstrained_value] = name

    def setdefault(self, name: str, init_constrained_value: Union[torch.Tensor, Callable[[], torch.Tensor]], constraint: constraints.Constraint=constraints.real) -> torch.Tensor:
        if False:
            return 10
        '\n        Retrieve a *constrained* parameter value from the if it exists, otherwise\n        set the initial value. Note that this is a little fancier than\n        :meth:`dict.setdefault`.\n\n        If the parameter already exists, ``init_constrained_tensor`` will be ignored. To avoid\n        expensive creation of ``init_constrained_tensor`` you can wrap it in a ``lambda`` that\n        will only be evaluated if the parameter does not already exist::\n\n            param_store.get("foo", lambda: (0.001 * torch.randn(1000, 1000)).exp(),\n                            constraint=constraints.positive)\n\n        :param str name: parameter name\n        :param init_constrained_value: initial constrained value\n        :type init_constrained_value: torch.Tensor or callable returning a torch.Tensor\n        :param constraint: torch constraint object\n        :type constraint: ~torch.distributions.constraints.Constraint\n        :returns: constrained parameter value\n        :rtype: torch.Tensor\n        '
        if name not in self._params:
            self._constraints[name] = constraint
            if callable(init_constrained_value):
                init_constrained_value = init_constrained_value()
            self[name] = init_constrained_value
        return self[name]

    def named_parameters(self) -> ItemsView[str, torch.Tensor]:
        if False:
            i = 10
            return i + 15
        '\n        Returns an iterator over ``(name, unconstrained_value)`` tuples for\n        each parameter in the ParamStore. Note that, in the event the parameter is constrained,\n        `unconstrained_value` is in the unconstrained space implicitly used by the constraint.\n        '
        return self._params.items()

    def get_all_param_names(self) -> KeysView[str]:
        if False:
            print('Hello World!')
        warnings.warn('ParamStore.get_all_param_names() is deprecated; use .keys() instead.', DeprecationWarning)
        return self.keys()

    def replace_param(self, param_name: str, new_param: torch.Tensor, old_param: torch.Tensor) -> None:
        if False:
            i = 10
            return i + 15
        warnings.warn('ParamStore.replace_param() is deprecated; use .__setitem__() instead.', DeprecationWarning)
        assert self._params[param_name] is old_param.unconstrained()
        self[param_name] = new_param

    def get_param(self, name: str, init_tensor: Optional[torch.Tensor]=None, constraint: constraints.Constraint=constraints.real, event_dim: Optional[int]=None) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get parameter from its name. If it does not yet exist in the\n        ParamStore, it will be created and stored.\n        The Pyro primitive `pyro.param` dispatches to this method.\n\n        :param name: parameter name\n        :type name: str\n        :param init_tensor: initial tensor\n        :type init_tensor: torch.Tensor\n        :param constraint: torch constraint\n        :type constraint: torch.distributions.constraints.Constraint\n        :param int event_dim: (ignored)\n        :returns: parameter\n        :rtype: torch.Tensor\n        '
        if init_tensor is None:
            return self[name]
        else:
            return self.setdefault(name, init_tensor, constraint)

    def match(self, name: str) -> Dict[str, torch.Tensor]:
        if False:
            return 10
        '\n        Get all parameters that match regex. The parameter must exist.\n\n        :param name: regular expression\n        :type name: str\n        :returns: dict with key param name and value torch Tensor\n        '
        pattern = re.compile(name)
        return {name: self[name] for name in self if pattern.match(name)}

    def param_name(self, p: torch.Tensor) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get parameter name from parameter\n\n        :param p: parameter\n        :returns: parameter name\n        '
        return self._param_to_name.get(p)

    def get_state(self) -> StateDict:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the ParamStore state.\n        '
        params = self._params.copy()
        for param in params.values():
            param.__dict__.pop('unconstrained', None)
        state: StateDict = {'params': params, 'constraints': self._constraints.copy()}
        return state

    def set_state(self, state: StateDict) -> None:
        if False:
            return 10
        '\n        Set the ParamStore state using state from a previous :meth:`get_state` call\n        '
        assert isinstance(state, dict), 'malformed ParamStore state'
        assert set(state.keys()) == set(['params', 'constraints']), 'malformed ParamStore keys {}'.format(state.keys())
        for (param_name, param) in state['params'].items():
            self._params[param_name] = param
            self._param_to_name[param] = param_name
        for (param_name, constraint) in state['constraints'].items():
            if isinstance(constraint, type(constraints.real)):
                constraint = constraints.real
            self._constraints[param_name] = constraint

    def save(self, filename: str) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Save parameters to file\n\n        :param filename: file name to save to\n        :type filename: str\n        '
        with open(filename, 'wb') as output_file:
            torch.save(self.get_state(), output_file)

    def load(self, filename: str, map_location: MAP_LOCATION=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Loads parameters from file\n\n        .. note::\n\n           If using :meth:`pyro.module` on parameters loaded from\n           disk, be sure to set the ``update_module_params`` flag::\n\n               pyro.get_param_store().load('saved_params.save')\n               pyro.module('module', nn, update_module_params=True)\n\n        :param filename: file name to load from\n        :type filename: str\n        :param map_location: specifies how to remap storage locations\n        :type map_location: function, torch.device, string or a dict\n        "
        with open(filename, 'rb') as input_file:
            state = torch.load(input_file, map_location)
        self.set_state(state)

    @contextmanager
    def scope(self, state: Optional[StateDict]=None) -> Iterator[StateDict]:
        if False:
            i = 10
            return i + 15
        "\n        Context manager for using multiple parameter stores within the same process.\n\n        This is a thin wrapper around :meth:`get_state`, :meth:`clear`, and\n        :meth:`set_state`. For large models where memory space is limiting, you\n        may want to instead manually use :meth:`save`, :meth:`clear`, and\n        :meth:`load`.\n\n        Example usage::\n\n            param_store = pyro.get_param_store()\n\n            # Train multiple models, while avoiding param name conflicts.\n            with param_store.scope() as scope1:\n                # ...Train one model,guide pair...\n            with param_store.scope() as scope2:\n                # ...Train another model,guide pair...\n\n            # Now evaluate each, still avoiding name conflicts.\n            with param_store.scope(scope1):  # loads the first model's scope\n               # ...evaluate the first model...\n            with param_store.scope(scope2):  # loads the second model's scope\n               # ...evaluate the second model...\n        "
        if state is None:
            state = {'params': {}, 'constraints': {}}
        old_state = self.get_state()
        try:
            self.clear()
            self.set_state(state)
            yield state
            state.update(self.get_state())
        finally:
            self.clear()
            self.set_state(old_state)
_MODULE_NAMESPACE_DIVIDER = '$$$'

def param_with_module_name(pyro_name: str, param_name: str) -> str:
    if False:
        i = 10
        return i + 15
    return _MODULE_NAMESPACE_DIVIDER.join([pyro_name, param_name])

def module_from_param_with_module_name(param_name: str) -> str:
    if False:
        print('Hello World!')
    return param_name.split(_MODULE_NAMESPACE_DIVIDER)[0]

def user_param_name(param_name: str) -> str:
    if False:
        while True:
            i = 10
    if _MODULE_NAMESPACE_DIVIDER in param_name:
        return param_name.split(_MODULE_NAMESPACE_DIVIDER)[1]
    return param_name

def normalize_param_name(name: str) -> str:
    if False:
        return 10
    return name.replace(_MODULE_NAMESPACE_DIVIDER, '.')