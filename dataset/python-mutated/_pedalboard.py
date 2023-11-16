import re
import platform
import weakref
from contextlib import contextmanager
from typing import cast, List, Optional, Dict, Tuple, Iterable, Type, Union, Set, no_type_check
from pedalboard_native import Plugin, ExternalPlugin, _AudioProcessorParameter
from pedalboard_native.utils import Chain

class Pedalboard(Chain):
    """
    A container for a series of :class:`Plugin` objects, to use for processing audio, like a
    `guitar pedalboard <https://en.wikipedia.org/wiki/Guitar_pedalboard>`_.

    :class:`Pedalboard` objects act like regular Python ``List`` objects,
    but come with an additional :py:meth:`process` method (also aliased to :py:meth:`__call__`),
    allowing audio to be passed through the entire :class:`Pedalboard` object for processing::

        my_pedalboard = Pedalboard()
        my_pedalboard.append(Reverb())
        output_audio = my_pedalboard(input_audio)

    .. warning::
        :class:`Pedalboard` objects may only contain effects plugins (i.e.: those for which
        :attr:`is_effect` is ``True``), and cannot contain instrument plugins (i.e.: those
        for which :attr:`is_instrument` is ``True``).
    """

    def __init__(self, plugins: Optional[List[Plugin]]=None):
        if False:
            i = 10
            return i + 15
        super().__init__(plugins or [])

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return '<{} with {} plugin{}: {}>'.format(self.__class__.__name__, len(self), '' if len(self) == 1 else 's', list(self))
FLOAT_SUFFIXES_TO_IGNORE: Set[str] = set(['x', '%', '*', ',', '.', 'hz', 'ms', 'sec', 'seconds', 'dB', 'dBTP'])

def strip_common_float_suffixes(s: Union[float, str, bool], strip_si_prefixes: bool=True) -> Union[float, str, bool]:
    if False:
        i = 10
        return i + 15
    if not isinstance(s, str) or (hasattr(s, 'type') and s.type != str):
        return s
    s = s.strip()
    if strip_si_prefixes:
        if s.lower().endswith('khz') and len(s) > 3:
            try:
                s = str(float(s[:-3]) * 1000)
            except ValueError:
                pass
    for suffix in FLOAT_SUFFIXES_TO_IGNORE:
        if suffix == 'hz' and 'khz' in s.lower():
            continue
        if s[-len(suffix):].lower() == suffix.lower():
            s = s[:-len(suffix)].strip()
    return s

def looks_like_float(s: Union[float, str]) -> bool:
    if False:
        print('Hello World!')
    if isinstance(s, float):
        return True
    try:
        float(strip_common_float_suffixes(s))
        return True
    except ValueError:
        return False

class ReadOnlyDictWrapper(dict):

    def __setitem__(self, name, value):
        if False:
            return 10
        raise TypeError(f'The .parameters dictionary on a Plugin instance returns a read-only dictionary of its parameters. To change a parameter, set the parameter on the plugin directly as an attribute. (`my_plugin.{name} = {value}`)')

@no_type_check
def wrap_type(base_type):
    if False:
        for i in range(10):
            print('nop')

    @no_type_check
    class WeakTypeWrapper(base_type):
        """
        A wrapper around `base_type` that allows adding additional
        accessors through a weak reference. Useful for syntax convenience.
        """

        def __new__(cls, value, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            try:
                return base_type.__new__(cls, value)
            except TypeError:
                return base_type.__new__(cls)

        def __init__(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            if 'wrapped' in kwargs:
                self._wrapped = weakref.ref(kwargs['wrapped'])
                del kwargs['wrapped']
            else:
                raise ValueError("WeakTypeWrapper({}) expected to be passed a 'wrapped' kwarg.".format(base_type))
            try:
                super().__init__(*args, **kwargs)
            except TypeError:
                pass

        def __getattr__(self, name):
            if False:
                return 10
            wrapped = self._wrapped()
            if hasattr(wrapped, name):
                return getattr(wrapped, name)
            if hasattr(super(), '__getattr__'):
                return super().__getattr__(name)
            raise AttributeError("'{}' has no attribute '{}'".format(base_type.__name__, name))

        def __dir__(self) -> Iterable[str]:
            if False:
                print('Hello World!')
            wrapped = self._wrapped()
            if wrapped:
                return list(dir(wrapped)) + list(super().__dir__())
            return super().__dir__()
    return WeakTypeWrapper

@no_type_check
class WrappedBool(object):

    def __init__(self, value):
        if False:
            i = 10
            return i + 15
        if not isinstance(value, bool):
            raise TypeError(f'WrappedBool should be passed a boolean, got {type(value)}')
        self.__value = value

    def __repr__(self):
        if False:
            while True:
                i = 10
        return repr(self.__value)

    def __eq__(self, o: object) -> bool:
        if False:
            i = 10
            return i + 15
        return self.__value == o

    def __hash__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return hash(self.__value)

    def __bool__(self):
        if False:
            for i in range(10):
                print('nop')
        return bool(self.__value)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return str(self.__value)

    def __getattr__(self, attr: str):
        if False:
            print('Hello World!')
        return getattr(self.__value, attr)

    def __hasattr__(self, attr: str):
        if False:
            i = 10
            return i + 15
        return hasattr(self.__value, attr)
StringWithParameter: Type[str] = wrap_type(str)
FloatWithParameter: Type[float] = wrap_type(float)
BooleanWithParameter: Type[bool] = wrap_type(WrappedBool)
PARAMETER_NAME_REGEXES_TO_IGNORE = set([re.compile(pattern) for pattern in ['MIDI CC ', 'P\\d\\d\\d']])
TRUE_BOOLEANS: Set[str] = {'on', 'yes', 'true', 'enabled'}

def get_text_for_raw_value(cpp_parameter: _AudioProcessorParameter, raw_value: float, slow: bool=False) -> Optional[str]:
    if False:
        i = 10
        return i + 15
    if slow:
        original_value = cpp_parameter.raw_value
        try:
            cpp_parameter.raw_value = raw_value
            return cpp_parameter.string_value
        finally:
            cpp_parameter.raw_value = original_value
    else:
        return cpp_parameter.get_text_for_raw_value(raw_value)

class AudioProcessorParameter(object):
    """
    A wrapper around various different parameters exposed by
    :class:`VST3Plugin` or :class:`AudioUnitPlugin` instances.

    :class:`AudioProcessorParameter` objects are rarely used directly,
    and usually used via their implicit interface::

       my_plugin = load_plugin("My Cool Audio Effect.vst3")
       # Print all of the parameter names:
       print(my_plugin.parameters.keys())
       # ["mix", "delay_time_ms", "foobar"]
       # Access each parameter as if it were just a Python attribute:
       my_plugin.mix = 0.5
       my_plugin.delay_time_ms = 400

    .. note::
        :class:`AudioProcessorParameter` tries to guess the range of
        valid parameter values, as well as the type/unit of the parameter,
        when instantiated. This guess may not always be accurate.
        Raw control over the underlying parameter's value can be had by
        accessing the :py:attr:`raw_value` attribute, which is always bounded
        on [0, 1] and is passed directly to the underlying plugin object.
    """

    def __init__(self, plugin: 'ExternalPlugin', parameter_name: str, search_steps: int=1000):
        if False:
            for i in range(10):
                print('nop')
        self.__plugin = plugin
        self.__parameter_name = parameter_name
        self.ranges: Dict[Tuple[float, float], Union[str, float, bool]] = {}
        with self.__get_cpp_parameter() as cpp_parameter:
            for fetch_slow in (False, True):
                start_of_range: float = 0
                text_value: Optional[str] = None
                self.ranges = {}
                for x in range(0, search_steps + 1):
                    raw_value = x / search_steps
                    x_text_value = get_text_for_raw_value(cpp_parameter, raw_value, fetch_slow)
                    if text_value is None:
                        text_value = x_text_value
                    elif x_text_value != text_value:
                        self.ranges[start_of_range, raw_value] = text_value
                        text_value = x_text_value
                        start_of_range = raw_value
                results_look_incorrect = not self.ranges or (len(self.ranges) == 1 and all((looks_like_float(v) for v in self.ranges.values())))
                if not results_look_incorrect:
                    break
            if text_value is None:
                raise NotImplementedError(f"Plugin parameter '{parameter_name}' failed to return a valid string for its value.")
            self.ranges[start_of_range, 1] = text_value
            self.python_name = to_python_parameter_name(cpp_parameter)
        self.min_value = None
        self.max_value = None
        self.step_size = None
        self.approximate_step_size = None
        self.type: Type = str
        if all((looks_like_float(v) for v in self.ranges.values())):
            self.type = float
            float_ranges = {k: float(strip_common_float_suffixes(v)) for (k, v) in self.ranges.items()}
            self.min_value = min(float_ranges.values())
            self.max_value = max(float_ranges.values())
            if not self.label:
                a_value = next(iter(self.ranges.values()))
                if isinstance(a_value, str):
                    stripped_value = strip_common_float_suffixes(a_value, strip_si_prefixes=False)
                    if stripped_value != a_value and isinstance(stripped_value, str) and (stripped_value in a_value):
                        all_possible_labels = set()
                        for value in self.ranges.values():
                            if not isinstance(value, str):
                                continue
                            stripped_value = strip_common_float_suffixes(value, strip_si_prefixes=False)
                            if not isinstance(stripped_value, str):
                                continue
                            all_possible_labels.add(value.replace(stripped_value, '').strip())
                        if len(all_possible_labels) == 1:
                            self._label = next(iter(all_possible_labels))
            sorted_values = sorted(float_ranges.values())
            first_derivative_steps = set([round(abs(b - a), 8) for (a, b) in zip(sorted_values, sorted_values[1:])])
            if len(first_derivative_steps) == 1:
                self.step_size = next(iter(first_derivative_steps))
            elif first_derivative_steps:
                self.approximate_step_size = sum(first_derivative_steps) / len(first_derivative_steps)
            self.ranges = dict(float_ranges)
        elif len(self.ranges) == 2 and TRUE_BOOLEANS & {v.lower() if isinstance(v, str) else v for v in self.ranges.values()}:
            self.type = bool
            self.ranges = {k: (v.lower() if isinstance(v, str) else v) in TRUE_BOOLEANS for (k, v) in self.ranges.items()}
            self.min_value = False
            self.max_value = True
            self.step_size = 1
        self.valid_values = list(self.ranges.values())
        self.range = (self.min_value, self.max_value, self.step_size)
        self._value_to_raw_value_ranges = {value: _range for (_range, value) in self.ranges.items()}

    @contextmanager
    def __get_cpp_parameter(self):
        if False:
            print('Hello World!')
        '\n        The C++ version of this class (`_AudioProcessorParameter`) is owned\n        by the ExternalPlugin object and is not guaranteed to exist at the\n        same memory address every time we might need it. This Python wrapper\n        looks it up dynamically.\n        '
        _parameter = self.__plugin._get_parameter(self.__parameter_name)
        if _parameter and _parameter.name == self.__parameter_name:
            yield _parameter
            return
        raise RuntimeError('Parameter {} on plugin {} is no longer available. This could indicate that the plugin has changed parameters.'.format(self.__parameter_name, self.__plugin))

    @property
    def label(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        The units used by this parameter (Hz, dB, etc).\n\n        May be ``None`` if the plugin does not expose units for this\n        parameter or if automatic unit detection fails.\n        '
        if hasattr(self, '_label') and self._label:
            return self._label
        with self.__get_cpp_parameter() as parameter:
            if parameter.label:
                return parameter.label
        return None

    @property
    def units(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Alias for "label" - the units used by this parameter (Hz, dB, etc).\n\n        May be ``None`` if the plugin does not expose units for this\n        parameter or if automatic unit detection fails.\n        '
        return self.label

    def __repr__(self):
        if False:
            print('Hello World!')
        with self.__get_cpp_parameter() as parameter:
            cpp_repr_value = repr(parameter)
            cpp_repr_value = cpp_repr_value.rstrip('>')
            if self.type is float:
                if self.step_size:
                    return '{} value={} range=({}, {}, {})>'.format(cpp_repr_value, self.string_value, self.min_value, self.max_value, self.step_size)
                elif self.approximate_step_size:
                    return '{} value={} range=({}, {}, ~{})>'.format(cpp_repr_value, self.string_value, self.min_value, self.max_value, self.approximate_step_size)
                else:
                    return '{} value={} range=({}, {}, ?)>'.format(cpp_repr_value, self.string_value, self.min_value, self.max_value)
            elif self.type is str:
                return '{} value="{}" ({} valid string value{})>'.format(cpp_repr_value, self.string_value, len(self.valid_values), '' if len(self.valid_values) == 1 else 's')
            elif self.type is bool:
                return '{} value={} boolean ("{}" and "{}")>'.format(cpp_repr_value, self.string_value, self.valid_values[0], self.valid_values[1])
            else:
                raise ValueError(f"Parameter {self.python_name} has an unknown type. (Found '{self.type}')")

    def __getattr__(self, name: str):
        if False:
            for i in range(10):
                print('nop')
        if not name.startswith('_'):
            try:
                with self.__get_cpp_parameter() as parameter:
                    return getattr(parameter, name)
            except RuntimeError:
                pass
        if hasattr(super(), '__getattr__'):
            return super().__getattr__(name)
        raise AttributeError("'{}' has no attribute '{}'".format(self.__class__.__name__, name))

    def __setattr__(self, name: str, value):
        if False:
            i = 10
            return i + 15
        if not name.startswith('_'):
            try:
                with self.__get_cpp_parameter() as parameter:
                    if hasattr(parameter, name):
                        return setattr(parameter, name, value)
            except RuntimeError:
                pass
        return super().__setattr__(name, value)

    def get_raw_value_for(self, new_value: Union[float, str, bool]) -> float:
        if False:
            print('Hello World!')
        if self.type is float:
            to_float_value: Union[str, float, bool]
            if isinstance(new_value, str) and self.label and new_value.endswith(self.label):
                to_float_value = new_value[:-len(self.label)]
            else:
                to_float_value = new_value
            try:
                new_value = float(to_float_value)
            except ValueError:
                if self.label:
                    raise ValueError("Value received for parameter '{}' ({}) must be a number or a string (with the optional suffix '{}')".format(self.python_name, repr(new_value), self.label))
                raise ValueError("Value received for parameter '{}' ({}) must be a number or a string".format(self.python_name, repr(new_value)))
            if self.min_value is not None and new_value < self.min_value or (self.max_value is not None and new_value > self.max_value):
                raise ValueError("Value received for parameter '{}' ({}) is out of range [{}{}, {}{}]".format(self.python_name, repr(new_value), self.min_value, self.label, self.max_value, self.label))
            plugin_reported_raw_value = self.get_raw_value_for_text(str(new_value))
            closest_diff = None
            closest_range_value: Optional[Tuple[float, float]] = None
            for (value, raw_value_range) in self._value_to_raw_value_ranges.items():
                if not isinstance(value, float):
                    continue
                diff = new_value - value
                if closest_diff is None or abs(diff) < abs(closest_diff):
                    closest_range_value = raw_value_range
                    closest_diff = diff
            if closest_range_value is not None:
                (expected_low, expected_high) = closest_range_value
                if plugin_reported_raw_value < expected_low or plugin_reported_raw_value > expected_high:
                    return expected_low
            return plugin_reported_raw_value
        elif self.type is str:
            if isinstance(new_value, (str, int, float, bool)):
                new_value = str(new_value)
            else:
                raise ValueError("Value received for parameter '{}' ({}) should be a string (or string-like), but got an object of type: {}".format(self.python_name, repr(new_value), type(new_value)))
            if new_value not in self.valid_values:
                raise ValueError("Value received for parameter '{}' ({}) not in list of valid values: {}".format(self.python_name, repr(new_value), self.valid_values))
            plugin_reported_raw_value = self.get_raw_value_for_text(new_value)
            (expected_low, expected_high) = self._value_to_raw_value_ranges[new_value]
            if plugin_reported_raw_value < expected_low or plugin_reported_raw_value > expected_high:
                return expected_low
            else:
                return plugin_reported_raw_value
        elif self.type is bool:
            if not isinstance(new_value, (bool, WrappedBool)):
                raise ValueError("Value received for parameter '{}' ({}) should be a boolean, but got an object of type: {}".format(self.python_name, repr(new_value), type(new_value)))
            return 1.0 if new_value else 0.0
        else:
            raise ValueError('Parameter has invalid type: {}. This should not be possible!'.format(self.type))

def to_python_parameter_name(parameter: _AudioProcessorParameter) -> Optional[str]:
    if False:
        print('Hello World!')
    if not parameter.name and (not parameter.label):
        return None
    name = parameter.name
    if parameter.label and (not parameter.label.startswith(':')):
        name = '{} {}'.format(name, parameter.label)
    return normalize_python_parameter_name(name)

def normalize_python_parameter_name(name: str) -> str:
    if False:
        return 10
    name = name.lower().strip()
    name = name.replace('#', '_sharp').replace('♯', '_sharp').replace('♭', '_flat')
    name_chars = [c if (c.isalpha() or c.isnumeric()) and c.isprintable() and (ord(c) < 128) else '_' for c in name]
    name_chars = [a for (a, b) in zip(name_chars, name_chars[1:]) if a != b or b != '_'] + [name_chars[-1]]
    name = ''.join(name_chars).strip('_')
    return name

class _PythonExternalPluginMixin:

    def __set_initial_parameter_values__(self, parameter_values: Optional[Dict[str, Union[str, int, float, bool]]]):
        if False:
            i = 10
            return i + 15
        if parameter_values is None:
            parameter_values = {}
        if not isinstance(parameter_values, dict):
            raise TypeError(f'Expected a dictionary to be passed to parameter_values, but received a {type(parameter_values).__name__}. (If passing a plugin name, pass "plugin_name=..." as a keyword argument instead.)')
        parameters = self.parameters
        for (key, value) in parameter_values.items():
            if key not in parameters:
                raise AttributeError('Parameter named "{}" not found. Valid options: {}'.format(key, ', '.join(self._parameter_weakrefs.keys())))
            setattr(self, key, value)

    @property
    def parameters(self) -> Dict[str, AudioProcessorParameter]:
        if False:
            for i in range(10):
                print('nop')
        return ReadOnlyDictWrapper(self._get_parameters())

    def _get_parameters(self):
        if False:
            i = 10
            return i + 15
        if not hasattr(self, '__python_parameter_cache__'):
            self.__python_parameter_cache__ = {}
        if not hasattr(self, '__python_to_cpp_names__'):
            self.__python_to_cpp_names__ = {}
        parameters = {}
        for cpp_parameter in self._parameters:
            if any([regex.match(cpp_parameter.name) for regex in PARAMETER_NAME_REGEXES_TO_IGNORE]):
                continue
            if cpp_parameter.name not in self.__python_parameter_cache__:
                self.__python_parameter_cache__[cpp_parameter.name] = AudioProcessorParameter(self, cpp_parameter.name)
            parameter = self.__python_parameter_cache__[cpp_parameter.name]
            if parameter.python_name:
                parameters[parameter.python_name] = parameter
                self.__python_to_cpp_names__[parameter.python_name] = cpp_parameter.name
        return parameters

    def _get_parameter_by_python_name(self, python_name: str) -> Optional[AudioProcessorParameter]:
        if False:
            for i in range(10):
                print('nop')
        if not hasattr(self, '__python_parameter_cache__'):
            self.__python_parameter_cache__ = {}
        if not hasattr(self, '__python_to_cpp_names__'):
            self.__python_to_cpp_names__ = {}
        cpp_name = self.__python_to_cpp_names__.get(python_name)
        if not cpp_name:
            return self._get_parameters().get(python_name)
        cpp_parameter = self._get_parameter(cpp_name)
        if not cpp_parameter:
            return None
        if cpp_parameter.name not in self.__python_parameter_cache__:
            self.__python_parameter_cache__[cpp_parameter.name] = AudioProcessorParameter(cast(ExternalPlugin, self), cpp_parameter.name)
        return self.__python_parameter_cache__[cpp_parameter.name]

    def __dir__(self):
        if False:
            print('Hello World!')
        parameter_names = []
        for parameter in self._parameters:
            name = to_python_parameter_name(parameter)
            if name:
                parameter_names.append(name)
        return super().__dir__() + parameter_names

    def __getattr__(self, name: str):
        if False:
            print('Hello World!')
        if not name.startswith('_'):
            parameter = self._get_parameter_by_python_name(name)
            if parameter:
                string_value = parameter.string_value
                if parameter.type is float:
                    return FloatWithParameter(float(strip_common_float_suffixes(string_value)), wrapped=parameter)
                elif parameter.type is bool:
                    return BooleanWithParameter(parameter.raw_value >= 0.5, wrapped=parameter)
                elif parameter.type is str:
                    return StringWithParameter(str(string_value), wrapped=parameter)
                else:
                    raise ValueError(f"Parameter {parameter.python_name} has an unknown type. (Found '{parameter.type}')")
        return getattr(super(), name)

    def __setattr__(self, name: str, value):
        if False:
            return 10
        if not name.startswith('__'):
            parameter = self._get_parameter_by_python_name(name)
            if parameter:
                parameter.raw_value = parameter.get_raw_value_for(value)
                return
        super().__setattr__(name, value)
ExternalPlugin.__bases__ = ExternalPlugin.__bases__ + (_PythonExternalPluginMixin,)
_AVAILABLE_PLUGIN_CLASSES: List[Type[ExternalPlugin]] = []
try:
    from pedalboard_native import VST3Plugin
    _AVAILABLE_PLUGIN_CLASSES.append(VST3Plugin)
except ImportError:
    pass
try:
    from pedalboard_native import AudioUnitPlugin
    _AVAILABLE_PLUGIN_CLASSES.append(AudioUnitPlugin)
except ImportError:
    pass

def load_plugin(path_to_plugin_file: str, parameter_values: Dict[str, Union[str, int, float, bool]]={}, plugin_name: Union[str, None]=None, initialization_timeout: float=10.0) -> ExternalPlugin:
    if False:
        i = 10
        return i + 15
    '\n    Load an audio plugin.\n\n    Two plugin formats are supported:\n     - VST3® format is supported on macOS, Windows, and Linux\n     - Audio Units are supported on macOS\n\n    Args:\n        path_to_plugin_file (``str``): The path of a VST3® or Audio Unit plugin file or bundle.\n\n        parameter_values (``Dict[str, Union[str, int, float, bool]]``):\n            An optional dictionary of initial values to provide to the plugin\n            after loading. Keys in this dictionary are expected to match the\n            parameter names reported by the plugin, but normalized to strings\n            that can be used as Python identifiers. (These are the same\n            identifiers that are used as keys in the ``.parameters`` dictionary\n            of a loaded plugin.)\n\n        plugin_name (``Optional[str]``):\n            An optional plugin name that can be used to load a specific plugin\n            from a multi-plugin package. If a package is loaded but a\n            ``plugin_name`` is not provided, an exception will be thrown.\n\n        initialization_timeout (``float``):\n            The number of seconds that Pedalboard will spend trying to load this plugin.\n            Some plugins load resources asynchronously in the background on startup;\n            using larger values for this parameter can give these plugins time to\n            load properly.\n\n            *Introduced in v0.7.6.*\n\n    Returns:\n        an instance of :class:`pedalboard.VST3Plugin` or :class:`pedalboard.AudioUnitPlugin`\n\n    Throws:\n        ``ImportError``: if the plugin cannot be found or loaded\n\n        ``RuntimeError``: if the plugin file contains more than one plugin,\n        but no ``plugin_name`` was provided\n    '
    if not _AVAILABLE_PLUGIN_CLASSES:
        raise ImportError('Pedalboard found no supported external plugin types in this installation ({}).'.format(platform.system()))
    exceptions = []
    for plugin_class in _AVAILABLE_PLUGIN_CLASSES:
        try:
            return plugin_class(path_to_plugin_file=path_to_plugin_file, parameter_values=parameter_values, plugin_name=plugin_name, initialization_timeout=initialization_timeout)
        except ImportError as e:
            exceptions.append(e)
        except Exception:
            raise
    else:
        tried_plugins = ', '.join([c.__name__ for c in _AVAILABLE_PLUGIN_CLASSES])
        if len(_AVAILABLE_PLUGIN_CLASSES) > 2:
            tried_plugins = ', or '.join(tried_plugins.rsplit(', ', 1))
        else:
            tried_plugins = ' or '.join(tried_plugins.rsplit(', ', 1))
        raise ImportError('Failed to load plugin as {}. Errors were:\n\t{}'.format(tried_plugins, '\n\t'.join(['{}: {}'.format(klass.__name__, exception) for (klass, exception) in zip(_AVAILABLE_PLUGIN_CLASSES, exceptions)])))