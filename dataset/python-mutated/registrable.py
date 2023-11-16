"""
`allennlp.common.registrable.Registrable` is a "mixin" for endowing
any base class with a named registry for its subclasses and a decorator
for registering them.
"""
import importlib
import logging
import inspect
from collections import defaultdict
from typing import Callable, ClassVar, DefaultDict, Dict, List, Optional, Tuple, Type, TypeVar, cast, Any
from allennlp.common.checks import ConfigurationError
from allennlp.common.from_params import FromParams
logger = logging.getLogger(__name__)
_T = TypeVar('_T')
_RegistrableT = TypeVar('_RegistrableT', bound='Registrable')
_SubclassRegistry = Dict[str, Tuple[type, Optional[str]]]

class Registrable(FromParams):
    """
    Any class that inherits from `Registrable` gains access to a named registry for its
    subclasses. To register them, just decorate them with the classmethod
    `@BaseClass.register(name)`.

    After which you can call `BaseClass.list_available()` to get the keys for the
    registered subclasses, and `BaseClass.by_name(name)` to get the corresponding subclass.
    Note that the registry stores the subclasses themselves; not class instances.
    In most cases you would then call `from_params(params)` on the returned subclass.

    You can specify a default by setting `BaseClass.default_implementation`.
    If it is set, it will be the first element of `list_available()`.

    Note that if you use this class to implement a new `Registrable` abstract class,
    you must ensure that all subclasses of the abstract class are loaded when the module is
    loaded, because the subclasses register themselves in their respective files. You can
    achieve this by having the abstract class and all subclasses in the __init__.py of the
    module in which they reside (as this causes any import of either the abstract class or
    a subclass to load all other subclasses and the abstract class).
    """
    _registry: ClassVar[DefaultDict[type, _SubclassRegistry]] = defaultdict(dict)
    default_implementation: Optional[str] = None

    @classmethod
    def register(cls, name: str, constructor: Optional[str]=None, exist_ok: bool=False) -> Callable[[Type[_T]], Type[_T]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Register a class under a particular name.\n\n        # Parameters\n\n        name : `str`\n            The name to register the class under.\n        constructor : `str`, optional (default=`None`)\n            The name of the method to use on the class to construct the object.  If this is given,\n            we will use this method (which must be a `@classmethod`) instead of the default\n            constructor.\n        exist_ok : `bool`, optional (default=`False`)\n            If True, overwrites any existing models registered under `name`. Else,\n            throws an error if a model is already registered under `name`.\n\n        # Examples\n\n        To use this class, you would typically have a base class that inherits from `Registrable`:\n\n        ```python\n        class Vocabulary(Registrable):\n            ...\n        ```\n\n        Then, if you want to register a subclass, you decorate it like this:\n\n        ```python\n        @Vocabulary.register("my-vocabulary")\n        class MyVocabulary(Vocabulary):\n            def __init__(self, param1: int, param2: str):\n                ...\n        ```\n\n        Registering a class like this will let you instantiate a class from a config file, where you\n        give `"type": "my-vocabulary"`, and keys corresponding to the parameters of the `__init__`\n        method (note that for this to work, those parameters must have type annotations).\n\n        If you want to have the instantiation from a config file call a method other than the\n        constructor, either because you have several different construction paths that could be\n        taken for the same object (as we do in `Vocabulary`) or because you have logic you want to\n        happen before you get to the constructor (as we do in `Embedding`), you can register a\n        specific `@classmethod` as the constructor to use, like this:\n\n        ```python\n        @Vocabulary.register("my-vocabulary-from-instances", constructor="from_instances")\n        @Vocabulary.register("my-vocabulary-from-files", constructor="from_files")\n        class MyVocabulary(Vocabulary):\n            def __init__(self, some_params):\n                ...\n\n            @classmethod\n            def from_instances(cls, some_other_params) -> MyVocabulary:\n                ...  # construct some_params from instances\n                return cls(some_params)\n\n            @classmethod\n            def from_files(cls, still_other_params) -> MyVocabulary:\n                ...  # construct some_params from files\n                return cls(some_params)\n        ```\n        '
        registry = Registrable._registry[cls]

        def add_subclass_to_registry(subclass: Type[_T]) -> Type[_T]:
            if False:
                while True:
                    i = 10
            if name in registry:
                if exist_ok:
                    message = f'{name} has already been registered as {registry[name][0].__name__}, but exist_ok=True, so overwriting with {cls.__name__}'
                    logger.info(message)
                else:
                    message = f'Cannot register {name} as {cls.__name__}; name already in use for {registry[name][0].__name__}'
                    raise ConfigurationError(message)
            registry[name] = (subclass, constructor)
            return subclass
        return add_subclass_to_registry

    @classmethod
    def by_name(cls: Type[_RegistrableT], name: str) -> Callable[..., _RegistrableT]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns a callable function that constructs an argument of the registered class.  Because\n        you can register particular functions as constructors for specific names, this isn't\n        necessarily the `__init__` method of some class.\n        "
        logger.debug(f'instantiating registered subclass {name} of {cls}')
        (subclass, constructor) = cls.resolve_class_name(name)
        if not constructor:
            return cast(Type[_RegistrableT], subclass)
        else:
            return cast(Callable[..., _RegistrableT], getattr(subclass, constructor))

    @classmethod
    def resolve_class_name(cls: Type[_RegistrableT], name: str) -> Tuple[Type[_RegistrableT], Optional[str]]:
        if False:
            print('Hello World!')
        '\n        Returns the subclass that corresponds to the given `name`, along with the name of the\n        method that was registered as a constructor for that `name`, if any.\n\n        This method also allows `name` to be a fully-specified module name, instead of a name that\n        was already added to the `Registry`.  In that case, you cannot use a separate function as\n        a constructor (as you need to call `cls.register()` in order to tell us what separate\n        function to use).\n        '
        if name in Registrable._registry[cls]:
            (subclass, constructor) = Registrable._registry[cls][name]
            return (subclass, constructor)
        elif '.' in name:
            parts = name.split('.')
            submodule = '.'.join(parts[:-1])
            class_name = parts[-1]
            try:
                module = importlib.import_module(submodule)
            except ModuleNotFoundError:
                raise ConfigurationError(f'tried to interpret {name} as a path to a class but unable to import module {submodule}')
            try:
                subclass = getattr(module, class_name)
                constructor = None
                return (subclass, constructor)
            except AttributeError:
                raise ConfigurationError(f'tried to interpret {name} as a path to a class but unable to find class {class_name} in {submodule}')
        else:
            available = cls.list_available()
            suggestion = _get_suggestion(name, available)
            raise ConfigurationError(f"'{name}' is not a registered name for '{cls.__name__}'" + ('. ' if not suggestion else f", did you mean '{suggestion}'? ") + 'If your registered class comes from custom code, you\'ll need to import the corresponding modules. If you\'re using AllenNLP from the command-line, this is done by using the \'--include-package\' flag, or by specifying your imports in a \'.allennlp_plugins\' file. Alternatively, you can specify your choices using fully-qualified paths, e.g. {"model": "my_module.models.MyModel"} in which case they will be automatically imported correctly.')

    @classmethod
    def list_available(cls) -> List[str]:
        if False:
            i = 10
            return i + 15
        'List default first if it exists'
        keys = list(Registrable._registry[cls].keys())
        default = cls.default_implementation
        if default is None:
            return keys
        elif default not in keys:
            raise ConfigurationError(f'Default implementation {default} is not registered')
        else:
            return [default] + [k for k in keys if k != default]

    def _to_params(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Default behavior to get a params dictionary from a registrable class\n        that does NOT have a _to_params implementation. It is NOT recommended to\n         use this method. Rather this method is a minial implementation that\n         exists so that calling `_to_params` does not break.\n\n        # Returns\n\n        parameter_dict: `Dict[str, Any]`\n            A minimal parameter dictionary for a given registrable class. Will\n            get the registered name and return that as well as any positional\n            arguments it can find the value of.\n\n        '
        logger.warning(f"'{self.__class__.__name__}' does not implement '_to_params`. Will use Registrable's `_to_params`.")
        mro = inspect.getmro(self.__class__)[1:]
        registered_name = None
        for parent in mro:
            try:
                registered_classes = self._registry[parent]
            except KeyError:
                continue
            for (name, registered_value) in registered_classes.items():
                (registered_class, _) = registered_value
                if registered_class == self.__class__:
                    registered_name = name
                    break
            if registered_name is not None:
                break
        if registered_name is None:
            raise KeyError(f"'{self.__class__.__name__}' is not registered")
        parameter_dict = {'type': registered_name}
        for parameter in inspect.signature(self.__class__).parameters.values():
            if parameter.default != inspect.Parameter.empty:
                logger.debug(f'Skipping parameter {parameter.name}')
                continue
            if hasattr(self, parameter.name):
                parameter_dict[parameter.name] = getattr(self, parameter.name)
            elif hasattr(self, f'_{parameter.name}'):
                parameter_dict[parameter.name] = getattr(self, f'_{parameter.name}')
            else:
                logger.warning(f'Could not find a value for positional argument {parameter.name}')
                continue
        return parameter_dict

def _get_suggestion(name: str, available: List[str]) -> Optional[str]:
    if False:
        while True:
            i = 10
    for (ch, repl_ch) in (('_', '-'), ('-', '_')):
        suggestion = name.replace(ch, repl_ch)
        if suggestion in available:
            return suggestion
    from nltk.metrics.distance import edit_distance
    for suggestion in available:
        if edit_distance(name, suggestion, transpositions=True) == 1:
            return suggestion
    return None