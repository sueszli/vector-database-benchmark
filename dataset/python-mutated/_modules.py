from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, NamedTuple, Optional, Set, Type, TypeVar
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from ._fwd import config as Config
T = TypeVar('T')
U = TypeVar('U')
console = Console()

class ParamSpec(NamedTuple):
    """
    Attributes:
        req         Whether this argument is required
        desc        A description of what this argument does
        default     The default value for this argument. Ignored if req == True or configPath is not None
        config_ref  The path to the config that should be the default value
        list        Whether this parameter is in the form of a list, and can therefore be specified more than once
        visible     Whether the user can tweak this via the command line
    """
    req: bool
    desc: str
    default: Optional[Any] = None
    list: bool = False
    config_ref: Optional[List[str]] = None
    visible: bool = True

class ConfigurableModule(ABC):

    @staticmethod
    @abstractmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            print('Hello World!')
        '\n        Returns a dictionary of `argument name: argument specification`\n        '
        pass

    def _checkParams(self):
        if False:
            while True:
                i = 10
        '\n        Fills the given params dict with default values where arguments are not given,\n        using None as the default value for default values\n        '
        params = self._params()
        config = self._config()
        for (key, value) in self.getParams().items():
            if key in params:
                continue
            if value.req:
                raise KeyError(f'Missing required param {key} for {type(self).__name__.lower()}')
            if value.config_ref is not None:
                tmp = getattr(config, value.config_ref[0])
                params[key] = tmp[value.config_ref[1:]] if len(value.config_ref) > 1 else tmp
            elif value.default is not None:
                params[key] = value.default

    def _params(self):
        if False:
            print('Hello World!')
        return self._params_obj

    def _config(self):
        if False:
            return 10
        return self._config_obj

    @abstractmethod
    def __init__(self, config: Config):
        if False:
            for i in range(10):
                print('nop')
        self._config_obj = config
        if self.getParams() is not None:
            self._params_obj = config.params.setdefault(type(self).__name__.lower(), {})
            self._checkParams()

class Targeted(ABC):

    @staticmethod
    @abstractmethod
    def getTarget() -> str:
        if False:
            for i in range(10):
                print('nop')
        'Should return the target that this object attacks/decodes'
        pass

class PolymorphicChecker(ConfigurableModule):

    @abstractmethod
    def check(self, text) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        'Should return some description (or an empty string) on success, otherwise return None'
        pass

    @abstractmethod
    def getExpectedRuntime(self, text) -> float:
        if False:
            print('Hello World!')
        pass

    def __call__(self, *args):
        if False:
            while True:
                i = 10
        return self.check(*args)

    @abstractmethod
    def __init__(self, config: Config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)

class Checker(Generic[T], ConfigurableModule):

    @abstractmethod
    def check(self, text: T) -> Optional[str]:
        if False:
            return 10
        'Should return some description (or an empty string) on success, otherwise return None'
        pass

    @abstractmethod
    def getExpectedRuntime(self, text: T) -> float:
        if False:
            for i in range(10):
                print('nop')
        pass

    def __call__(self, *args):
        if False:
            i = 10
            return i + 15
        return self.check(*args)

    @abstractmethod
    def __init__(self, config: Config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)

    @classmethod
    def convert(cls, expected: Set[type]):
        if False:
            print('Hello World!')

        class PolyWrapperClass(PolymorphicChecker):

            @staticmethod
            def getParams() -> Optional[Dict[str, ParamSpec]]:
                if False:
                    i = 10
                    return i + 15
                return cls.getParams()

            def check(self, text) -> Optional[str]:
                if False:
                    return 10
                'Should return some description (or an empty string) on success, otherwise return None'
                if type(text) not in expected:
                    return None
                else:
                    return self._base.check(text)

            def getExpectedRuntime(self, text) -> float:
                if False:
                    for i in range(10):
                        print('nop')
                if type(text) not in expected:
                    return 0
                else:
                    return self._base.getExpectedRuntime(text)

            def __init__(self, config: Config):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__(config)
                self._base = cls(config)
        PolyWrapperClass.__name__ = cls.__name__
        return PolyWrapperClass

class Decoder(Generic[T], ConfigurableModule, Targeted):
    """Represents the undoing of some encoding into a different (or the same) type"""

    @abstractmethod
    def decode(self, ctext: T) -> Optional[U]:
        if False:
            while True:
                i = 10
        pass

    @staticmethod
    @abstractmethod
    def priority() -> float:
        if False:
            print('Hello World!')
        'What proportion of decodings are this?'
        pass

    def __call__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        return self.decode(*args)

    @abstractmethod
    def __init__(self, config: Config):
        if False:
            return 10
        super().__init__(config)

class DecoderComparer:
    value: Type[Decoder]

    def __le__(self, other: 'DecoderComparer'):
        if False:
            print('Hello World!')
        return self.value.priority() <= other.value.priority()

    def __ge__(self, other: 'DecoderComparer'):
        if False:
            for i in range(10):
                print('nop')
        return self.value.priority() >= other.value.priority()

    def __lt__(self, other: 'DecoderComparer'):
        if False:
            return 10
        return self.value.priority() < other.value.priority() and self != other

    def __gt__(self, other: 'DecoderComparer'):
        if False:
            i = 10
            return i + 15
        return self.value.priority() > other.value.priority() and self != other

    def __init__(self, value: Type[Decoder]):
        if False:
            print('Hello World!')
        self.value = value

    def __repr__(self):
        if False:
            return 10
        return f'<DecoderComparer {self.value}:{self.value.priority()}>'

class CrackResult(NamedTuple):
    value: Any
    key_info: Optional[str] = None
    misc_info: Optional[str] = None

class CrackInfo(NamedTuple):
    success_likelihood: float
    success_runtime: float
    failure_runtime: float

class Cracker(Generic[T], ConfigurableModule, Targeted):

    @abstractmethod
    def getInfo(self, ctext: T) -> CrackInfo:
        if False:
            while True:
                i = 10
        'Should return some informed guesses on resource consumption when run on `ctext`'
        pass

    @abstractmethod
    def attemptCrack(self, ctext: T) -> List[CrackResult]:
        if False:
            while True:
                i = 10
        '\n        This should attempt to crack the cipher `target`, and return a list of candidate solutions\n        '
        pass

    def __call__(self, *args):
        if False:
            print('Hello World!')
        return self.attemptCrack(*args)

    @abstractmethod
    def __init__(self, config: Config):
        if False:
            return 10
        super().__init__(config)

class ResourceLoader(Generic[T], ConfigurableModule):

    @abstractmethod
    def whatResources(self) -> Optional[Set[str]]:
        if False:
            while True:
                i = 10
        '\n        Return a set of the names of instances T you can provide.\n        The names SHOULD be unique amongst ResourceLoaders of the same type\n\n        These names will be exposed as f"{self.__name__}::{name}", use split_resource_name to recover this\n\n        If you cannot reasonably determine what resources you provide, return None instead\n        '
        pass

    @abstractmethod
    def getResource(self, name: str) -> T:
        if False:
            while True:
                i = 10
        '\n        Returns the requested distribution\n\n        The behavior is undefined if `name not in self.what_resources()`\n        '
        pass

    def __call__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        return self.getResource(*args)

    def __getitem__(self, *args):
        if False:
            return 10
        return self.getResource(*args)

    @abstractmethod
    def __init__(self, config: Config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)

class SearchLevel(NamedTuple):
    name: str
    result: CrackResult

    @staticmethod
    def input(ctext: Any):
        if False:
            i = 10
            return i + 15
        return SearchLevel(name='input', result=CrackResult(ctext))

class SearchResult(NamedTuple):
    path: List[SearchLevel]
    check_res: str

class Searcher(ConfigurableModule):
    """A very basic interface for code that plans out how to crack the ciphertext"""

    @abstractmethod
    def search(self, ctext: Any) -> Optional[SearchResult]:
        if False:
            i = 10
            return i + 15
        'Returns the path to the correct ciphertext'
        pass

    @abstractmethod
    def __init__(self, config: Config):
        if False:
            print('Hello World!')
        super().__init__(config)

def pretty_search_results(res: SearchResult, display_intermediate: bool=False) -> str:
    if False:
        i = 10
        return i + 15
    ret: str = ''
    table = Table(show_header=False, box=box.ROUNDED, safe_box=False)
    if len(res.check_res) != 0 and ('The' == res.check_res[0:3] or 'Passed' == res.check_res[0:6]):
        ret += f'{res.check_res}\n'

    def add_one():
        if False:
            while True:
                i = 10
        out = ''
        if i.name == 'utf8':
            out += f'   [#808080]{i.name}[/#808080]\n'
        else:
            out += f'   {i.name}'
        already_broken = False
        if i.result.key_info is not None:
            out += f':\n    Key: {i.result.key_info}\n'
            already_broken = True
        if i.result.misc_info is not None:
            if not already_broken:
                out += ':\n'
            out += f'    Misc: {i.result.misc_info}\n'
            already_broken = True
        if display_intermediate:
            if not already_broken:
                out += ':\n'
            out += f'    Value: "{i.result.value}"\n'
            already_broken = True
        if not already_broken:
            out += '\n'
        return (out, already_broken)
    already_broken = False
    out = ''
    for i in res.path[1:]:
        (output, already_broken) = add_one()
        out += output
    if out:
        if len(out.split('\n')) > 1:
            ret += 'Formats used:\n'
        else:
            ret += 'Format used:\n'
        ret += out
    ret = ret[:-1]
    if already_broken:
        ret += f'\nPlaintext: [bold green]"{escape(res.path[-1].result.value)}"[bold green]'
    else:
        ret += f'Plaintext: [bold green]"{escape(res.path[-1].result.value)}"[bold green]'
    table.add_row(ret)
    return table
Distribution = Dict[str, float]
Translation = Dict[str, str]
WordList = Set[str]