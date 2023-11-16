from __future__ import annotations
import traceback
import warnings
from collections import defaultdict
from types import ModuleType
from typing import TYPE_CHECKING, DefaultDict, Dict, List, Tuple, Type
from zope.interface import implementer
from scrapy import Request, Spider
from scrapy.interfaces import ISpiderLoader
from scrapy.settings import BaseSettings
from scrapy.utils.misc import walk_modules
from scrapy.utils.spider import iter_spider_classes
if TYPE_CHECKING:
    from typing_extensions import Self

@implementer(ISpiderLoader)
class SpiderLoader:
    """
    SpiderLoader is a class which locates and loads spiders
    in a Scrapy project.
    """

    def __init__(self, settings: BaseSettings):
        if False:
            for i in range(10):
                print('nop')
        self.spider_modules: List[str] = settings.getlist('SPIDER_MODULES')
        self.warn_only: bool = settings.getbool('SPIDER_LOADER_WARN_ONLY')
        self._spiders: Dict[str, Type[Spider]] = {}
        self._found: DefaultDict[str, List[Tuple[str, str]]] = defaultdict(list)
        self._load_all_spiders()

    def _check_name_duplicates(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        dupes = []
        for (name, locations) in self._found.items():
            dupes.extend([f'  {cls} named {name!r} (in {mod})' for (mod, cls) in locations if len(locations) > 1])
        if dupes:
            dupes_string = '\n\n'.join(dupes)
            warnings.warn(f'There are several spiders with the same name:\n\n{dupes_string}\n\n  This can cause unexpected behavior.', category=UserWarning)

    def _load_spiders(self, module: ModuleType) -> None:
        if False:
            for i in range(10):
                print('nop')
        for spcls in iter_spider_classes(module):
            self._found[spcls.name].append((module.__name__, spcls.__name__))
            self._spiders[spcls.name] = spcls

    def _load_all_spiders(self) -> None:
        if False:
            while True:
                i = 10
        for name in self.spider_modules:
            try:
                for module in walk_modules(name):
                    self._load_spiders(module)
            except ImportError:
                if self.warn_only:
                    warnings.warn(f"\n{traceback.format_exc()}Could not load spiders from module '{name}'. See above traceback for details.", category=RuntimeWarning)
                else:
                    raise
        self._check_name_duplicates()

    @classmethod
    def from_settings(cls, settings: BaseSettings) -> Self:
        if False:
            while True:
                i = 10
        return cls(settings)

    def load(self, spider_name: str) -> Type[Spider]:
        if False:
            return 10
        '\n        Return the Spider class for the given spider name. If the spider\n        name is not found, raise a KeyError.\n        '
        try:
            return self._spiders[spider_name]
        except KeyError:
            raise KeyError(f'Spider not found: {spider_name}')

    def find_by_request(self, request: Request) -> List[str]:
        if False:
            while True:
                i = 10
        '\n        Return the list of spider names that can handle the given request.\n        '
        return [name for (name, cls) in self._spiders.items() if cls.handles_request(request)]

    def list(self) -> List[str]:
        if False:
            while True:
                i = 10
        '\n        Return a list with the names of all spiders available in the project.\n        '
        return list(self._spiders.keys())