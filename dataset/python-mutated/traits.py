"""Engine's traits are fetched from the origin engines and stored in a JSON file
in the *data folder*.  Most often traits are languages and region codes and
their mapping from SearXNG's representation to the representation in the origin
search engine.  For new traits new properties can be added to the class
:py:class:`EngineTraits`.

To load traits from the persistence :py:obj:`EngineTraitsMap.from_data` can be
used.
"""
from __future__ import annotations
import json
import dataclasses
import types
from typing import Dict, Iterable, Union, Callable, Optional, TYPE_CHECKING
from typing_extensions import Literal, Self
from searx import locales
from searx.data import data_dir, ENGINE_TRAITS
if TYPE_CHECKING:
    from . import Engine

class EngineTraitsEncoder(json.JSONEncoder):
    """Encodes :class:`EngineTraits` to a serializable object, see
    :class:`json.JSONEncoder`."""

    def default(self, o):
        if False:
            for i in range(10):
                print('nop')
        'Return dictionary of a :class:`EngineTraits` object.'
        if isinstance(o, EngineTraits):
            return o.__dict__
        return super().default(o)

@dataclasses.dataclass
class EngineTraits:
    """The class is intended to be instantiated for each engine."""
    regions: Dict[str, str] = dataclasses.field(default_factory=dict)
    "Maps SearXNG's internal representation of a region to the one of the engine.\n\n    SearXNG's internal representation can be parsed by babel and the value is\n    send to the engine:\n\n    .. code:: python\n\n       regions ={\n           'fr-BE' : <engine's region name>,\n       }\n\n       for key, egnine_region regions.items():\n          searxng_region = babel.Locale.parse(key, sep='-')\n          ...\n    "
    languages: Dict[str, str] = dataclasses.field(default_factory=dict)
    "Maps SearXNG's internal representation of a language to the one of the engine.\n\n    SearXNG's internal representation can be parsed by babel and the value is\n    send to the engine:\n\n    .. code:: python\n\n       languages = {\n           'ca' : <engine's language name>,\n       }\n\n       for key, egnine_lang in languages.items():\n          searxng_lang = babel.Locale.parse(key)\n          ...\n    "
    all_locale: Optional[str] = None
    'To which locale value SearXNG\'s ``all`` language is mapped (shown a "Default\n    language").\n    '
    data_type: Literal['traits_v1'] = 'traits_v1'
    "Data type, default is 'traits_v1'.\n    "
    custom: Dict[str, Union[Dict[str, Dict], Iterable[str]]] = dataclasses.field(default_factory=dict)
    "A place to store engine's custom traits, not related to the SearXNG core.\n    "

    def get_language(self, searxng_locale: str, default=None):
        if False:
            while True:
                i = 10
        "Return engine's language string that *best fits* to SearXNG's locale.\n\n        :param searxng_locale: SearXNG's internal representation of locale\n          selected by the user.\n\n        :param default: engine's default language\n\n        The *best fits* rules are implemented in\n        :py:obj:`searx.locales.get_engine_locale`.  Except for the special value ``all``\n        which is determined from :py:obj:`EngineTraits.all_locale`.\n        "
        if searxng_locale == 'all' and self.all_locale is not None:
            return self.all_locale
        return locales.get_engine_locale(searxng_locale, self.languages, default=default)

    def get_region(self, searxng_locale: str, default=None):
        if False:
            return 10
        "Return engine's region string that best fits to SearXNG's locale.\n\n        :param searxng_locale: SearXNG's internal representation of locale\n          selected by the user.\n\n        :param default: engine's default region\n\n        The *best fits* rules are implemented in\n        :py:obj:`searx.locales.get_engine_locale`.  Except for the special value ``all``\n        which is determined from :py:obj:`EngineTraits.all_locale`.\n        "
        if searxng_locale == 'all' and self.all_locale is not None:
            return self.all_locale
        return locales.get_engine_locale(searxng_locale, self.regions, default=default)

    def is_locale_supported(self, searxng_locale: str) -> bool:
        if False:
            print('Hello World!')
        "A *locale* (SearXNG's internal representation) is considered to be\n        supported by the engine if the *region* or the *language* is supported\n        by the engine.\n\n        For verification the functions :py:func:`EngineTraits.get_region` and\n        :py:func:`EngineTraits.get_language` are used.\n        "
        if self.data_type == 'traits_v1':
            return bool(self.get_region(searxng_locale) or self.get_language(searxng_locale))
        raise TypeError('engine traits of type %s is unknown' % self.data_type)

    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        'Create a copy of the dataclass object.'
        return EngineTraits(**dataclasses.asdict(self))

    @classmethod
    def fetch_traits(cls, engine: Engine) -> Union[Self, None]:
        if False:
            return 10
        'Call a function ``fetch_traits(engine_traits)`` from engines namespace to fetch\n        and set properties from the origin engine in the object ``engine_traits``.  If\n        function does not exists, ``None`` is returned.\n        '
        fetch_traits = getattr(engine, 'fetch_traits', None)
        engine_traits = None
        if fetch_traits:
            engine_traits = cls()
            fetch_traits(engine_traits)
        return engine_traits

    def set_traits(self, engine: Engine):
        if False:
            for i in range(10):
                print('nop')
        'Set traits from self object in a :py:obj:`.Engine` namespace.\n\n        :param engine: engine instance build by :py:func:`searx.engines.load_engine`\n        '
        if self.data_type == 'traits_v1':
            self._set_traits_v1(engine)
        else:
            raise TypeError('engine traits of type %s is unknown' % self.data_type)

    def _set_traits_v1(self, engine: Engine):
        if False:
            i = 10
            return i + 15
        traits = self.copy()
        _msg = "settings.yml - engine: '%s' / %s: '%s' not supported"
        languages = traits.languages
        if hasattr(engine, 'language'):
            if engine.language not in languages:
                raise ValueError(_msg % (engine.name, 'language', engine.language))
            traits.languages = {engine.language: languages[engine.language]}
        regions = traits.regions
        if hasattr(engine, 'region'):
            if engine.region not in regions:
                raise ValueError(_msg % (engine.name, 'region', engine.region))
            traits.regions = {engine.region: regions[engine.region]}
        engine.language_support = bool(traits.languages or traits.regions)
        engine.traits = traits

class EngineTraitsMap(Dict[str, EngineTraits]):
    """A python dictionary to map :class:`EngineTraits` by engine name."""
    ENGINE_TRAITS_FILE = (data_dir / 'engine_traits.json').resolve()
    'File with persistence of the :py:obj:`EngineTraitsMap`.'

    def save_data(self):
        if False:
            for i in range(10):
                print('nop')
        'Store EngineTraitsMap in in file :py:obj:`self.ENGINE_TRAITS_FILE`'
        with open(self.ENGINE_TRAITS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self, f, indent=2, sort_keys=True, cls=EngineTraitsEncoder)

    @classmethod
    def from_data(cls) -> Self:
        if False:
            i = 10
            return i + 15
        'Instantiate :class:`EngineTraitsMap` object from :py:obj:`ENGINE_TRAITS`'
        obj = cls()
        for (k, v) in ENGINE_TRAITS.items():
            obj[k] = EngineTraits(**v)
        return obj

    @classmethod
    def fetch_traits(cls, log: Callable) -> Self:
        if False:
            while True:
                i = 10
        from searx import engines
        names = list(engines.engines)
        names.sort()
        obj = cls()
        for engine_name in names:
            engine = engines.engines[engine_name]
            traits = EngineTraits.fetch_traits(engine)
            if traits is not None:
                log('%-20s: SearXNG languages --> %s ' % (engine_name, len(traits.languages)))
                log('%-20s: SearXNG regions   --> %s' % (engine_name, len(traits.regions)))
                obj[engine_name] = traits
        return obj

    def set_traits(self, engine: Engine | types.ModuleType):
        if False:
            i = 10
            return i + 15
        'Set traits in a :py:obj:`Engine` namespace.\n\n        :param engine: engine instance build by :py:func:`searx.engines.load_engine`\n        '
        engine_traits = EngineTraits(data_type='traits_v1')
        if engine.name in self.keys():
            engine_traits = self[engine.name]
        elif engine.engine in self.keys():
            engine_traits = self[engine.engine]
        engine_traits.set_traits(engine)