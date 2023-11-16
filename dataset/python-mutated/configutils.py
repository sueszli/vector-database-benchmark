"""Utilities and data structures used by various config code."""
import collections
import itertools
import operator
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Sequence, Set, Union, MutableMapping
from qutebrowser.qt.core import QUrl
from qutebrowser.qt.gui import QFontDatabase
from qutebrowser.qt.widgets import QApplication
from qutebrowser.utils import utils, urlmatch, urlutils, usertypes, qtutils
from qutebrowser.config import configexc
if TYPE_CHECKING:
    from qutebrowser.config import configdata

class ScopedValue:
    """A configuration value which is valid for a UrlPattern.

    Attributes:
        value: The value itself.
        pattern: The UrlPattern for the value, or None for global values.
        hide_userconfig: Hide this customization from config.dump_userconfig().
    """
    id_gen = itertools.count(0)

    def __init__(self, value: Any, pattern: Optional[urlmatch.UrlPattern], hide_userconfig: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.value = value
        self.pattern = pattern
        self.hide_userconfig = hide_userconfig
        self.pattern_id = next(ScopedValue.id_gen)

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return utils.get_repr(self, value=self.value, pattern=self.pattern, hide_userconfig=self.hide_userconfig, pattern_id=self.pattern_id)

class Values:
    """A collection of values for a single setting.

    Currently, we store patterns in two dictionaries for different types of
    lookups. A ordered, pattern keyed map, and an unordered, domain keyed map.

    This means that finding a value based on a pattern is fast, and matching
    url patterns is fast if all domains are unique.

    If there are many patterns under the domain (or subdomain) that is being
    evaluated, or any patterns that cannot have a concrete domain found, this
    will become slow again.

    Attributes:
        opt: The Option being customized.
        _vmap: A mapping of all pattern objects to ScopedValues.
        _domain_map: A mapping from hostnames to all associated ScopedValues.
    """
    _VmapKeyType = Optional[urlmatch.UrlPattern]

    def __init__(self, opt: 'configdata.Option', values: Sequence[ScopedValue]=()) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.opt = opt
        self._vmap: MutableMapping[Values._VmapKeyType, ScopedValue] = collections.OrderedDict()
        self._domain_map: Dict[Optional[str], Set[ScopedValue]] = collections.defaultdict(set)
        for scoped in values:
            self._add_scoped(scoped)

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return utils.get_repr(self, opt=self.opt, values=list(self._vmap.values()), constructor=True)

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        'Get the values as human-readable string.'
        lines = self.dump(include_hidden=True)
        if lines:
            return '\n'.join(lines)
        return '{}: <unchanged>'.format(self.opt.name)

    def dump(self, include_hidden: bool=False) -> Sequence[str]:
        if False:
            print('Hello World!')
        'Dump all customizations for this value.\n\n        Arguments:\n           include_hidden: Also show values with hide_userconfig=True.\n        '
        lines = []
        for scoped in self._vmap.values():
            if scoped.hide_userconfig and (not include_hidden):
                continue
            str_value = self.opt.typ.to_str(scoped.value)
            if scoped.pattern is None:
                lines.append('{} = {}'.format(self.opt.name, str_value))
            else:
                lines.append('{}: {} = {}'.format(scoped.pattern, self.opt.name, str_value))
        return lines

    def __iter__(self) -> Iterator['ScopedValue']:
        if False:
            return 10
        'Yield ScopedValue elements.\n\n        This yields in "normal" order, i.e. global and then first-set settings\n        first.\n        '
        yield from self._vmap.values()

    def __bool__(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Check whether this value is customized.'
        return bool(self._vmap)

    def _check_pattern_support(self, arg: Union[urlmatch.UrlPattern, QUrl, None]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Make sure patterns are supported if one was given.'
        if arg is not None and (not self.opt.supports_pattern):
            raise configexc.NoPatternError(self.opt.name)

    def add(self, value: Any, pattern: urlmatch.UrlPattern=None, *, hide_userconfig: bool=False) -> None:
        if False:
            print('Hello World!')
        'Add a value with the given pattern to the list of values.\n\n        If hide_userconfig is given, the value is hidden from\n        config.dump_userconfig() and thus qute://configdiff.\n        '
        scoped = ScopedValue(value, pattern, hide_userconfig=hide_userconfig)
        self._add_scoped(scoped)

    def _add_scoped(self, scoped: ScopedValue) -> None:
        if False:
            print('Hello World!')
        'Add an existing ScopedValue object.'
        self._check_pattern_support(scoped.pattern)
        self.remove(scoped.pattern)
        self._vmap[scoped.pattern] = scoped
        host = scoped.pattern.host if scoped.pattern else None
        self._domain_map[host].add(scoped)

    def remove(self, pattern: urlmatch.UrlPattern=None) -> bool:
        if False:
            print('Hello World!')
        'Remove the value with the given pattern.\n\n        If a matching pattern was removed, True is returned.\n        If no matching pattern was found, False is returned.\n        '
        self._check_pattern_support(pattern)
        if pattern not in self._vmap:
            return False
        host = pattern.host if pattern else None
        scoped_value = self._vmap[pattern]
        assert host in self._domain_map
        self._domain_map[host].remove(scoped_value)
        del self._vmap[pattern]
        return True

    def clear(self) -> None:
        if False:
            return 10
        'Clear all customization for this value.'
        self._vmap.clear()
        self._domain_map.clear()

    def _get_fallback(self, fallback: bool) -> Any:
        if False:
            while True:
                i = 10
        'Get the fallback global/default value.'
        if None in self._vmap:
            return self._vmap[None].value
        if fallback:
            return self.opt.default
        else:
            return usertypes.UNSET

    def get_for_url(self, url: QUrl=None, *, fallback: bool=True) -> Any:
        if False:
            while True:
                i = 10
        "Get a config value, falling back when needed.\n\n        This first tries to find a value matching the URL (if given).\n        If there's no match:\n          With fallback=True, the global/default setting is returned.\n          With fallback=False, usertypes.UNSET is returned.\n        "
        self._check_pattern_support(url)
        if url is None:
            return self._get_fallback(fallback)
        qtutils.ensure_valid(url)
        candidates: List[ScopedValue] = []
        widened_hosts = urlutils.widened_hostnames(url.host().rstrip('.'))
        for host in itertools.chain(widened_hosts, [None]):
            host_set = self._domain_map.get(host, ())
            for scoped in host_set:
                if scoped.pattern is not None and scoped.pattern.matches(url):
                    candidates.append(scoped)
        if candidates:
            scoped = max(candidates, key=operator.attrgetter('pattern_id'))
            return scoped.value
        if not fallback:
            return usertypes.UNSET
        return self._get_fallback(fallback)

    def get_for_pattern(self, pattern: Optional[urlmatch.UrlPattern], *, fallback: bool=True) -> Any:
        if False:
            while True:
                i = 10
        "Get a value only if it's been overridden for the given pattern.\n\n        This is useful when showing values to the user.\n\n        If there's no match:\n          With fallback=True, the global/default setting is returned.\n          With fallback=False, usertypes.UNSET is returned.\n        "
        self._check_pattern_support(pattern)
        if pattern is not None:
            if pattern in self._vmap:
                return self._vmap[pattern].value
            if not fallback:
                return usertypes.UNSET
        return self._get_fallback(fallback)

class FontFamilies:
    """A list of font family names."""

    def __init__(self, families: Sequence[str]) -> None:
        if False:
            i = 10
            return i + 15
        self._families = families
        self.family = families[0] if families else None

    def __iter__(self) -> Iterator[str]:
        if False:
            for i in range(10):
                print('nop')
        yield from self._families

    def __len__(self) -> int:
        if False:
            print('Hello World!')
        return len(self._families)

    def __repr__(self) -> str:
        if False:
            return 10
        return utils.get_repr(self, families=self._families, constructor=True)

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        return self.to_str()

    def _quoted_families(self) -> Iterator[str]:
        if False:
            i = 10
            return i + 15
        for f in self._families:
            needs_quoting = any((c in f for c in '., '))
            yield ('"{}"'.format(f) if needs_quoting else f)

    def to_str(self, *, quote: bool=True) -> str:
        if False:
            return 10
        families = self._quoted_families() if quote else self._families
        return ', '.join(families)

    @classmethod
    def from_system_default(cls, font_type: QFontDatabase.SystemFont=QFontDatabase.SystemFont.FixedFont) -> 'FontFamilies':
        if False:
            return 10
        'Get a FontFamilies object for the default system font.\n\n        By default, the monospace font is returned, though via the "font_type" argument,\n        other types can be requested as well.\n\n        Note that (at least) three ways of getting the default monospace font\n        exist:\n\n        1) f = QFont()\n           f.setStyleHint(QFont.StyleHint.Monospace)\n           print(f.defaultFamily())\n\n        2) f = QFont()\n           f.setStyleHint(QFont.StyleHint.TypeWriter)\n           print(f.defaultFamily())\n\n        3) f = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)\n           print(f.family())\n\n        They yield different results depending on the OS:\n\n                QFont.StyleHint.Monospace  | QFont.StyleHint.TypeWriter | QFontDatabase\n                -----------------------------------------------------------------------\n        Win:    Courier New                | Courier New                | Courier New\n        Linux:  DejaVu Sans Mono           | DejaVu Sans Mono           | monospace\n        macOS:  Menlo                      | American Typewriter        | Monaco\n\n        Test script: https://p.cmpl.cc/076835c4\n\n        On Linux, it seems like both actually resolve to the same font.\n\n        On macOS, "American Typewriter" looks like it indeed tries to imitate a\n        typewriter, so it\'s not really a suitable UI font.\n\n        Looking at those Wikipedia articles:\n\n        https://en.wikipedia.org/wiki/Monaco_(typeface)\n        https://en.wikipedia.org/wiki/Menlo_(typeface)\n\n        the "right" choice isn\'t really obvious. Thus, let\'s go for the\n        QFontDatabase approach here, since it\'s by far the simplest one.\n        '
        assert QApplication.instance() is not None
        font = QFontDatabase.systemFont(font_type)
        return cls([font.family()])

    @classmethod
    def from_str(cls, family_str: str) -> 'FontFamilies':
        if False:
            i = 10
            return i + 15
        'Parse a CSS-like string of font families.'
        families = []
        for part in family_str.split(','):
            part = part.strip()
            if part.startswith("'") and part.endswith("'") or (part.startswith('"') and part.endswith('"')):
                part = part[1:-1]
            if not part:
                continue
            families.append(part)
        return cls(families)