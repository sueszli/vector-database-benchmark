"""Index entries adapters for sphinx.environment."""
from __future__ import annotations
import re
import unicodedata
from itertools import groupby
from typing import TYPE_CHECKING, Any, Literal
from sphinx.errors import NoUri
from sphinx.locale import _, __
from sphinx.util import logging
from sphinx.util.index_entries import _split_into
if TYPE_CHECKING:
    from sphinx.builders import Builder
    from sphinx.environment import BuildEnvironment
logger = logging.getLogger(__name__)

class IndexEntries:

    def __init__(self, env: BuildEnvironment) -> None:
        if False:
            return 10
        self.env = env
        self.builder: Builder

    def create_index(self, builder: Builder, group_entries: bool=True, _fixre: re.Pattern=re.compile('(.*) ([(][^()]*[)])')) -> list[tuple[str, list[tuple[str, Any]]]]:
        if False:
            i = 10
            return i + 15
        'Create the real index from the collected index entries.'
        new: dict[str, list] = {}
        rel_uri: str | Literal[False]
        index_domain = self.env.domains['index']
        for (docname, entries) in index_domain.entries.items():
            try:
                rel_uri = builder.get_relative_uri('genindex', docname)
            except NoUri:
                rel_uri = False
            for (entry_type, value, target_id, main, category_key) in entries:
                uri = rel_uri is not False and f'{rel_uri}#{target_id}'
                try:
                    if entry_type == 'single':
                        try:
                            (entry, sub_entry) = _split_into(2, 'single', value)
                        except ValueError:
                            (entry,) = _split_into(1, 'single', value)
                            sub_entry = ''
                        _add_entry(entry, sub_entry, main, dic=new, link=uri, key=category_key)
                    elif entry_type == 'pair':
                        (first, second) = _split_into(2, 'pair', value)
                        _add_entry(first, second, main, dic=new, link=uri, key=category_key)
                        _add_entry(second, first, main, dic=new, link=uri, key=category_key)
                    elif entry_type == 'triple':
                        (first, second, third) = _split_into(3, 'triple', value)
                        _add_entry(first, second + ' ' + third, main, dic=new, link=uri, key=category_key)
                        _add_entry(second, third + ', ' + first, main, dic=new, link=uri, key=category_key)
                        _add_entry(third, first + ' ' + second, main, dic=new, link=uri, key=category_key)
                    elif entry_type == 'see':
                        (first, second) = _split_into(2, 'see', value)
                        _add_entry(first, _('see %s') % second, None, dic=new, link=False, key=category_key)
                    elif entry_type == 'seealso':
                        (first, second) = _split_into(2, 'see', value)
                        _add_entry(first, _('see also %s') % second, None, dic=new, link=False, key=category_key)
                    else:
                        logger.warning(__('unknown index entry type %r'), entry_type, location=docname)
                except ValueError as err:
                    logger.warning(str(err), location=docname)
        for (targets, sub_items, _category_key) in new.values():
            targets.sort(key=_key_func_0)
            for (sub_targets, _0, _sub_category_key) in sub_items.values():
                sub_targets.sort(key=_key_func_0)
        new_list = sorted(new.items(), key=_key_func_1)
        if group_entries:
            old_key = ''
            old_sub_items: dict[str, list] = {}
            i = 0
            while i < len(new_list):
                (key, (targets, sub_items, category_key)) = new_list[i]
                if not sub_items:
                    m = _fixre.match(key)
                    if m:
                        if old_key == m.group(1):
                            old_sub_items.setdefault(m.group(2), [[], {}, category_key])[0].extend(targets)
                            del new_list[i]
                            continue
                        old_key = m.group(1)
                    else:
                        old_key = key
                old_sub_items = sub_items
                i += 1
        return [(key_, list(group)) for (key_, group) in groupby(new_list, _key_func_3)]

def _add_entry(word: str, subword: str, main: str | None, *, dic: dict[str, list], link: str | Literal[False], key: str | None) -> None:
    if False:
        for i in range(10):
            print('nop')
    entry = dic.setdefault(word, [[], {}, key])
    if subword:
        entry = entry[1].setdefault(subword, [[], {}, key])
    if link:
        entry[0].append((main, link))

def _key_func_0(entry: tuple[str, str]) -> tuple[bool, str]:
    if False:
        return 10
    'sort the index entries for same keyword.'
    (main, uri) = entry
    return (not main, uri)

def _key_func_1(entry: tuple[str, list]) -> tuple[tuple[int, str], str]:
    if False:
        for i in range(10):
            print('nop')
    'Sort the index entries'
    (key, (_targets, _sub_items, category_key)) = entry
    if category_key:
        key = category_key
    lc_key = unicodedata.normalize('NFD', key.lower())
    if lc_key.startswith('\u200f'):
        lc_key = lc_key[1:]
    if not lc_key[0:1].isalpha() and (not lc_key.startswith('_')):
        group = 0
    else:
        group = 1
    return ((group, lc_key), entry[0])

def _key_func_2(entry: tuple[str, list]) -> str:
    if False:
        for i in range(10):
            print('nop')
    'sort the sub-index entries'
    key = unicodedata.normalize('NFD', entry[0].lower())
    if key.startswith('\u200f'):
        key = key[1:]
    if key[0:1].isalpha() or key.startswith('_'):
        key = chr(127) + key
    return key

def _key_func_3(entry: tuple[str, list]) -> str:
    if False:
        i = 10
        return i + 15
    'Group the entries by letter'
    (key, (targets, sub_items, category_key)) = entry
    entry[1][1] = sorted(((sub_key, sub_targets) for (sub_key, (sub_targets, _0, _sub_category_key)) in sub_items.items()), key=_key_func_2)
    if category_key is not None:
        return category_key
    if key.startswith('\u200f'):
        key = key[1:]
    letter = unicodedata.normalize('NFD', key[0])[0].upper()
    if letter.isalpha() or letter == '_':
        return letter
    return _('Symbols')