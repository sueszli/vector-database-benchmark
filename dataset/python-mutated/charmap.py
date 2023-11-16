import codecs
import gzip
import json
import os
import sys
import tempfile
import unicodedata
from functools import lru_cache
from typing import Dict, Tuple
from hypothesis.configuration import storage_directory
from hypothesis.errors import InvalidArgument
from hypothesis.internal.intervalsets import IntervalSet
intervals = Tuple[Tuple[int, int], ...]
cache_type = Dict[Tuple[Tuple[str, ...], int, int, intervals], IntervalSet]

def charmap_file(fname='charmap'):
    if False:
        return 10
    return storage_directory('unicode_data', unicodedata.unidata_version, f'{fname}.json.gz')
_charmap = None

def charmap():
    if False:
        for i in range(10):
            print('nop')
    "Return a dict that maps a Unicode category, to a tuple of 2-tuples\n    covering the codepoint intervals for characters in that category.\n\n    >>> charmap()['Co']\n    ((57344, 63743), (983040, 1048573), (1048576, 1114109))\n    "
    global _charmap
    if _charmap is None:
        f = charmap_file()
        try:
            with gzip.GzipFile(f, 'rb') as i:
                tmp_charmap = dict(json.load(i))
        except Exception:
            category = unicodedata.category
            tmp_charmap = {}
            last_cat = category(chr(0))
            last_start = 0
            for i in range(1, sys.maxunicode + 1):
                cat = category(chr(i))
                if cat != last_cat:
                    tmp_charmap.setdefault(last_cat, []).append([last_start, i - 1])
                    (last_cat, last_start) = (cat, i)
            tmp_charmap.setdefault(last_cat, []).append([last_start, sys.maxunicode])
            try:
                tmpdir = storage_directory('tmp')
                tmpdir.mkdir(exist_ok=True, parents=True)
                (fd, tmpfile) = tempfile.mkstemp(dir=tmpdir)
                os.close(fd)
                with gzip.GzipFile(tmpfile, 'wb', mtime=1) as o:
                    result = json.dumps(sorted(tmp_charmap.items()))
                    o.write(result.encode())
                os.renames(tmpfile, f)
            except Exception:
                pass
        _charmap = {k: tuple((tuple(pair) for pair in pairs)) for (k, pairs) in tmp_charmap.items()}
        for vs in _charmap.values():
            ints = list(sum(vs, ()))
            assert all((isinstance(x, int) for x in ints))
            assert ints == sorted(ints)
            assert all((len(tup) == 2 for tup in vs))
    assert _charmap is not None
    return _charmap

@lru_cache(maxsize=None)
def intervals_from_codec(codec_name: str) -> IntervalSet:
    if False:
        i = 10
        return i + 15
    'Return an IntervalSet of characters which are part of this codec.'
    assert codec_name == codecs.lookup(codec_name).name
    fname = charmap_file(f'codec-{codec_name}')
    try:
        with gzip.GzipFile(fname) as gzf:
            encodable_intervals = json.load(gzf)
    except Exception:
        encodable_intervals = []
        for i in range(sys.maxunicode + 1):
            try:
                chr(i).encode(codec_name)
            except Exception:
                pass
            else:
                encodable_intervals.append((i, i))
    res = IntervalSet(encodable_intervals)
    res = res.union(res)
    try:
        tmpdir = storage_directory('tmp')
        tmpdir.mkdir(exist_ok=True, parents=True)
        (fd, tmpfile) = tempfile.mkstemp(dir=tmpdir)
        os.close(fd)
        with gzip.GzipFile(tmpfile, 'wb', mtime=1) as o:
            o.write(json.dumps(res.intervals).encode())
        os.renames(tmpfile, fname)
    except Exception:
        pass
    return res
_categories = None

def categories():
    if False:
        for i in range(10):
            print('nop')
    "Return a tuple of Unicode categories in a normalised order.\n\n    >>> categories() # doctest: +ELLIPSIS\n    ('Zl', 'Zp', 'Co', 'Me', 'Pc', ..., 'Cc', 'Cs')\n    "
    global _categories
    if _categories is None:
        cm = charmap()
        _categories = sorted(cm.keys(), key=lambda c: len(cm[c]))
        _categories.remove('Cc')
        _categories.remove('Cs')
        _categories.append('Cc')
        _categories.append('Cs')
    return tuple(_categories)

def as_general_categories(cats, name='cats'):
    if False:
        while True:
            i = 10
    "Return a tuple of Unicode categories in a normalised order.\n\n    This function expands one-letter designations of a major class to include\n    all subclasses:\n\n    >>> as_general_categories(['N'])\n    ('Nd', 'Nl', 'No')\n\n    See section 4.5 of the Unicode standard for more on classes:\n    https://www.unicode.org/versions/Unicode10.0.0/ch04.pdf\n\n    If the collection ``cats`` includes any elements that do not represent a\n    major class or a class with subclass, a deprecation warning is raised.\n    "
    if cats is None:
        return None
    major_classes = ('L', 'M', 'N', 'P', 'S', 'Z', 'C')
    cs = categories()
    out = set(cats)
    for c in cats:
        if c in major_classes:
            out.discard(c)
            out.update((x for x in cs if x.startswith(c)))
        elif c not in cs:
            raise InvalidArgument(f'In {name}={cats!r}, {c!r} is not a valid Unicode category.')
    return tuple((c for c in cs if c in out))
category_index_cache = {(): ()}

def _category_key(cats):
    if False:
        print('Hello World!')
    "Return a normalised tuple of all Unicode categories that are in\n    `include`, but not in `exclude`.\n\n    If include is None then default to including all categories.\n    Any item in include that is not a unicode character will be excluded.\n\n    >>> _category_key(exclude=['So'], include=['Lu', 'Me', 'Cs', 'So'])\n    ('Me', 'Lu', 'Cs')\n    "
    cs = categories()
    if cats is None:
        cats = set(cs)
    return tuple((c for c in cs if c in cats))

def _query_for_key(key):
    if False:
        return 10
    "Return a tuple of codepoint intervals covering characters that match one\n    or more categories in the tuple of categories `key`.\n\n    >>> _query_for_key(categories())\n    ((0, 1114111),)\n    >>> _query_for_key(('Zl', 'Zp', 'Co'))\n    ((8232, 8233), (57344, 63743), (983040, 1048573), (1048576, 1114109))\n    "
    try:
        return category_index_cache[key]
    except KeyError:
        pass
    assert key
    if set(key) == set(categories()):
        result = IntervalSet([(0, sys.maxunicode)])
    else:
        result = IntervalSet(_query_for_key(key[:-1])).union(IntervalSet(charmap()[key[-1]]))
    assert isinstance(result, IntervalSet)
    category_index_cache[key] = result.intervals
    return result.intervals
limited_category_index_cache: cache_type = {}

def query(*, categories=None, min_codepoint=None, max_codepoint=None, include_characters='', exclude_characters=''):
    if False:
        for i in range(10):
            print('nop')
    "Return a tuple of intervals covering the codepoints for all characters\n    that meet the criteria.\n\n    >>> query()\n    ((0, 1114111),)\n    >>> query(min_codepoint=0, max_codepoint=128)\n    ((0, 128),)\n    >>> query(min_codepoint=0, max_codepoint=128, categories=['Lu'])\n    ((65, 90),)\n    >>> query(min_codepoint=0, max_codepoint=128, categories=['Lu'],\n    ...       include_characters='☃')\n    ((65, 90), (9731, 9731))\n    "
    if min_codepoint is None:
        min_codepoint = 0
    if max_codepoint is None:
        max_codepoint = sys.maxunicode
    catkey = _category_key(categories)
    character_intervals = IntervalSet.from_string(include_characters or '')
    exclude_intervals = IntervalSet.from_string(exclude_characters or '')
    qkey = (catkey, min_codepoint, max_codepoint, character_intervals.intervals, exclude_intervals.intervals)
    try:
        return limited_category_index_cache[qkey]
    except KeyError:
        pass
    base = _query_for_key(catkey)
    result = []
    for (u, v) in base:
        if v >= min_codepoint and u <= max_codepoint:
            result.append((max(u, min_codepoint), min(v, max_codepoint)))
    result = (IntervalSet(result) | character_intervals) - exclude_intervals
    limited_category_index_cache[qkey] = result
    return result