"""Parsing of strings into DBCore queries.
"""
import itertools
import re
from typing import Collection, Dict, List, Optional, Sequence, Tuple, Type
from . import Model, query
from .query import Query, Sort
PARSE_QUERY_PART_REGEX = re.compile('(-|\\^)?(?:(\\S+?)(?<!\\\\):)?(.*)', re.I)

def parse_query_part(part: str, query_classes: Dict={}, prefixes: Dict={}, default_class: Type[query.SubstringQuery]=query.SubstringQuery) -> Tuple[Optional[str], str, Type[query.Query], bool]:
    if False:
        while True:
            i = 10
    "Parse a single *query part*, which is a chunk of a complete query\n    string representing a single criterion.\n\n    A query part is a string consisting of:\n    - A *pattern*: the value to look for.\n    - Optionally, a *field name* preceding the pattern, separated by a\n      colon. So in `foo:bar`, `foo` is the field name and `bar` is the\n      pattern.\n    - Optionally, a *query prefix* just before the pattern (and after the\n      optional colon) indicating the type of query that should be used. For\n      example, in `~foo`, `~` might be a prefix. (The set of prefixes to\n      look for is given in the `prefixes` parameter.)\n    - Optionally, a negation indicator, `-` or `^`, at the very beginning.\n\n    Both prefixes and the separating `:` character may be escaped with a\n    backslash to avoid their normal meaning.\n\n    The function returns a tuple consisting of:\n    - The field name: a string or None if it's not present.\n    - The pattern, a string.\n    - The query class to use, which inherits from the base\n      :class:`Query` type.\n    - A negation flag, a bool.\n\n    The three optional parameters determine which query class is used (i.e.,\n    the third return value). They are:\n    - `query_classes`, which maps field names to query classes. These\n      are used when no explicit prefix is present.\n    - `prefixes`, which maps prefix strings to query classes.\n    - `default_class`, the fallback when neither the field nor a prefix\n      indicates a query class.\n\n    So the precedence for determining which query class to return is:\n    prefix, followed by field, and finally the default.\n\n    For example, assuming the `:` prefix is used for `RegexpQuery`:\n    - `'stapler'` -> `(None, 'stapler', SubstringQuery, False)`\n    - `'color:red'` -> `('color', 'red', SubstringQuery, False)`\n    - `':^Quiet'` -> `(None, '^Quiet', RegexpQuery, False)`, because\n      the `^` follows the `:`\n    - `'color::b..e'` -> `('color', 'b..e', RegexpQuery, False)`\n    - `'-color:red'` -> `('color', 'red', SubstringQuery, True)`\n    "
    part = part.strip()
    match = PARSE_QUERY_PART_REGEX.match(part)
    assert match
    negate = bool(match.group(1))
    key = match.group(2)
    term = match.group(3).replace('\\:', ':')
    for (pre, query_class) in prefixes.items():
        if term.startswith(pre):
            return (key, term[len(pre):], query_class, negate)
    query_class = query_classes.get(key, default_class)
    return (key, term, query_class, negate)

def construct_query_part(model_cls: Type[Model], prefixes: Dict, query_part: str) -> query.Query:
    if False:
        return 10
    "Parse a *query part* string and return a :class:`Query` object.\n\n    :param model_cls: The :class:`Model` class that this is a query for.\n      This is used to determine the appropriate query types for the\n      model's fields.\n    :param prefixes: A map from prefix strings to :class:`Query` types.\n    :param query_part: The string to parse.\n\n    See the documentation for `parse_query_part` for more information on\n    query part syntax.\n    "
    if not query_part:
        return query.TrueQuery()
    out_query: query.Query
    query_classes: Dict[str, Type[Query]] = {}
    for (k, t) in itertools.chain(model_cls._fields.items(), model_cls._types.items()):
        query_classes[k] = t.query
    query_classes.update(model_cls._queries)
    (key, pattern, query_class, negate) = parse_query_part(query_part, query_classes, prefixes)
    if key is None:
        if issubclass(query_class, query.FieldQuery):
            out_query = query.AnyFieldQuery(pattern, model_cls._search_fields, query_class)
        elif issubclass(query_class, query.NamedQuery):
            out_query = query_class(pattern)
        else:
            assert False, 'Unexpected query type'
    elif issubclass(query_class, query.FieldQuery):
        key = key.lower()
        out_query = query_class(key.lower(), pattern, key in model_cls._fields)
    elif issubclass(query_class, query.NamedQuery):
        out_query = query_class(pattern)
    else:
        assert False, 'Unexpected query type'
    if negate:
        return query.NotQuery(out_query)
    else:
        return out_query

def query_from_strings(query_cls: Type[query.CollectionQuery], model_cls: Type[Model], prefixes: Dict, query_parts: Collection[str]) -> query.Query:
    if False:
        while True:
            i = 10
    'Creates a collection query of type `query_cls` from a list of\n    strings in the format used by parse_query_part. `model_cls`\n    determines how queries are constructed from strings.\n    '
    subqueries = []
    for part in query_parts:
        subqueries.append(construct_query_part(model_cls, prefixes, part))
    if not subqueries:
        subqueries = [query.TrueQuery()]
    return query_cls(subqueries)

def construct_sort_part(model_cls: Type[Model], part: str, case_insensitive: bool=True) -> Sort:
    if False:
        i = 10
        return i + 15
    'Create a `Sort` from a single string criterion.\n\n    `model_cls` is the `Model` being queried. `part` is a single string\n    ending in ``+`` or ``-`` indicating the sort. `case_insensitive`\n    indicates whether or not the sort should be performed in a case\n    sensitive manner.\n    '
    assert part, 'part must be a field name and + or -'
    field = part[:-1]
    assert field, 'field is missing'
    direction = part[-1]
    assert direction in ('+', '-'), 'part must end with + or -'
    is_ascending = direction == '+'
    if field in model_cls._sorts:
        sort = model_cls._sorts[field](model_cls, is_ascending, case_insensitive)
    elif field in model_cls._fields:
        sort = query.FixedFieldSort(field, is_ascending, case_insensitive)
    else:
        sort = query.SlowFieldSort(field, is_ascending, case_insensitive)
    return sort

def sort_from_strings(model_cls: Type[Model], sort_parts: Sequence[str], case_insensitive: bool=True) -> Sort:
    if False:
        while True:
            i = 10
    'Create a `Sort` from a list of sort criteria (strings).'
    if not sort_parts:
        return query.NullSort()
    elif len(sort_parts) == 1:
        return construct_sort_part(model_cls, sort_parts[0], case_insensitive)
    else:
        sort = query.MultipleSort()
        for part in sort_parts:
            sort.add_sort(construct_sort_part(model_cls, part, case_insensitive))
        return sort

def parse_sorted_query(model_cls: Type[Model], parts: List[str], prefixes: Dict={}, case_insensitive: bool=True) -> Tuple[query.Query, Sort]:
    if False:
        i = 10
        return i + 15
    'Given a list of strings, create the `Query` and `Sort` that they\n    represent.\n    '
    query_parts = []
    sort_parts = []
    subquery_parts = []
    for part in parts + [',']:
        if part.endswith(','):
            last_subquery_part = part[:-1]
            if last_subquery_part:
                subquery_parts.append(last_subquery_part)
            query_parts.append(query_from_strings(query.AndQuery, model_cls, prefixes, subquery_parts))
            del subquery_parts[:]
        elif part.endswith(('+', '-')) and ':' not in part and (len(part) > 1):
            sort_parts.append(part)
        else:
            subquery_parts.append(part)
    q = query.OrQuery(query_parts) if len(query_parts) > 1 else query_parts[0]
    s = sort_from_strings(model_cls, sort_parts, case_insensitive)
    return (q, s)