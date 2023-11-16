from __future__ import annotations
import re
from collections import namedtuple
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import reduce
from typing import Any, List, Mapping, NamedTuple, Sequence, Set, Tuple, Union
from django.utils.functional import cached_property
from parsimonious.exceptions import IncompleteParseError
from parsimonious.expressions import Optional
from parsimonious.grammar import Grammar
from parsimonious.nodes import Node, NodeVisitor
from sentry.exceptions import InvalidSearchQuery
from sentry.search.events.constants import DURATION_UNITS, OPERATOR_NEGATION_MAP, SEARCH_MAP, SEMVER_ALIAS, SEMVER_BUILD_ALIAS, SIZE_UNITS, TAG_KEY_RE, TEAM_KEY_TRANSACTION_ALIAS
from sentry.search.events.fields import FIELD_ALIASES, FUNCTIONS
from sentry.search.events.types import QueryBuilderConfig
from sentry.search.utils import InvalidQuery, parse_datetime_range, parse_datetime_string, parse_datetime_value, parse_duration, parse_numeric_value, parse_percentage, parse_size
from sentry.snuba.dataset import Dataset
from sentry.utils.snuba import is_duration_measurement, is_measurement, is_span_op_breakdown
from sentry.utils.validators import is_event_id, is_span_id
WILDCARD_CHARS = re.compile('(?<!\\\\)(\\\\\\\\)*\\*')
event_search_grammar = Grammar('\nsearch = spaces term*\n\nterm = (boolean_operator / paren_group / filter / free_text) spaces\n\nboolean_operator = or_operator / and_operator\n\nparen_group = open_paren spaces term+ closed_paren\n\nfree_text          = free_text_quoted / free_text_unquoted\nfree_text_unquoted = (!filter !boolean_operator (free_parens / ~r"[^()\\n ]+") spaces)+\nfree_text_quoted   = quoted_value\nfree_parens        = open_paren free_text? closed_paren\n\n# All key:value filter types\nfilter = date_filter\n       / specific_date_filter\n       / rel_date_filter\n       / duration_filter\n       / size_filter\n       / boolean_filter\n       / numeric_in_filter\n       / numeric_filter\n       / aggregate_duration_filter\n       / aggregate_percentage_filter\n       / aggregate_numeric_filter\n       / aggregate_size_filter\n       / aggregate_date_filter\n       / aggregate_rel_date_filter\n       / has_filter\n       / is_filter\n       / text_in_filter\n       / text_filter\n\n# filter for dates\ndate_filter = search_key sep operator iso_8601_date_format\n\n# exact date filter for dates\nspecific_date_filter = search_key sep iso_8601_date_format\n\n# filter for relative dates\nrel_date_filter = search_key sep rel_date_format\n\n# filter for durations\nduration_filter = negation? search_key sep operator? duration_format\n\n# filter for size\nsize_filter = negation? search_key sep operator? size_format\n\n# boolean comparison filter\nboolean_filter = negation? search_key sep boolean_value\n\n# numeric in filter\nnumeric_in_filter = negation? search_key sep numeric_in_list\n\n# numeric comparison filter\nnumeric_filter = negation? search_key sep operator? numeric_value\n\n# aggregate duration filter\naggregate_duration_filter = negation? aggregate_key sep operator? duration_format\n\n# aggregate size filter\naggregate_size_filter = negation? aggregate_key sep operator? size_format\n\n# aggregate percentage filter\naggregate_percentage_filter = negation? aggregate_key sep operator? percentage_format\n\n# aggregate numeric filter\naggregate_numeric_filter = negation? aggregate_key sep operator? numeric_value\n\n# aggregate for dates\naggregate_date_filter = negation? aggregate_key sep operator? iso_8601_date_format\n\n# aggregate for relative dates\naggregate_rel_date_filter = negation? aggregate_key sep operator? rel_date_format\n\n# has filter for not null type checks\nhas_filter = negation? &"has:" search_key sep (search_key / search_value)\n\n# is filter. Specific to issue search\nis_filter  = negation? &"is:" search_key sep search_value\n\n# in filter key:[val1, val2]\ntext_in_filter = negation? text_key sep text_in_list\n\n# standard key:val filter\ntext_filter = negation? text_key sep operator? search_value\n\nkey                    = ~r"[a-zA-Z0-9_.-]+"\nquoted_key             = \'"\' ~r"[a-zA-Z0-9_.:-]+" \'"\'\nexplicit_tag_key       = "tags" open_bracket search_key closed_bracket\naggregate_key          = key open_paren spaces function_args? spaces closed_paren\nfunction_args          = aggregate_param (spaces comma spaces !comma aggregate_param?)*\naggregate_param        = quoted_aggregate_param / raw_aggregate_param\nraw_aggregate_param    = ~r"[^()\\t\\n, \\"]+"\nquoted_aggregate_param = \'"\' (\'\\\\"\' / ~r\'[^\\t\\n\\"]\')* \'"\'\nsearch_key             = key / quoted_key\ntext_key               = explicit_tag_key / search_key\nvalue                  = ~r"[^()\\t\\n ]*"\nquoted_value           = \'"\' (\'\\\\"\' / ~r\'[^"]\')* \'"\'\nin_value               = (&in_value_termination in_value_char)+\ntext_in_value          = quoted_value / in_value\nsearch_value           = quoted_value / value\nnumeric_value          = "-"? numeric ~r"[kmb]"? &(end_value / comma / closed_bracket)\nboolean_value          = ~r"(true|1|false|0)"i &end_value\ntext_in_list           = open_bracket text_in_value (spaces comma spaces !comma text_in_value?)* closed_bracket &end_value\nnumeric_in_list        = open_bracket numeric_value (spaces comma spaces !comma numeric_value?)* closed_bracket &end_value\n\n# See: https://stackoverflow.com/a/39617181/790169\nin_value_termination = in_value_char (!in_value_end in_value_char)* in_value_end\nin_value_char        = ~r"[^(), ]"\nin_value_end         = closed_bracket / (spaces comma)\n\n# Formats\ndate_format = ~r"\\d{4}-\\d{2}-\\d{2}"\ntime_format = ~r"T\\d{2}:\\d{2}:\\d{2}" ("." ms_format)?\nms_format   = ~r"\\d{1,6}"\ntz_format   = ~r"[+-]\\d{2}:\\d{2}"\n\niso_8601_date_format = date_format time_format? ("Z" / tz_format)? &end_value\nrel_date_format      = ~r"[+-][0-9]+[wdhm]" &end_value\nduration_format      = numeric ("ms"/"s"/"min"/"m"/"hr"/"h"/"day"/"d"/"wk"/"w") &end_value\nsize_format          = numeric ("bit"/"nb"/"bytes"/"kb"/"mb"/"gb"/"tb"/"pb"/"eb"/"zb"/"yb"/"kib"/"mib"/"gib"/"tib"/"pib"/"eib"/"zib"/"yib") &end_value\npercentage_format    = numeric "%"\n\n# NOTE: the order in which these operators are listed matters because for\n# example, if < comes before <= it will match that even if the operator is <=\noperator             = ">=" / "<=" / ">" / "<" / "=" / "!="\nor_operator          = ~r"OR"i  &end_value\nand_operator         = ~r"AND"i &end_value\nnumeric              = ~r"[0-9]+(?:\\.[0-9]*)?"\nopen_paren           = "("\nclosed_paren         = ")"\nopen_bracket         = "["\nclosed_bracket       = "]"\nsep                  = ":"\nnegation             = "!"\ncomma                = ","\nspaces               = " "*\n\nend_value = ~r"[\\t\\n )]|$"\n')

def translate_wildcard(pat: str) -> str:
    if False:
        print('Hello World!')
    '\n    Translate a shell PATTERN to a regular expression.\n    modified from: https://github.com/python/cpython/blob/2.7/Lib/fnmatch.py#L85\n    '
    (i, n) = (0, len(pat))
    res = ''
    while i < n:
        c = pat[i]
        i = i + 1
        if c == '\\' and i < n:
            res += re.escape(pat[i])
            i += 1
        elif c == '*':
            res += '.*'
        elif c in '()[]?*+-|^$\\.&~# \t\n\r\x0b\x0c':
            res += re.escape(c)
        else:
            res += c
    return '^' + res + '$'

def translate_escape_sequences(string: str) -> str:
    if False:
        i = 10
        return i + 15
    '\n    A non-wildcard pattern can contain escape sequences that we need to handle.\n    - \\* because a single asterisk represents a wildcard, so it needs to be escaped\n    '
    (i, n) = (0, len(string))
    res = ''
    while i < n:
        c = string[i]
        i = i + 1
        if c == '\\' and i < n:
            d = string[i]
            if d == '*':
                i += 1
                res += d
            else:
                res += c
        else:
            res += c
    return res

def flatten(children):
    if False:
        while True:
            i = 10

    def _flatten(seq):
        if False:
            print('Hello World!')
        for item in seq:
            if isinstance(item, list):
                yield from _flatten(item)
            else:
                yield item
    if not (children and isinstance(children, list) and isinstance(children[0], list)):
        return children
    children = [child for group in children for child in _flatten(group)]
    children = [_f for _f in _flatten(children) if _f]
    return children

def remove_optional_nodes(children):
    if False:
        print('Hello World!')

    def is_not_optional(child):
        if False:
            print('Hello World!')
        return not (isinstance(child, Node) and isinstance(child.expr, Optional))
    return list(filter(is_not_optional, children))

def remove_space(children):
    if False:
        return 10

    def is_not_space(text):
        if False:
            print('Hello World!')
        return not (isinstance(text, str) and text == ' ' * len(text))
    return list(filter(is_not_space, children))

def process_list(first, remaining):
    if False:
        i = 10
        return i + 15
    if any((isinstance(item[4], Node) for item in remaining)):
        raise InvalidSearchQuery('Lists should not have empty values')
    return [first, *(item[4][0] for item in remaining)]

def is_negated(node):
    if False:
        while True:
            i = 10
    if isinstance(node, list):
        node = node[0]
    return node.text == '!'

def handle_negation(negation, operator):
    if False:
        while True:
            i = 10
    operator = get_operator_value(operator)
    if is_negated(negation):
        return OPERATOR_NEGATION_MAP.get(operator, '!=')
    return operator

def get_operator_value(operator):
    if False:
        print('Hello World!')
    if isinstance(operator, Node):
        operator = '=' if isinstance(operator.expr, Optional) else operator.text
    elif isinstance(operator, list):
        operator = operator[0]
    return operator

class SearchBoolean(namedtuple('SearchBoolean', 'left_term operator right_term')):
    BOOLEAN_AND = 'AND'
    BOOLEAN_OR = 'OR'

    @staticmethod
    def is_or_operator(value):
        if False:
            i = 10
            return i + 15
        return value == SearchBoolean.BOOLEAN_OR

    @staticmethod
    def is_operator(value):
        if False:
            while True:
                i = 10
        return value == SearchBoolean.BOOLEAN_AND or SearchBoolean.is_or_operator(value)

class ParenExpression(namedtuple('ParenExpression', 'children')):

    def to_query_string(self):
        if False:
            print('Hello World!')
        children = ''
        for child in self.children:
            if isinstance(child, str):
                children += f' {child}'
            else:
                children += f' {child.to_query_string()}'
        return f'({children})'

class SearchKey(NamedTuple):
    name: str

    @property
    def is_tag(self) -> bool:
        if False:
            i = 10
            return i + 15
        return TAG_KEY_RE.match(self.name) or (self.name not in SEARCH_MAP and self.name not in FIELD_ALIASES and (not self.is_measurement) and (not self.is_span_op_breakdown))

    @property
    def is_measurement(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return is_measurement(self.name) and self.name not in SEARCH_MAP

    @property
    def is_span_op_breakdown(self) -> bool:
        if False:
            print('Hello World!')
        return is_span_op_breakdown(self.name) and self.name not in SEARCH_MAP

class SearchValue(NamedTuple):
    raw_value: Union[str, int, datetime, Sequence[int], Sequence[str]]

    @property
    def value(self):
        if False:
            return 10
        if self.is_wildcard():
            return translate_wildcard(self.raw_value)
        elif isinstance(self.raw_value, str):
            return translate_escape_sequences(self.raw_value)
        return self.raw_value

    def to_query_string(self):
        if False:
            for i in range(10):
                print('nop')
        if type(self.raw_value) in [list, tuple]:
            ret_val = reduce(lambda acc, elm: f'{acc}, {elm}', self.raw_value)
            ret_val = '[' + ret_val + ']'
            return ret_val
        if isinstance(self.raw_value, datetime):
            return self.raw_value.isoformat()
        return str(self.value)

    def is_wildcard(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(self.raw_value, str):
            return False
        return bool(WILDCARD_CHARS.search(self.raw_value))

    def is_event_id(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Return whether the current value is a valid event id\n\n        Empty strings are valid, so that it can be used for has:id queries\n        '
        if isinstance(self.raw_value, list):
            return all((isinstance(value, str) and is_event_id(value) for value in self.raw_value))
        if not isinstance(self.raw_value, str):
            return False
        return is_event_id(self.raw_value) or self.raw_value == ''

    def is_span_id(self) -> bool:
        if False:
            while True:
                i = 10
        'Return whether the current value is a valid span id\n\n        Empty strings are valid, so that it can be used for has:trace.span queries\n        '
        if not isinstance(self.raw_value, str):
            return False
        return is_span_id(self.raw_value) or self.raw_value == ''

class SearchFilter(NamedTuple):
    key: SearchKey
    operator: str
    value: SearchValue

    def __str__(self):
        if False:
            return 10
        return f'{self.key.name}{self.operator}{self.value.raw_value}'

    def to_query_string(self):
        if False:
            print('Hello World!')
        if self.operator == 'IN':
            return f'{self.key.name}:{self.value.to_query_string()}'
        elif self.operator == 'NOT IN':
            return f'!{self.key.name}:{self.value.to_query_string()}'
        else:
            return f'{self.key.name}:{self.operator}{self.value.to_query_string()}'

    @property
    def is_negation(self) -> bool:
        if False:
            return 10
        return bool(self.operator == '!=' and self.value.raw_value != '' or (self.operator == '=' and self.value.raw_value == '') or (self.operator == 'NOT IN' and self.value.raw_value))

    @property
    def is_in_filter(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.operator in ('IN', 'NOT IN')

class AggregateFilter(NamedTuple):
    key: SearchKey
    operator: str
    value: SearchValue

    def __str__(self):
        if False:
            return 10
        return f'{self.key.name}{self.operator}{self.value.raw_value}'

class AggregateKey(NamedTuple):
    name: str

@dataclass
class SearchConfig:
    """
    Configures how the search parser interprets a search query
    """
    key_mappings: Mapping[str, List[str]] = field(default_factory=dict)
    text_operator_keys: Set[str] = field(default_factory=set)
    duration_keys: Set[str] = field(default_factory=set)
    percentage_keys: Set[str] = field(default_factory=set)
    numeric_keys: Set[str] = field(default_factory=set)
    date_keys: Set[str] = field(default_factory=set)
    boolean_keys: Set[str] = field(default_factory=set)
    is_filter_translation: Mapping[str, Tuple[str, Any]] = field(default_factory=dict)
    allow_boolean = True
    allowed_keys: Set[str] = field(default_factory=set)
    blocked_keys: Set[str] = field(default_factory=set)
    free_text_key = 'message'

    @classmethod
    def create_from(cls, search_config: SearchConfig, **overrides):
        if False:
            print('Hello World!')
        config = cls(**asdict(search_config))
        for (key, val) in overrides.items():
            setattr(config, key, val)
        return config

class SearchVisitor(NodeVisitor):
    unwrapped_exceptions = (InvalidSearchQuery,)

    def __init__(self, config=None, params=None, builder=None):
        if False:
            while True:
                i = 10
        super().__init__()
        if config is None:
            config = SearchConfig()
        self.config = config
        self.params = params if params is not None else {}
        if builder is None:
            from sentry.search.events.builder import UnresolvedQuery
            self.builder = UnresolvedQuery(dataset=Dataset.Discover, params=self.params, config=QueryBuilderConfig(functions_acl=list(FUNCTIONS)))
        else:
            self.builder = builder

    @cached_property
    def key_mappings_lookup(self):
        if False:
            for i in range(10):
                print('nop')
        lookup = {}
        for (target_field, source_fields) in self.config.key_mappings.items():
            for source_field in source_fields:
                lookup[source_field] = target_field
        return lookup

    def is_numeric_key(self, key):
        if False:
            return 10
        return key in self.config.numeric_keys or is_measurement(key) or is_span_op_breakdown(key) or (self.builder.get_field_type(key) == 'number') or self.is_duration_key(key)

    def is_duration_key(self, key):
        if False:
            return 10
        duration_types = [*DURATION_UNITS, 'duration']
        return key in self.config.duration_keys or is_duration_measurement(key) or is_span_op_breakdown(key) or (self.builder.get_field_type(key) in duration_types)

    def is_size_key(self, key):
        if False:
            i = 10
            return i + 15
        return self.builder.get_field_type(key) in SIZE_UNITS

    def is_date_key(self, key):
        if False:
            while True:
                i = 10
        return key in self.config.date_keys

    def is_boolean_key(self, key):
        if False:
            i = 10
            return i + 15
        return key in self.config.boolean_keys

    def is_percentage_key(self, key):
        if False:
            print('Hello World!')
        return key in self.config.percentage_keys

    def visit_search(self, node, children):
        if False:
            while True:
                i = 10
        return flatten(remove_space(children[1]))

    def visit_term(self, node, children):
        if False:
            print('Hello World!')
        return flatten(remove_space(children[0]))

    def visit_boolean_operator(self, node, children):
        if False:
            i = 10
            return i + 15
        if not self.config.allow_boolean:
            raise InvalidSearchQuery('Boolean statements containing "OR" or "AND" are not supported in this search')
        return children[0]

    def visit_free_text_unquoted(self, node, children):
        if False:
            return 10
        return node.text.strip(' ') or None

    def visit_free_text(self, node, children):
        if False:
            while True:
                i = 10
        if not children[0]:
            return None
        return SearchFilter(SearchKey(self.config.free_text_key), '=', SearchValue(children[0]))

    def visit_paren_group(self, node, children):
        if False:
            for i in range(10):
                print('nop')
        if not self.config.allow_boolean:
            return SearchFilter(SearchKey(self.config.free_text_key), '=', SearchValue(node.text))
        children = remove_space(remove_optional_nodes(flatten(children)))
        children = flatten(children[1])
        if len(children) == 0:
            return node.text
        return ParenExpression(children)

    def _handle_basic_filter(self, search_key, operator, search_value):
        if False:
            i = 10
            return i + 15
        if self.is_date_key(search_key.name):
            raise InvalidSearchQuery(f'{search_key.name}: Invalid date: {search_value.raw_value}. Expected +/-duration (e.g. +1h) or ISO 8601-like (e.g. {datetime.now().isoformat()[:-4]}).')
        if self.is_boolean_key(search_key.name):
            raise InvalidSearchQuery(f'{search_key.name}: Invalid boolean: {search_value.raw_value}. Expected true, 1, false, or 0.')
        if self.is_numeric_key(search_key.name):
            raise InvalidSearchQuery(f'{search_key.name}: Invalid number: {search_value.raw_value}. Expected number then optional k, m, or b suffix (e.g. 500k).')
        return SearchFilter(search_key, operator, search_value)

    def _handle_numeric_filter(self, search_key, operator, search_value):
        if False:
            return 10
        operator = get_operator_value(operator)
        if self.is_numeric_key(search_key.name):
            try:
                search_value = SearchValue(parse_numeric_value(*search_value))
            except InvalidQuery as exc:
                raise InvalidSearchQuery(str(exc))
            return SearchFilter(search_key, operator, search_value)
        return self._handle_text_filter(search_key, operator, SearchValue(''.join(search_value)))

    def visit_date_filter(self, node, children):
        if False:
            return 10
        (search_key, _, operator, search_value) = children
        if self.is_date_key(search_key.name):
            try:
                search_value = parse_datetime_string(search_value)
            except InvalidQuery as exc:
                raise InvalidSearchQuery(str(exc))
            return SearchFilter(search_key, operator, SearchValue(search_value))
        search_value = operator + search_value if operator != '=' else search_value
        return self._handle_basic_filter(search_key, '=', SearchValue(search_value))

    def visit_specific_date_filter(self, node, children):
        if False:
            for i in range(10):
                print('nop')
        (search_key, _, date_value) = children
        if not self.is_date_key(search_key.name):
            return self._handle_basic_filter(search_key, '=', SearchValue(date_value))
        try:
            (from_val, to_val) = parse_datetime_value(date_value)
        except InvalidQuery as exc:
            raise InvalidSearchQuery(str(exc))
        return [SearchFilter(search_key, '>=', SearchValue(from_val[0])), SearchFilter(search_key, '<', SearchValue(to_val[0]))]

    def visit_rel_date_filter(self, node, children):
        if False:
            return 10
        (search_key, _, value) = children
        if self.is_date_key(search_key.name):
            try:
                (from_val, to_val) = parse_datetime_range(value.text)
            except InvalidQuery as exc:
                raise InvalidSearchQuery(str(exc))
            if from_val is not None:
                operator = '>='
                search_value = from_val[0]
            else:
                operator = '<='
                search_value = to_val[0]
            return SearchFilter(search_key, operator, SearchValue(search_value))
        return self._handle_basic_filter(search_key, '=', SearchValue(value.text))

    def visit_duration_filter(self, node, children):
        if False:
            i = 10
            return i + 15
        (negation, search_key, _, operator, search_value) = children
        if self.is_duration_key(search_key.name) or self.is_numeric_key(search_key.name):
            operator = handle_negation(negation, operator)
        else:
            operator = get_operator_value(operator)
        if self.is_duration_key(search_key.name):
            try:
                search_value = parse_duration(*search_value)
            except InvalidQuery as exc:
                raise InvalidSearchQuery(str(exc))
            return SearchFilter(search_key, operator, SearchValue(search_value))
        if self.is_numeric_key(search_key.name):
            return self._handle_numeric_filter(search_key, operator, search_value)
        search_value = ''.join(search_value)
        search_value = operator + search_value if operator not in ('=', '!=') else search_value
        operator = '!=' if is_negated(negation) else '='
        return self._handle_basic_filter(search_key, operator, SearchValue(search_value))

    def visit_size_filter(self, node, children):
        if False:
            return 10
        (negation, search_key, _, operator, search_value) = children
        if self.is_size_key(search_key.name):
            operator = handle_negation(negation, operator)
        else:
            operator = get_operator_value(operator)
        if self.is_size_key(search_key.name):
            try:
                search_value = parse_size(*search_value)
            except InvalidQuery as exc:
                raise InvalidSearchQuery(str(exc))
            return SearchFilter(search_key, operator, SearchValue(search_value))
        search_value = ''.join(search_value)
        search_value = operator + search_value if operator not in ('=', '!=') else search_value
        operator = '!=' if is_negated(negation) else '='
        return self._handle_basic_filter(search_key, operator, SearchValue(search_value))

    def visit_boolean_filter(self, node, children):
        if False:
            return 10
        (negation, search_key, sep, search_value) = children
        negated = is_negated(negation)
        if self.is_numeric_key(search_key.name):
            return self._handle_numeric_filter(search_key, '!=' if negated else '=', [search_value.text, ''])
        if self.is_boolean_key(search_key.name):
            if search_value.text.lower() in ('true', '1'):
                search_value = SearchValue(0 if negated else 1)
            elif search_value.text.lower() in ('false', '0'):
                search_value = SearchValue(1 if negated else 0)
            else:
                raise InvalidSearchQuery(f'Invalid boolean field: {search_key}')
            return SearchFilter(search_key, '=', search_value)
        search_value = SearchValue(search_value.text)
        return self._handle_basic_filter(search_key, '=' if not negated else '!=', search_value)

    def visit_numeric_in_filter(self, node, children):
        if False:
            while True:
                i = 10
        (negation, search_key, _, search_value) = children
        operator = handle_negation(negation, 'IN')
        if self.is_numeric_key(search_key.name):
            try:
                search_value = SearchValue([parse_numeric_value(*val) for val in search_value])
            except InvalidQuery as exc:
                raise InvalidSearchQuery(str(exc))
            return SearchFilter(search_key, operator, search_value)
        search_value = SearchValue([''.join(value) for value in search_value])
        return self._handle_basic_filter(search_key, operator, search_value)

    def visit_numeric_filter(self, node, children):
        if False:
            return 10
        (negation, search_key, _, operator, search_value) = children
        if self.is_numeric_key(search_key.name) or search_key.name in self.config.text_operator_keys:
            operator = handle_negation(negation, operator)
        else:
            operator = get_operator_value(operator)
        if self.is_numeric_key(search_key.name):
            return self._handle_numeric_filter(search_key, operator, search_value)
        search_value = SearchValue(''.join(search_value))
        if operator not in ('=', '!=') and search_key.name not in self.config.text_operator_keys:
            search_value = search_value._replace(raw_value=f'{operator}{search_value.raw_value}')
        if search_key.name not in self.config.text_operator_keys:
            operator = '!=' if is_negated(negation) else '='
        return self._handle_basic_filter(search_key, operator, search_value)

    def visit_aggregate_duration_filter(self, node, children):
        if False:
            i = 10
            return i + 15
        (negation, search_key, _, operator, search_value) = children
        operator = handle_negation(negation, operator)
        try:
            result_type = self.builder.get_function_result_type(search_key.name)
            if result_type == 'duration' or result_type in DURATION_UNITS:
                aggregate_value = parse_duration(*search_value)
            else:
                aggregate_value = parse_numeric_value(*search_value)
        except ValueError:
            raise InvalidSearchQuery(f'Invalid aggregate query condition: {search_key}')
        except InvalidQuery as exc:
            raise InvalidSearchQuery(str(exc))
        return AggregateFilter(search_key, operator, SearchValue(aggregate_value))

    def visit_aggregate_size_filter(self, node, children):
        if False:
            for i in range(10):
                print('nop')
        (negation, search_key, _, operator, search_value) = children
        operator = handle_negation(negation, operator)
        try:
            aggregate_value = parse_size(*search_value)
        except ValueError:
            raise InvalidSearchQuery(f'Invalid aggregate query condition: {search_key}')
        except InvalidQuery as exc:
            raise InvalidSearchQuery(str(exc))
        return AggregateFilter(search_key, operator, SearchValue(aggregate_value))

    def visit_aggregate_percentage_filter(self, node, children):
        if False:
            return 10
        (negation, search_key, _, operator, search_value) = children
        operator = handle_negation(negation, operator)
        aggregate_value = None
        try:
            result_type = self.builder.get_function_result_type(search_key.name)
            if result_type == 'percentage':
                aggregate_value = parse_percentage(search_value)
        except ValueError:
            raise InvalidSearchQuery(f'Invalid aggregate query condition: {search_key}')
        except InvalidQuery as exc:
            raise InvalidSearchQuery(str(exc))
        if aggregate_value is not None:
            return AggregateFilter(search_key, operator, SearchValue(aggregate_value))
        search_value = operator + search_value if operator != '=' else search_value
        return AggregateFilter(search_key, '=', SearchValue(search_value))

    def visit_aggregate_numeric_filter(self, node, children):
        if False:
            for i in range(10):
                print('nop')
        (negation, search_key, _, operator, search_value) = children
        operator = handle_negation(negation, operator)
        try:
            aggregate_value = parse_numeric_value(*search_value)
        except InvalidQuery as exc:
            raise InvalidSearchQuery(str(exc))
        return AggregateFilter(search_key, operator, SearchValue(aggregate_value))

    def visit_aggregate_date_filter(self, node, children):
        if False:
            while True:
                i = 10
        (negation, search_key, _, operator, search_value) = children
        operator = handle_negation(negation, operator)
        is_date_aggregate = any((key in search_key.name for key in self.config.date_keys))
        if is_date_aggregate:
            try:
                search_value = parse_datetime_string(search_value)
            except InvalidQuery as exc:
                raise InvalidSearchQuery(str(exc))
            return AggregateFilter(search_key, operator, SearchValue(search_value))
        search_value = operator + search_value if operator != '=' else search_value
        return AggregateFilter(search_key, '=', SearchValue(search_value))

    def visit_aggregate_rel_date_filter(self, node, children):
        if False:
            print('Hello World!')
        (negation, search_key, _, operator, search_value) = children
        operator = handle_negation(negation, operator)
        is_date_aggregate = any((key in search_key.name for key in self.config.date_keys))
        if is_date_aggregate:
            try:
                (from_val, to_val) = parse_datetime_range(search_value.text)
            except InvalidQuery as exc:
                raise InvalidSearchQuery(str(exc))
            if from_val is not None:
                operator = '>='
                search_value = from_val[0]
            else:
                operator = '<='
                search_value = to_val[0]
            return AggregateFilter(search_key, operator, SearchValue(search_value))
        search_value = operator + search_value.text if operator != '=' else search_value
        return AggregateFilter(search_key, '=', SearchValue(search_value))

    def visit_has_filter(self, node, children):
        if False:
            while True:
                i = 10
        (negation, _, _, _, (search_key,)) = children
        if isinstance(search_key, SearchValue):
            raise InvalidSearchQuery('Invalid format for "has" search: was expecting a field or tag instead')
        operator = '=' if is_negated(negation) else '!='
        return SearchFilter(search_key, operator, SearchValue(''))

    def visit_is_filter(self, node, children):
        if False:
            for i in range(10):
                print('nop')
        (negation, _, _, _, search_value) = children
        translators = self.config.is_filter_translation
        if not translators:
            raise InvalidSearchQuery('"is:" queries are not supported in this search.')
        if search_value.raw_value.startswith('['):
            raise InvalidSearchQuery('"in" syntax invalid for "is" search')
        if search_value.raw_value not in translators:
            valid_keys = sorted(translators.keys())
            raise InvalidSearchQuery(f'Invalid value for "is" search, valid values are {valid_keys}')
        (search_key, search_value) = translators[search_value.raw_value]
        operator = '!=' if is_negated(negation) else '='
        search_key = SearchKey(search_key)
        search_value = SearchValue(search_value)
        return SearchFilter(search_key, operator, search_value)

    def visit_text_in_filter(self, node, children):
        if False:
            i = 10
            return i + 15
        (negation, search_key, _, search_value) = children
        operator = 'IN'
        search_value = SearchValue(search_value)
        operator = handle_negation(negation, operator)
        return self._handle_basic_filter(search_key, operator, search_value)

    def visit_text_filter(self, node, children):
        if False:
            for i in range(10):
                print('nop')
        (negation, search_key, _, operator, search_value) = children
        operator = get_operator_value(operator)
        if not search_value.raw_value and (not node.children[4].text):
            raise InvalidSearchQuery(f"Empty string after '{search_key.name}:'")
        if operator not in ('=', '!=') and search_key.name not in self.config.text_operator_keys:
            search_value = search_value._replace(raw_value=f'{operator}{search_value.raw_value}')
            operator = '='
        operator = handle_negation(negation, operator)
        return self._handle_text_filter(search_key, operator, search_value)

    def _handle_text_filter(self, search_key, operator, search_value):
        if False:
            return 10
        if operator not in ('=', '!=') and search_key.name not in self.config.text_operator_keys:
            search_value = search_value._replace(raw_value=f'{operator}{search_value.raw_value}')
            operator = '='
        return self._handle_basic_filter(search_key, operator, search_value)

    def visit_key(self, node, children):
        if False:
            i = 10
            return i + 15
        return node.text

    def visit_quoted_key(self, node, children):
        if False:
            print('Hello World!')
        return children[1].text

    def visit_explicit_tag_key(self, node, children):
        if False:
            return 10
        return SearchKey(f'tags[{children[2].name}]')

    def visit_aggregate_key(self, node, children):
        if False:
            for i in range(10):
                print('nop')
        children = remove_optional_nodes(children)
        children = remove_space(children)
        if len(children) == 3:
            (function_name, open_paren, close_paren) = children
            args = ''
        else:
            (function_name, open_paren, args, close_paren) = children
            args = ', '.join(args[0])
        key = ''.join([function_name, open_paren, args, close_paren])
        return AggregateKey(self.key_mappings_lookup.get(key, key))

    def visit_function_args(self, node, children):
        if False:
            print('Hello World!')
        return process_list(children[0], children[1])

    def visit_aggregate_param(self, node, children):
        if False:
            i = 10
            return i + 15
        return children[0]

    def visit_raw_aggregate_param(self, node, children):
        if False:
            print('Hello World!')
        return node.text

    def visit_quoted_aggregate_param(self, node, children):
        if False:
            for i in range(10):
                print('nop')
        value = ''.join((node.text for node in flatten(children[1])))
        return f'"{value}"'

    def visit_search_key(self, node, children):
        if False:
            i = 10
            return i + 15
        key = children[0]
        if self.config.allowed_keys and key not in self.config.allowed_keys or key in self.config.blocked_keys:
            raise InvalidSearchQuery(f'Invalid key for this search: {key}')
        return SearchKey(self.key_mappings_lookup.get(key, key))

    def visit_text_key(self, node, children):
        if False:
            while True:
                i = 10
        return children[0]

    def visit_value(self, node, children):
        if False:
            print('Hello World!')
        value = node.text
        idx = value.find('"')
        if idx == 0:
            raise InvalidSearchQuery(f"Invalid quote at '{node.text}': quotes must enclose text or be escaped.")
        while idx != -1:
            if value[idx - 1] != '\\':
                raise InvalidSearchQuery(f"Invalid quote at '{node.text}': quotes must enclose text or be escaped.")
            value = value[idx + 1:]
            idx = value.find('"')
        return node.text.replace('\\"', '"')

    def visit_quoted_value(self, node, children):
        if False:
            while True:
                i = 10
        value = ''.join((node.text for node in flatten(children[1])))
        value = value.replace('\\"', '"')
        return value

    def visit_in_value(self, node, children):
        if False:
            while True:
                i = 10
        return node.text.replace('\\"', '"')

    def visit_text_in_value(self, node, children):
        if False:
            while True:
                i = 10
        return children[0]

    def visit_search_value(self, node, children):
        if False:
            print('Hello World!')
        return SearchValue(children[0])

    def visit_numeric_value(self, node, children):
        if False:
            print('Hello World!')
        (sign, value, suffix, _) = children
        sign = sign[0].text if isinstance(sign, list) else ''
        suffix = suffix[0].text if isinstance(suffix, list) else ''
        return [f'{sign}{value}', suffix]

    def visit_boolean_value(self, node, children):
        if False:
            i = 10
            return i + 15
        return node

    def visit_text_in_list(self, node, children):
        if False:
            print('Hello World!')
        return process_list(children[1], children[2])

    def visit_numeric_in_list(self, node, children):
        if False:
            i = 10
            return i + 15
        return process_list(children[1], children[2])

    def visit_iso_8601_date_format(self, node, children):
        if False:
            print('Hello World!')
        return node.text

    def visit_rel_date_format(self, node, children):
        if False:
            i = 10
            return i + 15
        return node

    def visit_duration_format(self, node, children):
        if False:
            i = 10
            return i + 15
        return [children[0], children[1][0].text]

    def visit_size_format(self, node, children):
        if False:
            for i in range(10):
                print('nop')
        return [children[0], children[1][0].text]

    def visit_percentage_format(self, node, children):
        if False:
            return 10
        return children[0]

    def visit_operator(self, node, children):
        if False:
            return 10
        return node.text

    def visit_or_operator(self, node, children):
        if False:
            for i in range(10):
                print('nop')
        return node.text.upper()

    def visit_and_operator(self, node, children):
        if False:
            i = 10
            return i + 15
        return node.text.upper()

    def visit_numeric(self, node, children):
        if False:
            while True:
                i = 10
        return node.text

    def visit_open_paren(self, node, children):
        if False:
            while True:
                i = 10
        return node.text

    def visit_closed_paren(self, node, children):
        if False:
            while True:
                i = 10
        return node.text

    def visit_open_bracket(self, node, children):
        if False:
            while True:
                i = 10
        return node.text

    def visit_closed_bracket(self, node, children):
        if False:
            while True:
                i = 10
        return node.text

    def visit_sep(self, node, children):
        if False:
            print('Hello World!')
        return node

    def visit_negation(self, node, children):
        if False:
            for i in range(10):
                print('nop')
        return node

    def visit_comma(self, node, children):
        if False:
            print('Hello World!')
        return node

    def visit_spaces(self, node, children):
        if False:
            i = 10
            return i + 15
        return ' '

    def generic_visit(self, node, children):
        if False:
            print('Hello World!')
        return children or node
default_config = SearchConfig(duration_keys={'transaction.duration'}, percentage_keys={'percentage'}, text_operator_keys={SEMVER_ALIAS, SEMVER_BUILD_ALIAS}, numeric_keys={'project_id', 'project.id', 'issue.id', 'stack.colno', 'stack.lineno', 'stack.stack_level', 'transaction.duration'}, date_keys={'start', 'end', 'last_seen()', 'time', 'timestamp', 'timestamp.to_hour', 'timestamp.to_day', 'error.received'}, boolean_keys={'error.handled', 'error.unhandled', 'error.main_thread', 'stack.in_app', 'is_application', TEAM_KEY_TRANSACTION_ALIAS})

def parse_search_query(query, config=None, params=None, builder=None, config_overrides=None) -> list[SearchFilter]:
    if False:
        while True:
            i = 10
    if config is None:
        config = default_config
    try:
        tree = event_search_grammar.parse(query)
    except IncompleteParseError as e:
        idx = e.column()
        prefix = query[max(0, idx - 5):idx]
        suffix = query[idx:idx + 5]
        raise InvalidSearchQuery('{} {}'.format(f"Parse error at '{prefix}{suffix}' (column {e.column():d}).", 'This is commonly caused by unmatched parentheses. Enclose any text in double quotes.'))
    if config_overrides:
        config = SearchConfig.create_from(config, **config_overrides)
    return SearchVisitor(config, params=params, builder=builder).visit(tree)