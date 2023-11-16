from __future__ import absolute_import
import re
import sys
from sre_parse import parse, AT, AT_BEGINNING, AT_BEGINNING_STRING, AT_END, AT_END_STRING, BRANCH, SUBPATTERN
from st2common.util.jinja import render_values
from st2common.constants import keyvalue as kv_constants
from st2common.services import keyvalues as kv_service
from st2common.exceptions.content import ParseException
from st2common import log
__all__ = ['ActionAliasFormatParser', 'extract_parameters_for_action_alias_db', 'extract_parameters', 'search_regex_tokens']
LOG = log.getLogger(__name__)
if sys.version_info > (3,):
    SUBPATTERN_INDEX = 3
else:
    SUBPATTERN_INDEX = 1

class ActionAliasFormatParser(object):

    def __init__(self, alias_format=None, param_stream=None):
        if False:
            for i in range(10):
                print('nop')
        self._format = alias_format or ''
        self._original_param_stream = param_stream or ''
        self._param_stream = self._original_param_stream
        self._snippets = self.generate_snippets()
        (self._kv_pairs, self._param_stream) = self.match_kv_pairs_at_end()
        self._optional = self.generate_optional_params_regex()
        self._regex = self.transform_format_string_into_regex()

    def generate_snippets(self):
        if False:
            for i in range(10):
                print('nop')
        snippets = dict()
        snippets['key'] = '\\s*(\\S+?)\\s*'
        snippets['value'] = '""|\\\'\\\'|"(.+?)"|\\\'(.+?)\\\'|({.+?})|(\\S+)'
        snippets['ext_value'] = '""|\\\'\\\'|"(.+?)"|\\\'(.+?)\\\'|({.+?})|(.+?)'
        snippets['pairs'] = '(?:^|\\s+){key}=({value})'.format(**snippets)
        snippets['ending'] = '.*?(({pairs}\\s*)*)$'.format(**snippets)
        snippets['default'] = '\\s*=\\s*(?:{ext_value})\\s*'.format(**snippets)
        snippets['optional'] = '{{' + snippets['key'] + snippets['default'] + '}}'
        snippets['required'] = '{{' + snippets['key'] + '}}'
        return snippets

    def match_kv_pairs_at_end(self):
        if False:
            for i in range(10):
                print('nop')
        param_stream = self._param_stream
        ending_pairs = re.match(self._snippets['ending'], param_stream, re.DOTALL)
        has_ending_pairs = ending_pairs and ending_pairs.group(1)
        if has_ending_pairs:
            kv_pairs = re.findall(self._snippets['pairs'], ending_pairs.group(1), re.DOTALL)
            param_stream = param_stream.replace(ending_pairs.group(1), '')
        else:
            kv_pairs = []
        param_stream = ' %s ' % param_stream
        return (kv_pairs, param_stream)

    def generate_optional_params_regex(self):
        if False:
            while True:
                i = 10
        return re.findall(self._snippets['optional'], self._format, re.DOTALL)

    def transform_format_string_into_regex(self):
        if False:
            print('Hello World!')
        param_match = '\\1["\\\']?(?P<\\2>(?:(?<=\\\').+?(?=\\\')|(?<=").+?(?=")|{.+?}|.+?))["\\\']?'
        reg = re.sub('(\\s*)' + self._snippets['optional'], '(?:' + param_match + ')?', self._format)
        reg = re.sub('(\\s*)' + self._snippets['required'], param_match, reg)
        reg_tokens = parse(reg, flags=re.DOTALL)
        if not search_regex_tokens(((AT, AT_BEGINNING), (AT, AT_BEGINNING_STRING)), reg_tokens):
            reg = '^\\s*' + reg
        if not search_regex_tokens(((AT, AT_END), (AT, AT_END_STRING)), reg_tokens, backwards=True):
            reg = reg + '\\s*$'
        return re.compile(reg, re.DOTALL)

    def match_params_in_stream(self, matched_stream):
        if False:
            return 10
        if not matched_stream:
            raise ParseException('Command "%s" doesn\'t match format string "%s"' % (self._original_param_stream, self._format))
        if matched_stream:
            result = matched_stream.groupdict()
        for param in self._optional:
            matched_value = result[param[0]] if matched_stream else None
            matched_result = matched_value or ''.join(param[1:])
            if matched_result is not None:
                result[param[0]] = matched_result
        for pair in self._kv_pairs:
            result[pair[0]] = ''.join(pair[2:])
        if self._format and (not (self._param_stream.strip() or any(result.values()))):
            raise ParseException('No value supplied and no default value found.')
        return result

    def get_extracted_param_value(self):
        if False:
            print('Hello World!')
        '\n        Match command against the format string and extract parameters from the command string.\n\n        :rtype: ``dict``\n        '
        matched_stream = self._regex.search(self._param_stream)
        return self.match_params_in_stream(matched_stream)

    def get_multiple_extracted_param_value(self):
        if False:
            return 10
        '\n        Match command against the format string and extract parameters from the command string.\n\n        :rtype: ``list of dicts``\n        '
        matched_streams = self._regex.finditer(self._param_stream)
        results = []
        for matched_stream in matched_streams:
            results.append(self.match_params_in_stream(matched_stream))
        return results

def extract_parameters_for_action_alias_db(action_alias_db, format_str, param_stream, match_multiple=False):
    if False:
        i = 10
        return i + 15
    '\n    Extract parameters from the user input based on the provided format string.\n\n    Note: This function makes sure that the provided format string is indeed available in the\n    action_alias_db.formats.\n    '
    formats = []
    formats = action_alias_db.get_format_strings()
    if format_str not in formats:
        raise ValueError('Format string "%s" is not available on the alias "%s"' % (format_str, action_alias_db.name))
    result = extract_parameters(format_str=format_str, param_stream=param_stream, match_multiple=match_multiple)
    return result

def extract_parameters(format_str, param_stream, match_multiple=False):
    if False:
        for i in range(10):
            print('nop')
    parser = ActionAliasFormatParser(alias_format=format_str, param_stream=param_stream)
    if match_multiple:
        return parser.get_multiple_extracted_param_value()
    else:
        return parser.get_extracted_param_value()

def inject_immutable_parameters(action_alias_db, multiple_execution_parameters, action_context):
    if False:
        while True:
            i = 10
    '\n    Inject immutable parameters from the alias definiton on the execution parameters.\n    Jinja expressions will be resolved.\n    '
    immutable_parameters = action_alias_db.immutable_parameters or {}
    if not immutable_parameters:
        return multiple_execution_parameters
    user = action_context.get('user', None)
    context = {}
    context.update({kv_constants.DATASTORE_PARENT_SCOPE: {kv_constants.SYSTEM_SCOPE: kv_service.KeyValueLookup(scope=kv_constants.FULL_SYSTEM_SCOPE), kv_constants.USER_SCOPE: kv_service.UserKeyValueLookup(scope=kv_constants.FULL_USER_SCOPE, user=user)}})
    context.update(action_context)
    rendered_params = render_values(immutable_parameters, context)
    for exec_params in multiple_execution_parameters:
        overriden = [param for param in immutable_parameters.keys() if param in exec_params]
        if overriden:
            raise ValueError('Immutable arguments cannot be overriden: {}'.format(','.join(overriden)))
        exec_params.update(rendered_params)
    return multiple_execution_parameters

def search_regex_tokens(needle_tokens, haystack_tokens, backwards=False):
    if False:
        return 10
    "\n    Search a tokenized regex for any tokens in needle_tokens. Returns True if\n    any token tuple in needle_tokens is found, and False otherwise.\n\n    >>> search_regex_tokens(((AT, AT_END), (AT, AT_END)), parse(r'^asdf'))\n    False\n\n    :param needle_tokens: an iterable of token tuples\n\n    >>> needle_tokens = ((AT, AT_END), (AT, AT_END))\n    >>> search_regex_tokens(needle_tokens, parse(r'^asdf$'))\n    True\n\n    :param haystack_tokens: an iterable of token tuples from sre_parse.parse\n\n    >>> regex_tokens = parse(r'^(?:more regex)$')\n    >>> list(regex_tokens)  # doctest: +NORMALIZE_WHITESPACE\n    [(AT, AT_BEGINNING),\n     (SUBPATTERN, (None, 0, 0,\n     [(LITERAL, 109), (LITERAL, 111), (LITERAL, 114), (LITERAL, 101),\n      (LITERAL, 32), (LITERAL, 114), (LITERAL, 101), (LITERAL, 103),\n      (LITERAL, 101), (LITERAL, 120)])), (AT, AT_END)]\n\n    >>> search_regex_tokens(((AT, AT_END), (AT, AT_END)), regex_tokens)\n    True\n\n    :param backwards: Controls direction of search, defaults to False.\n    :type backwards: bool or None\n\n    .. note:: Set backwards to True if needle_tokens are more likely to be\n    found at the end of the haystack_tokens iterable, eg: ending anchors.\n\n    >>> search_regex_tokens(((AT, AT_END), (AT, AT_END)), parse(r'^asdf$'))\n    True\n    >>> search_regex_tokens(((AT, AT_END), (AT, AT_END)), parse(r'^asdf$'), backwards=True)\n    True\n\n    :rtype: ``bool``\n    "
    if backwards:
        haystack_tokens = reversed(haystack_tokens)
    for (rtoken_type, rtoken) in haystack_tokens:
        LOG.debug('Matching: ({}, {})'.format(rtoken_type, rtoken))
        if rtoken_type == SUBPATTERN:
            LOG.debug('SUBPATTERN: {}'.format(rtoken))
            if search_regex_tokens(needle_tokens, rtoken[SUBPATTERN_INDEX]):
                return True
        elif rtoken_type == BRANCH:
            LOG.debug('BRANCH: {}'.format(rtoken))
            if search_regex_tokens(needle_tokens, rtoken[1][1]):
                return True
        elif (rtoken_type, rtoken) in needle_tokens:
            LOG.debug('Found: {}'.format((rtoken_type, rtoken)))
            return True
    else:
        LOG.debug('Not found: {}'.format(needle_tokens))
        return False