from __future__ import absolute_import
import six
from six.moves import zip
from st2common.util.deep_copy import fast_deepcopy_dict
RULE_CRITERIA_UNESCAPED = ['.']
RULE_CRITERIA_ESCAPED = ['․']
RULE_CRITERIA_ESCAPE_TRANSLATION = dict(list(zip(RULE_CRITERIA_UNESCAPED, RULE_CRITERIA_ESCAPED)))
RULE_CRITERIA_UNESCAPE_TRANSLATION = dict(list(zip(RULE_CRITERIA_ESCAPED, RULE_CRITERIA_UNESCAPED)))
UNESCAPED = ['.', '$']
ESCAPED = ['．', '＄']
ESCAPE_TRANSLATION = dict(list(zip(UNESCAPED, ESCAPED)))
UNESCAPE_TRANSLATION = dict(list(zip(ESCAPED, UNESCAPED)) + list(zip(RULE_CRITERIA_ESCAPED, RULE_CRITERIA_UNESCAPED)))

def _translate_chars(field, translation):
    if False:
        return 10
    if isinstance(field, list):
        return _translate_chars_in_list(field, translation)
    if isinstance(field, dict):
        return _translate_chars_in_dict(field, translation)
    return field

def _translate_chars_in_list(field, translation):
    if False:
        return 10
    return [_translate_chars(value, translation) for value in field]

def _translate_chars_in_key(key, translation):
    if False:
        i = 10
        return i + 15
    for (k, v) in six.iteritems(translation):
        if k in key:
            key = key.replace(k, v)
    return key

def _translate_chars_in_dict(field, translation):
    if False:
        return 10
    return {_translate_chars_in_key(k, translation): _translate_chars(v, translation) for (k, v) in six.iteritems(field)}

def escape_chars(field):
    if False:
        i = 10
        return i + 15
    if not isinstance(field, dict) and (not isinstance(field, list)):
        return field
    value = fast_deepcopy_dict(field)
    return _translate_chars(value, ESCAPE_TRANSLATION)

def unescape_chars(field):
    if False:
        print('Hello World!')
    if not isinstance(field, dict) and (not isinstance(field, list)):
        return field
    value = fast_deepcopy_dict(field)
    return _translate_chars(value, UNESCAPE_TRANSLATION)