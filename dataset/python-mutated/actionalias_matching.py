from __future__ import absolute_import
import six
from mongoengine.queryset.visitor import Q
from st2common.exceptions.content import ParseException
from st2common.exceptions.actionalias import ActionAliasAmbiguityException
from st2common.persistence.actionalias import ActionAlias
from st2common.models.utils.action_alias_utils import extract_parameters
__all__ = ['list_format_strings_from_aliases', 'normalise_alias_format_string', 'match_command_to_alias', 'get_matching_alias']

def list_format_strings_from_aliases(aliases, match_multiple=False):
    if False:
        i = 10
        return i + 15
    '\n    List patterns from a collection of alias objects\n\n    :param aliases: The list of aliases\n    :type  aliases: ``list`` of :class:`st2common.models.api.action.ActionAliasAPI`\n\n    :return: A description of potential execution patterns in a list of aliases.\n    :rtype: ``list`` of ``list``\n    '
    patterns = []
    for alias in aliases:
        for format_ in alias.formats:
            (display, representations, _match_multiple) = normalise_alias_format_string(format_)
            if display and len(representations) == 0:
                patterns.append({'alias': alias, 'format': format_, 'display': display, 'representation': ''})
            else:
                patterns.extend([{'alias': alias, 'format': format_, 'display': display, 'representation': representation, 'match_multiple': _match_multiple} for representation in representations])
    return patterns

def normalise_alias_format_string(alias_format):
    if False:
        print('Hello World!')
    '\n    StackStorm action aliases come in two forms;\n        1. A string holding the format, which is also used as the help string.\n        2. A dictionary containing "display" and/or "representation" keys.\n           "representation": a list of numerous alias format "representation(s)"\n           "display": a help string to be displayed.\n    This function processes both forms and returns a standardized form.\n\n    :param alias_format: The alias format\n    :type  alias_format: ``str`` or ``dict``\n\n    :return: The representation of the alias\n    :rtype: ``tuple`` of (``str``, ``str``)\n    '
    display = None
    representation = []
    match_multiple = False
    if isinstance(alias_format, six.string_types):
        display = alias_format
        representation.append(alias_format)
    elif isinstance(alias_format, dict):
        display = alias_format.get('display')
        representation = alias_format.get('representation') or []
        if isinstance(representation, six.string_types):
            representation = [representation]
        match_multiple = alias_format.get('match_multiple', match_multiple)
    else:
        raise TypeError("alias_format '%s' is neither a dictionary or string type." % repr(alias_format))
    return (display, representation, match_multiple)

def match_command_to_alias(command, aliases, match_multiple=False):
    if False:
        print('Hello World!')
    '\n    Match the text against an action and return the action reference.\n    '
    results = []
    for alias in aliases:
        formats = list_format_strings_from_aliases([alias], match_multiple)
        for format_ in formats:
            try:
                extract_parameters(format_str=format_['representation'], param_stream=command)
            except ParseException:
                continue
            results.append(format_)
    return results

def get_matching_alias(command):
    if False:
        i = 10
        return i + 15
    '\n    Find a matching ActionAliasDB object (if any) for the provided command.\n    '
    action_alias_dbs = ActionAlias.query(Q(formats__match_multiple=None) | Q(formats__match_multiple=False), enabled=True)
    matches = match_command_to_alias(command=command, aliases=action_alias_dbs)
    if len(matches) > 1:
        raise ActionAliasAmbiguityException("Command '%s' matched more than 1 pattern" % command, matches=matches, command=command)
    elif len(matches) == 0:
        match_multiple_action_alias_dbs = ActionAlias.query(formats__match_multiple=True, enabled=True)
        matches = match_command_to_alias(command=command, aliases=match_multiple_action_alias_dbs, match_multiple=True)
        if len(matches) > 1:
            raise ActionAliasAmbiguityException("Command '%s' matched more than 1 (multi) pattern" % command, matches=matches, command=command)
        if len(matches) == 0:
            raise ActionAliasAmbiguityException("Command '%s' matched no patterns" % command, matches=[], command=command)
    return matches[0]