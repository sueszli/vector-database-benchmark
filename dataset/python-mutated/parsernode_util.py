"""ParserNode utils"""
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Tuple
from certbot_apache._internal.interfaces import ParserNode

def validate_kwargs(kwargs: Dict[str, Any], required_names: Iterable[str]) -> Dict[str, Any]:
    if False:
        return 10
    '\n    Ensures that the kwargs dict has all the expected values. This function modifies\n    the kwargs dictionary, and hence the returned dictionary should be used instead\n    in the caller function instead of the original kwargs.\n\n    :param dict kwargs: Dictionary of keyword arguments to validate.\n    :param list required_names: List of required parameter names.\n    '
    validated_kwargs: Dict[str, Any] = {}
    for name in required_names:
        try:
            validated_kwargs[name] = kwargs.pop(name)
        except KeyError:
            raise TypeError('Required keyword argument: {} undefined.'.format(name))
    if kwargs:
        unknown = ', '.join(kwargs.keys())
        raise TypeError('Unknown keyword argument(s): {}'.format(unknown))
    return validated_kwargs

def parsernode_kwargs(kwargs: Dict[str, Any]) -> Tuple[Optional[ParserNode], bool, Optional[str], Dict[str, Any]]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Validates keyword arguments for ParserNode. This function modifies the kwargs\n    dictionary, and hence the returned dictionary should be used instead in the\n    caller function instead of the original kwargs.\n\n    If metadata is provided, the otherwise required argument "filepath" may be\n    omitted if the implementation is able to extract its value from the metadata.\n    This usecase is handled within this function. Filepath defaults to None.\n\n    :param dict kwargs: Keyword argument dictionary to validate.\n\n    :returns: Tuple of validated and prepared arguments.\n    '
    if 'metadata' in kwargs:
        kwargs.setdefault('filepath', None)
    kwargs.setdefault('dirty', False)
    kwargs.setdefault('metadata', {})
    kwargs = validate_kwargs(kwargs, ['ancestor', 'dirty', 'filepath', 'metadata'])
    return (kwargs['ancestor'], kwargs['dirty'], kwargs['filepath'], kwargs['metadata'])

def commentnode_kwargs(kwargs: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, str]]:
    if False:
        while True:
            i = 10
    '\n    Validates keyword arguments for CommentNode and sets the default values for\n    optional kwargs. This function modifies the kwargs dictionary, and hence the\n    returned dictionary should be used instead in the caller function instead of\n    the original kwargs.\n\n    If metadata is provided, the otherwise required argument "comment" may be\n    omitted if the implementation is able to extract its value from the metadata.\n    This usecase is handled within this function.\n\n    :param dict kwargs: Keyword argument dictionary to validate.\n\n    :returns: Tuple of validated and prepared arguments and ParserNode kwargs.\n    '
    if 'metadata' in kwargs:
        kwargs.setdefault('comment', None)
        kwargs.setdefault('filepath', None)
    kwargs.setdefault('dirty', False)
    kwargs.setdefault('metadata', {})
    kwargs = validate_kwargs(kwargs, ['ancestor', 'dirty', 'filepath', 'comment', 'metadata'])
    comment = kwargs.pop('comment')
    return (comment, kwargs)

def directivenode_kwargs(kwargs: Dict[str, Any]) -> Tuple[Optional[str], Tuple[str, ...], bool, Dict[str, Any]]:
    if False:
        while True:
            i = 10
    '\n    Validates keyword arguments for DirectiveNode and BlockNode and sets the\n    default values for optional kwargs. This function modifies the kwargs\n    dictionary, and hence the returned dictionary should be used instead in the\n    caller function instead of the original kwargs.\n\n    If metadata is provided, the otherwise required argument "name" may be\n    omitted if the implementation is able to extract its value from the metadata.\n    This usecase is handled within this function.\n\n    :param dict kwargs: Keyword argument dictionary to validate.\n\n    :returns: Tuple of validated and prepared arguments and ParserNode kwargs.\n    '
    if 'metadata' in kwargs:
        kwargs.setdefault('name', None)
        kwargs.setdefault('filepath', None)
    kwargs.setdefault('dirty', False)
    kwargs.setdefault('enabled', True)
    kwargs.setdefault('parameters', ())
    kwargs.setdefault('metadata', {})
    kwargs = validate_kwargs(kwargs, ['ancestor', 'dirty', 'filepath', 'name', 'parameters', 'enabled', 'metadata'])
    name = kwargs.pop('name')
    parameters = kwargs.pop('parameters')
    enabled = kwargs.pop('enabled')
    return (name, parameters, enabled, kwargs)