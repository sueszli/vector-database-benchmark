"""
Various XML-related utilities.
"""
from astropy.logger import log
from astropy.utils import data
from astropy.utils.xml import check as xml_check
from astropy.utils.xml import validate
from .exceptions import W02, W03, W04, W05, vo_warn, warn_or_raise
__all__ = ['check_id', 'fix_id', 'check_token', 'check_mime_content_type', 'check_anyuri', 'validate_schema']

def check_id(ID, name='ID', config=None, pos=None):
    if False:
        i = 10
        return i + 15
    '\n    Raises a `~astropy.io.votable.exceptions.VOTableSpecError` if *ID*\n    is not a valid XML ID_.\n\n    *name* is the name of the attribute being checked (used only for\n    error messages).\n    '
    if ID is not None and (not xml_check.check_id(ID)):
        warn_or_raise(W02, W02, (name, ID), config, pos)
        return False
    return True

def fix_id(ID, config=None, pos=None):
    if False:
        print('Hello World!')
    '\n    Given an arbitrary string, create one that can be used as an xml id.\n\n    This is rather simplistic at the moment, since it just replaces\n    non-valid characters with underscores.\n    '
    if ID is None:
        return None
    corrected = xml_check.fix_id(ID)
    if corrected != ID:
        vo_warn(W03, (ID, corrected), config, pos)
    return corrected
_token_regex = '(?![\\r\\l\\t ])[^\\r\\l\\t]*(?![\\r\\l\\t ])'

def check_token(token, attr_name, config=None, pos=None):
    if False:
        print('Hello World!')
    '\n    Raises a `ValueError` if *token* is not a valid XML token.\n\n    As defined by XML Schema Part 2.\n    '
    if token is not None and (not xml_check.check_token(token)):
        return False
    return True

def check_mime_content_type(content_type, config=None, pos=None):
    if False:
        print('Hello World!')
    '\n    Raises a `~astropy.io.votable.exceptions.VOTableSpecError` if\n    *content_type* is not a valid MIME content type.\n\n    As defined by RFC 2045 (syntactically, at least).\n    '
    if content_type is not None and (not xml_check.check_mime_content_type(content_type)):
        warn_or_raise(W04, W04, content_type, config, pos)
        return False
    return True

def check_anyuri(uri, config=None, pos=None):
    if False:
        return 10
    '\n    Raises a `~astropy.io.votable.exceptions.VOTableSpecError` if\n    *uri* is not a valid URI.\n\n    As defined in RFC 2396.\n    '
    if uri is not None and (not xml_check.check_anyuri(uri)):
        warn_or_raise(W05, W05, uri, config, pos)
        return False
    return True

def validate_schema(filename, version='1.1'):
    if False:
        print('Hello World!')
    '\n    Validates the given file against the appropriate VOTable schema.\n\n    Parameters\n    ----------\n    filename : str\n        The path to the XML file to validate\n\n    version : str, optional\n        The VOTABLE version to check, which must be a string "1.0",\n        "1.1", "1.2" or "1.3".  If it is not one of these,\n        version "1.1" is assumed.\n\n        For version "1.0", it is checked against a DTD, since that\n        version did not have an XML Schema.\n\n    Returns\n    -------\n    returncode, stdout, stderr : int, str, str\n        Returns the returncode from xmllint and the stdout and stderr\n        as strings\n    '
    if version not in ('1.0', '1.1', '1.2', '1.3'):
        log.info(f'{filename} has version {version}, using schema 1.1')
        version = '1.1'
    if version in ('1.1', '1.2', '1.3'):
        schema_path = data.get_pkg_data_filename(f'data/VOTable.v{version}.xsd')
    else:
        schema_path = data.get_pkg_data_filename('data/VOTable.dtd')
    return validate.validate_schema(filename, schema_path)