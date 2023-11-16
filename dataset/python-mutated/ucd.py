"""
This file contains routines to verify the correctness of UCD strings.
"""
import re
from astropy.utils import data
__all__ = ['parse_ucd', 'check_ucd']

class UCDWords:
    """
    Manages a list of acceptable UCD words.

    Works by reading in a data file exactly as provided by IVOA.  This
    file resides in data/ucd1p-words.txt.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._primary = set()
        self._secondary = set()
        self._descriptions = {}
        self._capitalization = {}
        with data.get_pkg_data_fileobj('data/ucd1p-words.txt', encoding='ascii') as fd:
            for line in fd.readlines():
                if line.startswith('#'):
                    continue
                (type, name, descr) = (x.strip() for x in line.split('|'))
                name_lower = name.lower()
                if type in 'QPEVC':
                    self._primary.add(name_lower)
                if type in 'QSEVC':
                    self._secondary.add(name_lower)
                self._descriptions[name_lower] = descr
                self._capitalization[name_lower] = name

    def is_primary(self, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns True if *name* is a valid primary name.\n        '
        return name.lower() in self._primary

    def is_secondary(self, name):
        if False:
            while True:
                i = 10
        '\n        Returns True if *name* is a valid secondary name.\n        '
        return name.lower() in self._secondary

    def get_description(self, name):
        if False:
            print('Hello World!')
        '\n        Returns the official English description of the given UCD\n        *name*.\n        '
        return self._descriptions[name.lower()]

    def normalize_capitalization(self, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the standard capitalization form of the given name.\n        '
        return self._capitalization[name.lower()]
_ucd_singleton = None

def parse_ucd(ucd, check_controlled_vocabulary=False, has_colon=False):
    if False:
        i = 10
        return i + 15
    "\n    Parse the UCD into its component parts.\n\n    Parameters\n    ----------\n    ucd : str\n        The UCD string\n\n    check_controlled_vocabulary : bool, optional\n        If `True`, then each word in the UCD will be verified against\n        the UCD1+ controlled vocabulary, (as required by the VOTable\n        specification version 1.2), otherwise not.\n\n    has_colon : bool, optional\n        If `True`, the UCD may contain a colon (as defined in earlier\n        versions of the standard).\n\n    Returns\n    -------\n    parts : list\n        The result is a list of tuples of the form:\n\n            (*namespace*, *word*)\n\n        If no namespace was explicitly specified, *namespace* will be\n        returned as ``'ivoa'`` (i.e., the default namespace).\n\n    Raises\n    ------\n    ValueError\n        if *ucd* is invalid\n    "
    global _ucd_singleton
    if _ucd_singleton is None:
        _ucd_singleton = UCDWords()
    if has_colon:
        m = re.search('[^A-Za-z0-9_.:;\\-]', ucd)
    else:
        m = re.search('[^A-Za-z0-9_.;\\-]', ucd)
    if m is not None:
        raise ValueError(f"UCD has invalid character '{m.group(0)}' in '{ucd}'")
    word_component_re = '[A-Za-z0-9][A-Za-z0-9\\-_]*'
    word_re = f'{word_component_re}(\\.{word_component_re})*'
    parts = ucd.split(';')
    words = []
    for (i, word) in enumerate(parts):
        colon_count = word.count(':')
        if colon_count == 1:
            (ns, word) = word.split(':', 1)
            if not re.match(word_component_re, ns):
                raise ValueError(f"Invalid namespace '{ns}'")
            ns = ns.lower()
        elif colon_count > 1:
            raise ValueError(f"Too many colons in '{word}'")
        else:
            ns = 'ivoa'
        if not re.match(word_re, word):
            raise ValueError(f"Invalid word '{word}'")
        if ns == 'ivoa' and check_controlled_vocabulary:
            if i == 0:
                if not _ucd_singleton.is_primary(word):
                    if _ucd_singleton.is_secondary(word):
                        raise ValueError(f"Secondary word '{word}' is not valid as a primary word")
                    else:
                        raise ValueError(f"Unknown word '{word}'")
            elif not _ucd_singleton.is_secondary(word):
                if _ucd_singleton.is_primary(word):
                    raise ValueError(f"Primary word '{word}' is not valid as a secondary word")
                else:
                    raise ValueError(f"Unknown word '{word}'")
        try:
            normalized_word = _ucd_singleton.normalize_capitalization(word)
        except KeyError:
            normalized_word = word
        words.append((ns, normalized_word))
    return words

def check_ucd(ucd, check_controlled_vocabulary=False, has_colon=False):
    if False:
        while True:
            i = 10
    '\n    Returns False if *ucd* is not a valid `unified content descriptor`_.\n\n    Parameters\n    ----------\n    ucd : str\n        The UCD string\n\n    check_controlled_vocabulary : bool, optional\n        If `True`, then each word in the UCD will be verified against\n        the UCD1+ controlled vocabulary, (as required by the VOTable\n        specification version 1.2), otherwise not.\n\n    has_colon : bool, optional\n        If `True`, the UCD may contain a colon (as defined in earlier\n        versions of the standard).\n\n    Returns\n    -------\n    valid : bool\n    '
    if ucd is None:
        return True
    try:
        parse_ucd(ucd, check_controlled_vocabulary=check_controlled_vocabulary, has_colon=has_colon)
    except ValueError:
        return False
    return True