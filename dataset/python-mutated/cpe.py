"""This submodule contains functions to manipulate CPE values for
documents from the active (nmap & view) purposes.

"""
from typing import Any, Dict, List
from ivre.types import CpeDict
from ivre.utils import LOGGER

def cpe2dict(cpe_str: str) -> CpeDict:
    if False:
        i = 10
        return i + 15
    'Helper function to parse CPEs. This is a very partial/simple parser.\n\n    Raises:\n        ValueError if the cpe string is not parsable.\n\n    '
    if not cpe_str.startswith('cpe:/'):
        raise ValueError('invalid cpe format (%s)\n' % cpe_str)
    cpe_body = cpe_str[5:]
    parts = cpe_body.split(':', 3)
    nparts = len(parts)
    if nparts < 2:
        raise ValueError('invalid cpe format (%s)\n' % cpe_str)
    cpe_type = parts[0]
    cpe_vend = parts[1]
    cpe_prod = parts[2] if nparts > 2 else ''
    cpe_vers = parts[3] if nparts > 3 else ''
    ret: CpeDict = {'type': cpe_type, 'vendor': cpe_vend, 'product': cpe_prod, 'version': cpe_vers}
    return ret

def add_cpe_values(hostrec: Dict[str, Any], path: str, cpe_values: List[str]) -> None:
    if False:
        return 10
    'Add CPE values (`cpe_values`) to the `hostrec` at the given `path`.\n\n    CPEs are indexed in a dictionary to agglomerate origins, but this dict\n    is replaced with its values() in ._pre_addhost() or in\n    .store_scan_json_zgrab(), or in the function that calls\n    add_cpe_values(), depending on the context.\n\n    '
    cpes = hostrec.setdefault('cpes', {})
    for cpe in cpe_values:
        if cpe not in cpes:
            try:
                cpeobj = cpe2dict(cpe)
            except ValueError:
                LOGGER.warning('Invalid cpe format (%s)', cpe)
                continue
            cpes[cpe] = cpeobj
        else:
            cpeobj = cpes[cpe]
        cpeobj.setdefault('origins', set()).add(path)