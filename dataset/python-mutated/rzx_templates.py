"""
Convenience function to load RZXGate based templates.
"""
from enum import Enum
from typing import List, Dict
from qiskit.circuit.library.templates import rzx

class RZXTemplateMap(Enum):
    """Mapping of instruction name to decomposition template."""
    ZZ1 = rzx.rzx_zz1()
    ZZ2 = rzx.rzx_zz2()
    ZZ3 = rzx.rzx_zz3()
    YZ = rzx.rzx_yz()
    XZ = rzx.rzx_xz()
    CY = rzx.rzx_cy()

def rzx_templates(template_list: List[str]=None) -> Dict:
    if False:
        i = 10
        return i + 15
    'Convenience function to get the cost_dict and templates for template matching.\n\n    Args:\n        template_list: List of instruction names.\n\n    Returns:\n        Decomposition templates and cost values.\n    '
    if template_list is None:
        template_list = ['zz1', 'zz2', 'zz3', 'yz', 'xz', 'cy']
    templates = [RZXTemplateMap[gate.upper()].value for gate in template_list]
    cost_dict = {'rzx': 0, 'cx': 6, 'rz': 0, 'sx': 1, 'p': 0, 'h': 1, 'rx': 1, 'ry': 1}
    rzx_dict = {'template_list': templates, 'user_cost_dict': cost_dict}
    return rzx_dict