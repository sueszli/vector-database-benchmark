from __future__ import absolute_import
import pkg_resources
import jinja2
import st2common.constants.pack
import st2common.constants.action
from st2common.rbac.types import PermissionType
from st2common.util import isotime
from st2common.util import yml as yaml_utils
__all__ = ['load_spec', 'generate_spec']
ARGUMENTS = {'DEFAULT_PACK_NAME': st2common.constants.pack.DEFAULT_PACK_NAME, 'LIVEACTION_STATUSES': st2common.constants.action.LIVEACTION_STATUSES, 'PERMISSION_TYPE': PermissionType, 'ISO8601_UTC_REGEX': isotime.ISO8601_UTC_REGEX}

def load_spec(module_name, spec_file):
    if False:
        for i in range(10):
            print('nop')
    spec_string = generate_spec(module_name, spec_file)
    return yaml_utils.unique_key_loader_safe_load(spec_string)

def generate_spec(module_name, spec_file):
    if False:
        for i in range(10):
            print('nop')
    spec_template = pkg_resources.resource_string(module_name, spec_file)
    if not isinstance(spec_template, str):
        spec_template = spec_template.decode()
    spec_string = jinja2.Template(spec_template).render(**ARGUMENTS)
    return spec_string