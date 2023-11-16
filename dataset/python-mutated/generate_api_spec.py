"""
A script generates final openapi.yaml file based on openapi.yaml.j2 Jinja
template file.
"""
from __future__ import absolute_import
from st2common import config
from st2common import log as logging
from st2common.util import spec_loader
from st2common.script_setup import setup as common_setup
from st2common.script_setup import teardown as common_teardown
__all__ = ['main']
LOG = logging.getLogger(__name__)
SPEC_HEADER = '# NOTE: This file is auto-generated - DO NOT EDIT MANUALLY\n# Edit st2common/st2common/openapi.yaml.j2 and then run\n# make .generate-api-spec\n# to generate the final spec file\n'

def setup():
    if False:
        for i in range(10):
            print('nop')
    common_setup(config=config, setup_db=False, register_mq_exchanges=False)

def generate_spec():
    if False:
        i = 10
        return i + 15
    spec_string = spec_loader.generate_spec('st2common', 'openapi.yaml.j2')
    print(SPEC_HEADER.rstrip())
    print(spec_string)

def teartown():
    if False:
        print('Hello World!')
    common_teardown()

def main():
    if False:
        for i in range(10):
            print('nop')
    setup()
    try:
        generate_spec()
        ret = 0
    except Exception:
        LOG.error('Failed to generate openapi.yaml file', exc_info=True)
        ret = 1
    finally:
        teartown()
    return ret