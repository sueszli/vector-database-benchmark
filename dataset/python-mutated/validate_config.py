"""
Script for validating a config file against a a particular config schema.
"""
from __future__ import absolute_import
import os
import yaml
import six
from oslo_config import cfg
from st2common.config import do_register_cli_opts
from st2common.constants.system import VERSION_STRING
from st2common.constants.exit_codes import SUCCESS_EXIT_CODE
from st2common.constants.exit_codes import FAILURE_EXIT_CODE
from st2common.util.pack import validate_config_against_schema
__all__ = ['main']

def _do_register_cli_opts(opts, ignore_errors=False):
    if False:
        for i in range(10):
            print('nop')
    for opt in opts:
        try:
            cfg.CONF.register_cli_opt(opt)
        except:
            if not ignore_errors:
                raise

def _register_cli_opts():
    if False:
        print('Hello World!')
    cli_opts = [cfg.StrOpt('schema-path', default=None, required=True, help='Path to the config schema to use for validation.'), cfg.StrOpt('config-path', default=None, required=True, help='Path to the config file to validate.')]
    do_register_cli_opts(cli_opts)

def main():
    if False:
        while True:
            i = 10
    _register_cli_opts()
    cfg.CONF(args=None, version=VERSION_STRING)
    schema_path = os.path.abspath(cfg.CONF.schema_path)
    config_path = os.path.abspath(cfg.CONF.config_path)
    print('Validating config "%s" against schema in "%s"' % (config_path, schema_path))
    with open(schema_path, 'r') as fp:
        config_schema = yaml.safe_load(fp.read())
    with open(config_path, 'r') as fp:
        config_object = yaml.safe_load(fp.read())
    try:
        validate_config_against_schema(config_schema=config_schema, config_object=config_object, config_path=config_path)
    except Exception as e:
        print('Failed to validate pack config.\n%s' % six.text_type(e))
        return FAILURE_EXIT_CODE
    print('Config "%s" successfully validated against schema in %s.' % (config_path, schema_path))
    return SUCCESS_EXIT_CODE