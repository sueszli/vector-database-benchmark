"""Schema validation of ansible-core's ansible_builtin_runtime.yml and collection's meta/runtime.yml"""
from __future__ import annotations
import datetime
import os
import re
import sys
from functools import partial
import yaml
from voluptuous import All, Any, MultipleInvalid, PREVENT_EXTRA
from voluptuous import Required, Schema, Invalid
from voluptuous.humanize import humanize_error
from ansible.module_utils.compat.version import StrictVersion, LooseVersion
from ansible.module_utils.six import string_types
from ansible.utils.collection_loader import AnsibleCollectionRef
from ansible.utils.version import SemanticVersion

def fqcr(value):
    if False:
        return 10
    'Validate a FQCR.'
    if not isinstance(value, string_types):
        raise Invalid('Must be a string that is a FQCR')
    if not AnsibleCollectionRef.is_valid_fqcr(value):
        raise Invalid('Must be a FQCR')
    return value

def isodate(value, check_deprecation_date=False, is_tombstone=False):
    if False:
        while True:
            i = 10
    'Validate a datetime.date or ISO 8601 date string.'
    if isinstance(value, datetime.date):
        removal_date = value
    else:
        msg = 'Expected ISO 8601 date string (YYYY-MM-DD), or YAML date'
        if not isinstance(value, string_types):
            raise Invalid(msg)
        if not re.match('^[0-9]{4}-[0-9]{2}-[0-9]{2}$', value):
            raise Invalid(msg)
        try:
            removal_date = datetime.datetime.strptime(value, '%Y-%m-%d').date()
        except ValueError:
            raise Invalid(msg)
    today = datetime.date.today()
    if is_tombstone:
        if today < removal_date:
            raise Invalid('The tombstone removal_date (%s) must not be after today (%s)' % (removal_date, today))
    elif check_deprecation_date and today > removal_date:
        raise Invalid('The deprecation removal_date (%s) must be after today (%s)' % (removal_date, today))
    return value

def removal_version(value, is_ansible, current_version=None, is_tombstone=False):
    if False:
        print('Hello World!')
    'Validate a removal version string.'
    msg = 'Removal version must be a string' if is_ansible else 'Removal version must be a semantic version (https://semver.org/)'
    if not isinstance(value, string_types):
        raise Invalid(msg)
    try:
        if is_ansible:
            version = StrictVersion()
            version.parse(value)
            version = LooseVersion(value)
        else:
            version = SemanticVersion()
            version.parse(value)
            if version.major != 0 and (version.minor != 0 or version.patch != 0):
                raise Invalid('removal_version (%r) must be a major release, not a minor or patch release (see specification at https://semver.org/)' % (value,))
        if current_version is not None:
            if is_tombstone:
                if version > current_version:
                    raise Invalid('The tombstone removal_version (%r) must not be after the current version (%s)' % (value, current_version))
            elif version <= current_version:
                raise Invalid('The deprecation removal_version (%r) must be after the current version (%s)' % (value, current_version))
    except ValueError:
        raise Invalid(msg)
    return value

def any_value(value):
    if False:
        while True:
            i = 10
    'Accepts anything.'
    return value

def get_ansible_version():
    if False:
        for i in range(10):
            print('nop')
    'Return current ansible-core version'
    from ansible.release import __version__
    return LooseVersion('.'.join(__version__.split('.')[:3]))

def get_collection_version():
    if False:
        for i in range(10):
            print('nop')
    'Return current collection version, or None if it is not available'
    import importlib.util
    collection_detail_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'tools', 'collection_detail.py')
    collection_detail_spec = importlib.util.spec_from_file_location('collection_detail', collection_detail_path)
    collection_detail = importlib.util.module_from_spec(collection_detail_spec)
    sys.modules['collection_detail'] = collection_detail
    collection_detail_spec.loader.exec_module(collection_detail)
    try:
        result = collection_detail.read_manifest_json('.') or collection_detail.read_galaxy_yml('.')
        return SemanticVersion(result['version'])
    except Exception:
        return None

def validate_metadata_file(path, is_ansible, check_deprecation_dates=False):
    if False:
        while True:
            i = 10
    'Validate explicit runtime metadata file'
    try:
        with open(path, 'r', encoding='utf-8') as f_path:
            routing = yaml.safe_load(f_path)
    except yaml.error.MarkedYAMLError as ex:
        print('%s:%d:%d: YAML load failed: %s' % (path, ex.context_mark.line + 1 if ex.context_mark else 0, ex.context_mark.column + 1 if ex.context_mark else 0, re.sub('\\s+', ' ', str(ex))))
        return
    except Exception as ex:
        print('%s:%d:%d: YAML load failed: %s' % (path, 0, 0, re.sub('\\s+', ' ', str(ex))))
        return
    if is_ansible:
        current_version = get_ansible_version()
    else:
        current_version = get_collection_version()
    avoid_additional_data = Schema(Any({Required('removal_version'): any_value, 'warning_text': any_value}, {Required('removal_date'): any_value, 'warning_text': any_value}), extra=PREVENT_EXTRA)
    deprecation_schema = All(Schema({'removal_version': partial(removal_version, is_ansible=is_ansible, current_version=current_version), 'removal_date': partial(isodate, check_deprecation_date=check_deprecation_dates), 'warning_text': Any(*string_types)}), avoid_additional_data)
    tombstoning_schema = All(Schema({'removal_version': partial(removal_version, is_ansible=is_ansible, current_version=current_version, is_tombstone=True), 'removal_date': partial(isodate, is_tombstone=True), 'warning_text': Any(*string_types)}), avoid_additional_data)
    plugin_routing_schema = Any(Schema({'deprecation': Any(deprecation_schema), 'tombstone': Any(tombstoning_schema), 'redirect': fqcr}, extra=PREVENT_EXTRA))
    plugin_routing_schema_mu = Any(Schema({'deprecation': Any(deprecation_schema), 'tombstone': Any(tombstoning_schema), 'redirect': Any(*string_types)}, extra=PREVENT_EXTRA))
    list_dict_plugin_routing_schema = [{str_type: plugin_routing_schema} for str_type in string_types]
    list_dict_plugin_routing_schema_mu = [{str_type: plugin_routing_schema_mu} for str_type in string_types]
    plugin_schema = Schema({'action': Any(None, *list_dict_plugin_routing_schema), 'become': Any(None, *list_dict_plugin_routing_schema), 'cache': Any(None, *list_dict_plugin_routing_schema), 'callback': Any(None, *list_dict_plugin_routing_schema), 'cliconf': Any(None, *list_dict_plugin_routing_schema), 'connection': Any(None, *list_dict_plugin_routing_schema), 'doc_fragments': Any(None, *list_dict_plugin_routing_schema), 'filter': Any(None, *list_dict_plugin_routing_schema), 'httpapi': Any(None, *list_dict_plugin_routing_schema), 'inventory': Any(None, *list_dict_plugin_routing_schema), 'lookup': Any(None, *list_dict_plugin_routing_schema), 'module_utils': Any(None, *list_dict_plugin_routing_schema_mu), 'modules': Any(None, *list_dict_plugin_routing_schema), 'netconf': Any(None, *list_dict_plugin_routing_schema), 'shell': Any(None, *list_dict_plugin_routing_schema), 'strategy': Any(None, *list_dict_plugin_routing_schema), 'terminal': Any(None, *list_dict_plugin_routing_schema), 'test': Any(None, *list_dict_plugin_routing_schema), 'vars': Any(None, *list_dict_plugin_routing_schema)}, extra=PREVENT_EXTRA)
    import_redirection_schema = Any(Schema({'redirect': Any(*string_types)}, extra=PREVENT_EXTRA))
    list_dict_import_redirection_schema = [{str_type: import_redirection_schema} for str_type in string_types]
    schema = Schema({'plugin_routing': Any(plugin_schema), 'import_redirection': Any(None, *list_dict_import_redirection_schema), 'requires_ansible': Any(*string_types), 'action_groups': dict}, extra=PREVENT_EXTRA)
    try:
        schema(routing)
    except MultipleInvalid as ex:
        for error in ex.errors:
            print('%s:%d:%d: %s' % (path, 0, 0, humanize_error(routing, error)))

def main():
    if False:
        for i in range(10):
            print('nop')
    'Main entry point.'
    paths = sys.argv[1:] or sys.stdin.read().splitlines()
    collection_legacy_file = 'meta/routing.yml'
    collection_runtime_file = 'meta/runtime.yml'
    check_deprecation_dates = False
    for path in paths:
        if path == collection_legacy_file:
            print('%s:%d:%d: %s' % (path, 0, 0, "Should be called '%s'" % collection_runtime_file))
            continue
        validate_metadata_file(path, is_ansible=path not in (collection_legacy_file, collection_runtime_file), check_deprecation_dates=check_deprecation_dates)
if __name__ == '__main__':
    main()