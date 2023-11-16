"""Data previewer functions

Functions and data structures that are needed for the ckan data preview.
"""
from __future__ import annotations
import logging
from typing import Any, Iterable, Optional
from urllib.parse import urlparse
import ckan.plugins as p
from ckan import logic
from ckan.common import _, config
from ckan.types import Context
log = logging.getLogger(__name__)
DEFAULT_RESOURCE_VIEW_TYPES = ['image_view', 'datatables_view']

def res_format(resource: dict[str, Any]) -> Optional[str]:
    if False:
        i = 10
        return i + 15
    ' The assumed resource format in lower case. '
    if not resource['url']:
        return None
    return (resource['format'] or resource['url'].split('.')[-1]).lower()

def compare_domains(urls: Iterable[str]) -> bool:
    if False:
        while True:
            i = 10
    ' Return True if the domains of the provided urls are the same.\n    '
    first_domain = None
    for url in urls:
        try:
            if not urlparse(url).scheme and (not url.startswith('/')):
                url = '//' + url
            parsed = urlparse(url.lower(), 'http')
            domain = (parsed.scheme, parsed.hostname, parsed.port)
        except ValueError:
            return False
        if not first_domain:
            first_domain = domain
            continue
        if first_domain != domain:
            return False
    return True

def on_same_domain(data_dict: dict[str, Any]) -> bool:
    if False:
        print('Hello World!')
    ckan_url = config.get('ckan.site_url')
    resource_url = data_dict['resource']['url']
    return compare_domains([ckan_url, resource_url])

def get_view_plugin(view_type: Optional[str]) -> Optional[p.IResourceView]:
    if False:
        return 10
    '\n    Returns the IResourceView plugin associated with the given view_type.\n    '
    for plugin in p.PluginImplementations(p.IResourceView):
        info = plugin.info()
        name = info.get('name')
        if name == view_type:
            return plugin
    return None

def get_view_plugins(view_types: Iterable[str]) -> list[p.IResourceView]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a list of the view plugins associated with the given view_types.\n    '
    view_plugins = []
    for view_type in view_types:
        view_plugin = get_view_plugin(view_type)
        if view_plugin:
            view_plugins.append(view_plugin)
    return view_plugins

def get_allowed_view_plugins(data_dict: dict[str, Any]) -> list[p.IResourceView]:
    if False:
        i = 10
        return i + 15
    '\n    Returns a list of view plugins that work against the resource provided\n\n    The ``data_dict`` contains: ``resource`` and ``package``.\n    '
    can_view = []
    for plugin in p.PluginImplementations(p.IResourceView):
        plugin_info = plugin.info()
        if plugin_info.get('always_available', False) or plugin.can_view(data_dict):
            can_view.append(plugin)
    return can_view

def get_default_view_plugins(get_datastore_views: bool=False) -> list[p.IResourceView]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the list of view plugins to be created by default on new resources\n\n    The default view types are defined via the `ckan.views.default_views`\n    configuration option. If this is not set (as opposed to empty, which means\n    no default views), the value of DEFAULT_RESOURCE_VIEW_TYPES is used to\n    look up the plugins.\n\n    If get_datastore_views is False, only the ones not requiring data to be in\n    the DataStore are returned, and if True, only the ones requiring it are.\n\n    To flag a view plugin as requiring the DataStore, it must have the\n    `requires_datastore` key set to True in the dict returned by its `info()`\n    method.\n\n    Returns a list of IResourceView plugins\n    '
    default_view_types = config.get('ckan.views.default_views')
    default_view_plugins = []
    for view_type in default_view_types:
        view_plugin = get_view_plugin(view_type)
        if not view_plugin:
            log.warning('Plugin for view {0} could not be found'.format(view_type))
            continue
        info = view_plugin.info()
        plugin_requires_datastore = info.get('requires_datastore', False)
        if plugin_requires_datastore == get_datastore_views:
            default_view_plugins.append(view_plugin)
    return default_view_plugins

def add_views_to_resource(context: Context, resource_dict: dict[str, Any], dataset_dict: Optional[dict[str, Any]]=None, view_types: Optional[list[str]]=None, create_datastore_views: bool=False) -> list[dict[str, Any]]:
    if False:
        print('Hello World!')
    "\n    Creates the provided views (if necessary) on the provided resource\n\n    Views to create are provided as a list of ``view_types``. If no types are\n    provided, the default views defined in the ``ckan.views.default_views``\n    will be created.\n\n    The function will get the plugins for the default views defined in\n    the configuration, and if some were found the `can_view` method of\n    each one of them will be called to determine if a resource view should\n    be created.\n\n    Resource views extensions get the resource dict and the parent dataset\n    dict. If the latter is not provided, `package_show` is called to get it.\n\n    By default only view plugins that don't require the resource data to be in\n    the DataStore are called. This is only relevant when the default view\n    plugins are used, not when explicitly passing view types. See\n    :py:func:`ckan.logic.action.create.package_create_default_resource_views.``\n    for details on the ``create_datastore_views`` parameter.\n\n    Returns a list of resource views created (empty if none were created)\n    "
    if not dataset_dict:
        dataset_dict = logic.get_action('package_show')(context, {'id': resource_dict['package_id']})
    if not view_types:
        view_plugins = get_default_view_plugins(create_datastore_views)
    else:
        view_plugins = get_view_plugins(view_types)
    if not view_plugins:
        return []
    existing_views = logic.get_action('resource_view_list')(context, {'id': resource_dict['id']})
    existing_view_types = [v['view_type'] for v in existing_views] if existing_views else []
    created_views = []
    for view_plugin in view_plugins:
        view_info = view_plugin.info()
        if view_info['name'] in existing_view_types:
            continue
        if view_plugin.can_view({'resource': resource_dict, 'package': dataset_dict}):
            view = {'resource_id': resource_dict['id'], 'view_type': view_info['name'], 'title': view_info.get('default_title', _('View')), 'description': view_info.get('default_description', '')}
            view_dict = logic.get_action('resource_view_create')(context, view)
            created_views.append(view_dict)
    return created_views

def add_views_to_dataset_resources(context: Context, dataset_dict: dict[str, Any], view_types: Optional[list[str]]=None, create_datastore_views: bool=False) -> list[dict[str, Any]]:
    if False:
        while True:
            i = 10
    "\n    Creates the provided views on all resources of the provided dataset\n\n    Views to create are provided as a list of ``view_types``. If no types are\n    provided, the default views defined in the ``ckan.views.default_views``\n    will be created. Note that in both cases only these views that can render\n    the resource will be created (ie its view plugin ``can_view`` method\n    returns True.\n\n    By default only view plugins that don't require the resource data to be in\n    the DataStore are called. This is only relevant when the default view\n    plugins are used, not when explicitly passing view types. See\n    :py:func:`ckan.logic.action.create.package_create_default_resource_views.``\n    for details on the ``create_datastore_views`` parameter.\n\n    Returns a list of resource views created (empty if none were created)\n    "
    created_views = []
    for resource_dict in dataset_dict.get('resources', []):
        new_views = add_views_to_resource(context, resource_dict, dataset_dict, view_types, create_datastore_views)
        created_views.extend(new_views)
    return created_views