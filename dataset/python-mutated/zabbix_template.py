"""
.. versionadded:: 2017.7.0

Management of Zabbix Template object over Zabbix API.

:codeauthor: Jakub Sliva <jakub.sliva@ultimum.io>
"""
import json
import logging
from salt.exceptions import SaltException
log = logging.getLogger(__name__)
__deprecated__ = (3009, 'zabbix', 'https://github.com/salt-extensions/saltext-zabbix')
TEMPLATE_RELATIONS = ['groups', 'hosts', 'macros']
TEMPLATE_COMPONENT_ORDER = ('applications', 'items', 'gitems', 'graphs', 'screens', 'httpTests', 'triggers', 'discoveries')
DISCOVERYRULE_COMPONENT_ORDER = ('itemprototypes', 'triggerprototypes', 'graphprototypes', 'hostprototypes')
TEMPLATE_COMPONENT_DEF = {'applications': {'qtype': 'application', 'qidname': 'applicationid', 'qselectpid': 'templateids', 'ptype': 'template', 'pid': 'templateid', 'pid_ref_name': 'hostid', 'res_id_name': 'applicationids', 'output': {'output': 'extend', 'templated': 'true'}, 'inherited': 'inherited', 'adjust': True, 'filter': 'name', 'ro_attrs': ['applicationid', 'flags', 'templateids']}, 'items': {'qtype': 'item', 'qidname': 'itemid', 'qselectpid': 'templateids', 'ptype': 'template', 'pid': 'templateid', 'pid_ref_name': 'hostid', 'res_id_name': 'itemids', 'output': {'output': 'extend', 'selectApplications': 'extend', 'templated': 'true'}, 'inherited': 'inherited', 'adjust': False, 'filter': 'name', 'ro_attrs': ['itemid', 'error', 'flags', 'lastclock', 'lastns', 'lastvalue', 'prevvalue', 'state', 'templateid']}, 'triggers': {'qtype': 'trigger', 'qidname': 'triggerid', 'qselectpid': 'templateids', 'ptype': 'template', 'pid': 'templateid', 'pid_ref_name': None, 'res_id_name': 'triggerids', 'output': {'output': 'extend', 'selectDependencies': 'expand', 'templated': 'true', 'expandExpression': 'true'}, 'inherited': 'inherited', 'adjust': False, 'filter': 'description', 'ro_attrs': ['error', 'flags', 'lastchange', 'state', 'templateid', 'value']}, 'graphs': {'qtype': 'graph', 'qidname': 'graphid', 'qselectpid': 'templateids', 'ptype': 'template', 'pid': 'templateid', 'pid_ref_name': None, 'res_id_name': 'graphids', 'output': {'output': 'extend', 'selectGraphItems': 'extend', 'templated': 'true'}, 'inherited': 'inherited', 'adjust': False, 'filter': 'name', 'ro_attrs': ['graphid', 'flags', 'templateid']}, 'gitems': {'qtype': 'graphitem', 'qidname': 'itemid', 'qselectpid': 'graphids', 'ptype': 'graph', 'pid': 'graphid', 'pid_ref_name': None, 'res_id_name': None, 'output': {'output': 'extend'}, 'inherited': 'inherited', 'adjust': False, 'filter': 'name', 'ro_attrs': ['gitemid']}, 'screens': {'qtype': 'templatescreen', 'qidname': 'screenid', 'qselectpid': 'templateids', 'ptype': 'template', 'pid': 'templateid', 'pid_ref_name': 'templateid', 'res_id_name': 'screenids', 'output': {'output': 'extend', 'selectUsers': 'extend', 'selectUserGroups': 'extend', 'selectScreenItems': 'extend', 'noInheritance': 'true'}, 'inherited': 'noInheritance', 'adjust': False, 'filter': 'name', 'ro_attrs': ['screenid']}, 'discoveries': {'qtype': 'discoveryrule', 'qidname': 'itemid', 'qselectpid': 'templateids', 'ptype': 'template', 'pid': 'templateid', 'pid_ref_name': 'hostid', 'res_id_name': 'itemids', 'output': {'output': 'extend', 'selectFilter': 'extend', 'templated': 'true'}, 'inherited': 'inherited', 'adjust': False, 'filter': 'key_', 'ro_attrs': ['itemid', 'error', 'state', 'templateid']}, 'httpTests': {'qtype': 'httptest', 'qidname': 'httptestid', 'qselectpid': 'templateids', 'ptype': 'template', 'pid': 'templateid', 'pid_ref_name': 'hostid', 'res_id_name': 'httptestids', 'output': {'output': 'extend', 'selectSteps': 'extend', 'templated': 'true'}, 'inherited': 'inherited', 'adjust': False, 'filter': 'name', 'ro_attrs': ['httptestid', 'nextcheck', 'templateid']}, 'itemprototypes': {'qtype': 'itemprototype', 'qidname': 'itemid', 'qselectpid': 'discoveryids', 'ptype': 'discoveryrule', 'pid': 'itemid', 'pid_ref_name': 'ruleid', 'pid_ref_name2': 'hostid', 'res_id_name': 'itemids', 'output': {'output': 'extend', 'selectSteps': 'extend', 'selectApplications': 'extend', 'templated': 'true'}, 'adjust': False, 'inherited': 'inherited', 'filter': 'name', 'ro_attrs': ['itemid', 'templateid']}, 'triggerprototypes': {'qtype': 'triggerprototype', 'qidname': 'triggerid', 'qselectpid': 'discoveryids', 'ptype': 'discoveryrule', 'pid': 'itemid', 'pid_ref_name': None, 'res_id_name': 'triggerids', 'output': {'output': 'extend', 'selectTags': 'extend', 'selectDependencies': 'extend', 'templated': 'true', 'expandExpression': 'true'}, 'inherited': 'inherited', 'adjust': False, 'filter': 'description', 'ro_attrs': ['triggerid', 'templateid']}, 'graphprototypes': {'qtype': 'graphprototype', 'qidname': 'graphid', 'qselectpid': 'discoveryids', 'ptype': 'discoveryrule', 'pid': 'itemid', 'pid_ref_name': None, 'res_id_name': 'graphids', 'output': {'output': 'extend', 'selectGraphItems': 'extend', 'templated': 'true'}, 'inherited': 'inherited', 'adjust': False, 'filter': 'name', 'ro_attrs': ['graphid', 'templateid']}, 'hostprototypes': {'qtype': 'hostprototype', 'qidname': 'hostid', 'qselectpid': 'discoveryids', 'ptype': 'discoveryrule', 'pid': 'itemid', 'pid_ref_name': 'ruleid', 'res_id_name': 'hostids', 'output': {'output': 'extend', 'selectGroupLinks': 'expand', 'selectGroupPrototypes': 'expand', 'selectTemplates': 'expand'}, 'inherited': 'inherited', 'adjust': False, 'filter': 'host', 'ro_attrs': ['hostid', 'templateid']}}
CHANGE_STACK = []

def __virtual__():
    if False:
        return 10
    '\n    Only make these states available if Zabbix module and run_query function is available\n    and all 3rd party modules imported.\n    '
    if 'zabbix.run_query' in __salt__:
        return True
    return (False, 'Import zabbix or other needed modules failed.')

def _diff_and_merge_host_list(defined, existing):
    if False:
        for i in range(10):
            print('nop')
    '\n    If Zabbix template is to be updated then list of assigned hosts must be provided in all or nothing manner to prevent\n    some externally assigned hosts to be detached.\n\n    :param defined: list of hosts defined in sls\n    :param existing: list of hosts taken from live Zabbix\n    :return: list to be updated (combinated or empty list)\n    '
    try:
        defined_host_ids = {host['hostid'] for host in defined}
        existing_host_ids = {host['hostid'] for host in existing}
    except KeyError:
        raise SaltException('List of hosts in template not defined correctly.')
    diff = defined_host_ids - existing_host_ids
    return [{'hostid': str(hostid)} for hostid in diff | existing_host_ids] if diff else []

def _get_existing_template_c_list(component, parent_id, **kwargs):
    if False:
        return 10
    '\n    Make a list of given component type not inherited from other templates because Zabbix API returns only list of all\n    and list of inherited component items so we have to do a difference list.\n\n    :param component: Template component (application, item, etc...)\n    :param parent_id: ID of existing template the component is assigned to\n    :return List of non-inherited (own) components\n    '
    c_def = TEMPLATE_COMPONENT_DEF[component]
    q_params = dict(c_def['output'])
    q_params.update({c_def['qselectpid']: parent_id})
    existing_clist_all = __salt__['zabbix.run_query'](c_def['qtype'] + '.get', q_params, **kwargs)
    if c_def['inherited'] == 'inherited':
        q_params.update({c_def['inherited']: 'true'})
        existing_clist_inherited = __salt__['zabbix.run_query'](c_def['qtype'] + '.get', q_params, **kwargs)
    else:
        existing_clist_inherited = []
    if existing_clist_inherited:
        return [c_all for c_all in existing_clist_all if c_all not in existing_clist_inherited]
    return existing_clist_all

def _adjust_object_lists(obj):
    if False:
        for i in range(10):
            print('nop')
    '\n    For creation or update of object that have attribute which contains a list Zabbix awaits plain list of IDs while\n    querying Zabbix for same object returns list of dicts\n\n    :param obj: Zabbix object parameters\n    '
    for subcomp in TEMPLATE_COMPONENT_DEF:
        if subcomp in obj and TEMPLATE_COMPONENT_DEF[subcomp]['adjust']:
            obj[subcomp] = [item[TEMPLATE_COMPONENT_DEF[subcomp]['qidname']] for item in obj[subcomp]]

def _manage_component(component, parent_id, defined, existing, template_id=None, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Takes particular component list, compares it with existing, call appropriate API methods - create, update, delete.\n\n    :param component: component name\n    :param parent_id: ID of parent entity under which component should be created\n    :param defined: list of defined items of named component\n    :param existing: list of existing items of named component\n    :param template_id: In case that component need also template ID for creation (although parent_id is given?!?!?)\n    '
    zabbix_id_mapper = __salt__['zabbix.get_zabbix_id_mapper']()
    dry_run = __opts__['test']
    c_def = TEMPLATE_COMPONENT_DEF[component]
    compare_key = c_def['filter']
    defined_set = {item[compare_key] for item in defined}
    existing_set = {item[compare_key] for item in existing}
    create_set = defined_set - existing_set
    update_set = defined_set & existing_set
    delete_set = existing_set - defined_set
    create_list = [item for item in defined if item[compare_key] in create_set]
    for object_params in create_list:
        if parent_id:
            object_params.update({c_def['pid_ref_name']: parent_id})
        if 'pid_ref_name2' in c_def:
            object_params.update({c_def['pid_ref_name2']: template_id})
        _adjust_object_lists(object_params)
        if not dry_run:
            object_create = __salt__['zabbix.run_query'](c_def['qtype'] + '.create', object_params, **kwargs)
            if object_create:
                object_ids = object_create[c_def['res_id_name']]
                CHANGE_STACK.append({'component': component, 'action': 'create', 'params': object_params, c_def['filter']: object_params[c_def['filter']], 'object_id': object_ids})
        else:
            CHANGE_STACK.append({'component': component, 'action': 'create', 'params': object_params, 'object_id': 'CREATED ' + TEMPLATE_COMPONENT_DEF[component]['qtype'] + ' ID'})
    delete_list = [item for item in existing if item[compare_key] in delete_set]
    for object_del in delete_list:
        object_id_name = zabbix_id_mapper[c_def['qtype']]
        CHANGE_STACK.append({'component': component, 'action': 'delete', 'params': [object_del[object_id_name]]})
        if not dry_run:
            __salt__['zabbix.run_query'](c_def['qtype'] + '.delete', [object_del[object_id_name]], **kwargs)
    for object_name in update_set:
        ditem = next((item for item in defined if item[compare_key] == object_name), None)
        eitem = next((item for item in existing if item[compare_key] == object_name), None)
        diff_params = __salt__['zabbix.compare_params'](ditem, eitem, True)
        if diff_params['new']:
            diff_params['new'][zabbix_id_mapper[c_def['qtype']]] = eitem[zabbix_id_mapper[c_def['qtype']]]
            diff_params['old'][zabbix_id_mapper[c_def['qtype']]] = eitem[zabbix_id_mapper[c_def['qtype']]]
            _adjust_object_lists(diff_params['new'])
            _adjust_object_lists(diff_params['old'])
            CHANGE_STACK.append({'component': component, 'action': 'update', 'params': diff_params['new']})
            if not dry_run:
                __salt__['zabbix.run_query'](c_def['qtype'] + '.update', diff_params['new'], **kwargs)

def is_present(name, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Check if Zabbix Template already exists.\n\n    :param name: Zabbix Template name\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    .. code-block:: yaml\n\n        does_zabbix-template-exist:\n            zabbix_template.is_present:\n                - name: Template OS Linux\n    "
    ret = {'name': name, 'result': False, 'comment': '', 'changes': {}}
    try:
        object_id = __salt__['zabbix.get_object_id_by_params']('template', {'filter': {'name': name}}, **kwargs)
    except SaltException:
        object_id = False
    if not object_id:
        ret['result'] = False
        ret['comment'] = f'Zabbix Template "{name}" does not exist.'
    else:
        ret['result'] = True
        ret['comment'] = f'Zabbix Template "{name}" exists.'
    return ret

def present(name, params, static_host_list=True, **kwargs):
    if False:
        return 10
    '\n    Creates Zabbix Template object or if differs update it according defined parameters. See Zabbix API documentation.\n\n    Zabbix API version: >3.0\n\n    :param name: Zabbix Template name\n    :param params: Additional parameters according to Zabbix API documentation\n    :param static_host_list: If hosts assigned to the template are controlled\n        only by this state or can be also assigned externally\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module\'s docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module\'s docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module\'s docstring)\n\n    .. note::\n\n        If there is a need to get a value from current zabbix online (e.g. ids of host groups you want the template\n        to be associated with), put a dictionary with two keys "query_object" and "query_name" instead of the value.\n        In this example we want to create template named "Testing Template", assign it to hostgroup Templates,\n        link it to two ceph nodes and create a macro.\n\n    .. note::\n\n        IMPORTANT NOTE:\n        Objects (except for template name) are identified by name (or by other key in some exceptional cases)\n        so changing name of object means deleting old one and creating new one with new ID !!!\n\n    .. note::\n\n        NOT SUPPORTED FEATURES:\n            - linked templates\n            - trigger dependencies\n            - groups and group prototypes for host prototypes\n\n    SLS Example:\n\n    .. code-block:: yaml\n\n        zabbix-template-present:\n            zabbix_template.present:\n                - name: Testing Template\n                # Do not touch existing assigned hosts\n                # True will detach all other hosts than defined here\n                - static_host_list: False\n                - params:\n                    description: Template for Ceph nodes\n                    groups:\n                        # groups must already exist\n                        # template must be at least in one hostgroup\n                        - groupid:\n                            query_object: hostgroup\n                            query_name: Templates\n                    macros:\n                        - macro: "{$CEPH_CLUSTER_NAME}"\n                          value: ceph\n                    hosts:\n                        # hosts must already exist\n                        - hostid:\n                            query_object: host\n                            query_name: ceph-osd-01\n                        - hostid:\n                            query_object: host\n                            query_name: ceph-osd-02\n                    # templates:\n                    # Linked templates - not supported by state module but can be linked manually (will not be touched)\n\n                    applications:\n                        - name: Ceph OSD\n                    items:\n                        - name: Ceph OSD avg fill item\n                          key_: ceph.osd_avg_fill\n                          type: 2\n                          value_type: 0\n                          delay: 60\n                          units: \'%\'\n                          description: \'Average fill of OSD\'\n                          applications:\n                              - applicationid:\n                                  query_object: application\n                                  query_name: Ceph OSD\n                    triggers:\n                        - description: "Ceph OSD filled more that 90%"\n                          expression: "{{\'{\'}}Testing Template:ceph.osd_avg_fill.last(){{\'}\'}}>90"\n                          priority: 4\n                    discoveries:\n                        - name: Mounted filesystem discovery\n                          key_: vfs.fs.discovery\n                          type: 0\n                          delay: 60\n                          itemprototypes:\n                              - name: Free disk space on {{\'{#\'}}FSNAME}\n                                key_: vfs.fs.size[{{\'{#\'}}FSNAME},free]\n                                type: 0\n                                value_type: 3\n                                delay: 60\n                                applications:\n                                    - applicationid:\n                                        query_object: application\n                                        query_name: Ceph OSD\n                          triggerprototypes:\n                              - description: "Free disk space is less than 20% on volume {{\'{#\'}}FSNAME{{\'}\'}}"\n                                expression: "{{\'{\'}}Testing Template:vfs.fs.size[{{\'{#\'}}FSNAME},free].last(){{\'}\'}}<20"\n                    graphs:\n                        - name: Ceph OSD avg fill graph\n                          width: 900\n                          height: 200\n                          graphtype: 0\n                          gitems:\n                              - color: F63100\n                                itemid:\n                                  query_object: item\n                                  query_name: Ceph OSD avg fill item\n                    screens:\n                        - name: Ceph\n                          hsize: 1\n                          vsize: 1\n                          screenitems:\n                              - x: 0\n                                y: 0\n                                resourcetype: 0\n                                resourceid:\n                                    query_object: graph\n                                    query_name: Ceph OSD avg fill graph\n    '
    zabbix_id_mapper = __salt__['zabbix.get_zabbix_id_mapper']()
    dry_run = __opts__['test']
    ret = {'name': name, 'result': False, 'comment': '', 'changes': {}}
    params['host'] = name
    del CHANGE_STACK[:]
    template_definition = {}
    template_components = {}
    discovery_components = []
    for attr in params:
        if attr in TEMPLATE_COMPONENT_ORDER and str(attr) != 'discoveries':
            template_components[attr] = params[attr]
        elif str(attr) == 'discoveries':
            d_rules = []
            for d_rule in params[attr]:
                d_rule_components = {'query_pid': {'component': attr, 'filter_val': d_rule[TEMPLATE_COMPONENT_DEF[attr]['filter']]}}
                for proto_name in DISCOVERYRULE_COMPONENT_ORDER:
                    if proto_name in d_rule:
                        d_rule_components[proto_name] = d_rule[proto_name]
                        del d_rule[proto_name]
                discovery_components.append(d_rule_components)
                d_rules.append(d_rule)
            template_components[attr] = d_rules
        else:
            template_definition[attr] = params[attr]
    for attr in TEMPLATE_COMPONENT_ORDER:
        if attr not in template_components:
            template_components[attr] = []
    for attr in TEMPLATE_RELATIONS:
        template_definition[attr] = params[attr] if attr in params and params[attr] else []
    defined_obj = __salt__['zabbix.substitute_params'](template_definition, **kwargs)
    log.info('SUBSTITUTED template_definition: %s', str(json.dumps(defined_obj, indent=4)))
    tmpl_get = __salt__['zabbix.run_query']('template.get', {'output': 'extend', 'selectGroups': 'groupid', 'selectHosts': 'hostid', 'selectTemplates': 'templateid', 'selectMacros': 'extend', 'filter': {'host': name}}, **kwargs)
    log.info('TEMPLATE get result: %s', str(json.dumps(tmpl_get, indent=4)))
    existing_obj = __salt__['zabbix.substitute_params'](tmpl_get[0], **kwargs) if tmpl_get and len(tmpl_get) == 1 else False
    if existing_obj:
        template_id = existing_obj[zabbix_id_mapper['template']]
        if not static_host_list:
            defined_wo_hosts = defined_obj
            if 'hosts' in defined_obj:
                defined_hosts = defined_obj['hosts']
                del defined_wo_hosts['hosts']
            else:
                defined_hosts = []
            existing_wo_hosts = existing_obj
            if 'hosts' in existing_obj:
                existing_hosts = existing_obj['hosts']
                del existing_wo_hosts['hosts']
            else:
                existing_hosts = []
            hosts_list = _diff_and_merge_host_list(defined_hosts, existing_hosts)
            diff_params = __salt__['zabbix.compare_params'](defined_wo_hosts, existing_wo_hosts, True)
            if 'new' in diff_params and 'hosts' in diff_params['new'] or hosts_list:
                diff_params['new']['hosts'] = hosts_list
        else:
            diff_params = __salt__['zabbix.compare_params'](defined_obj, existing_obj, True)
        if diff_params['new']:
            diff_params['new'][zabbix_id_mapper['template']] = template_id
            diff_params['old'][zabbix_id_mapper['template']] = template_id
            log.info('TEMPLATE: update params: %s', str(json.dumps(diff_params, indent=4)))
            CHANGE_STACK.append({'component': 'template', 'action': 'update', 'params': diff_params['new']})
            if not dry_run:
                tmpl_update = __salt__['zabbix.run_query']('template.update', diff_params['new'], **kwargs)
                log.info('TEMPLATE update result: %s', str(tmpl_update))
    else:
        CHANGE_STACK.append({'component': 'template', 'action': 'create', 'params': defined_obj})
        if not dry_run:
            tmpl_create = __salt__['zabbix.run_query']('template.create', defined_obj, **kwargs)
            log.info('TEMPLATE create result: %s', str(tmpl_create))
            if tmpl_create:
                template_id = tmpl_create['templateids'][0]
    log.info('\n\ntemplate_components: %s', json.dumps(template_components, indent=4))
    log.info('\n\ndiscovery_components: %s', json.dumps(discovery_components, indent=4))
    log.info('\n\nCurrent CHANGE_STACK: %s', str(json.dumps(CHANGE_STACK, indent=4)))
    if existing_obj or not dry_run:
        for component in TEMPLATE_COMPONENT_ORDER:
            log.info('\n\n\n\n\nCOMPONENT: %s\n\n', str(json.dumps(component)))
            existing_c_list = _get_existing_template_c_list(component, template_id, **kwargs)
            existing_c_list_subs = __salt__['zabbix.substitute_params'](existing_c_list, **kwargs) if existing_c_list else []
            if component in template_components:
                defined_c_list_subs = __salt__['zabbix.substitute_params'](template_components[component], extend_params={TEMPLATE_COMPONENT_DEF[component]['qselectpid']: template_id}, filter_key=TEMPLATE_COMPONENT_DEF[component]['filter'], **kwargs)
            else:
                defined_c_list_subs = []
            _manage_component(component, template_id, defined_c_list_subs, existing_c_list_subs, **kwargs)
        log.info('\n\nCurrent CHANGE_STACK: %s', str(json.dumps(CHANGE_STACK, indent=4)))
        for d_rule_component in discovery_components:
            q_def = d_rule_component['query_pid']
            c_def = TEMPLATE_COMPONENT_DEF[q_def['component']]
            q_object = c_def['qtype']
            q_params = dict(c_def['output'])
            q_params.update({c_def['qselectpid']: template_id})
            q_params.update({'filter': {c_def['filter']: q_def['filter_val']}})
            parent_id = __salt__['zabbix.get_object_id_by_params'](q_object, q_params, **kwargs)
            for proto_name in DISCOVERYRULE_COMPONENT_ORDER:
                log.info('\n\n\n\n\nPROTOTYPE_NAME: %s\n\n', str(json.dumps(proto_name)))
                existing_p_list = _get_existing_template_c_list(proto_name, parent_id, **kwargs)
                existing_p_list_subs = __salt__['zabbix.substitute_params'](existing_p_list, **kwargs) if existing_p_list else []
                if proto_name in d_rule_component:
                    defined_p_list_subs = __salt__['zabbix.substitute_params'](d_rule_component[proto_name], extend_params={c_def['qselectpid']: template_id}, **kwargs)
                else:
                    defined_p_list_subs = []
                _manage_component(proto_name, parent_id, defined_p_list_subs, existing_p_list_subs, template_id=template_id, **kwargs)
    log.info('\n\nCurrent CHANGE_STACK: %s', str(json.dumps(CHANGE_STACK, indent=4)))
    if not CHANGE_STACK:
        ret['result'] = True
        ret['comment'] = 'Zabbix Template "{}" already exists and corresponds to a definition.'.format(name)
    else:
        tmpl_action = next((item for item in CHANGE_STACK if item['component'] == 'template' and item['action'] == 'create'), None)
        if tmpl_action:
            ret['result'] = True
            if dry_run:
                ret['comment'] = f'Zabbix Template "{name}" would be created.'
                ret['changes'] = {name: {'old': f'Zabbix Template "{name}" does not exist.', 'new': 'Zabbix Template "{}" would be created according definition.'.format(name)}}
            else:
                ret['comment'] = f'Zabbix Template "{name}" created.'
                ret['changes'] = {name: {'old': f'Zabbix Template "{name}" did not exist.', 'new': 'Zabbix Template "{}" created according definition.'.format(name)}}
        else:
            ret['result'] = True
            if dry_run:
                ret['comment'] = f'Zabbix Template "{name}" would be updated.'
                ret['changes'] = {name: {'old': f'Zabbix Template "{name}" differs.', 'new': 'Zabbix Template "{}" would be updated according definition.'.format(name)}}
            else:
                ret['comment'] = f'Zabbix Template "{name}" updated.'
                ret['changes'] = {name: {'old': f'Zabbix Template "{name}" differed.', 'new': 'Zabbix Template "{}" updated according definition.'.format(name)}}
    return ret

def absent(name, **kwargs):
    if False:
        print('Hello World!')
    "\n    Makes the Zabbix Template to be absent (either does not exist or delete it).\n\n    :param name: Zabbix Template name\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    .. code-block:: yaml\n\n        zabbix-template-absent:\n            zabbix_template.absent:\n                - name: Ceph OSD\n    "
    dry_run = __opts__['test']
    ret = {'name': name, 'result': False, 'comment': '', 'changes': {}}
    try:
        object_id = __salt__['zabbix.get_object_id_by_params']('template', {'filter': {'name': name}}, **kwargs)
    except SaltException:
        object_id = False
    if not object_id:
        ret['result'] = True
        ret['comment'] = f'Zabbix Template "{name}" does not exist.'
    elif dry_run:
        ret['result'] = True
        ret['comment'] = f'Zabbix Template "{name}" would be deleted.'
        ret['changes'] = {name: {'old': f'Zabbix Template "{name}" exists.', 'new': f'Zabbix Template "{name}" would be deleted.'}}
    else:
        tmpl_delete = __salt__['zabbix.run_query']('template.delete', [object_id], **kwargs)
        if tmpl_delete:
            ret['result'] = True
            ret['comment'] = f'Zabbix Template "{name}" deleted.'
            ret['changes'] = {name: {'old': f'Zabbix Template "{name}" existed.', 'new': f'Zabbix Template "{name}" deleted.'}}
    return ret