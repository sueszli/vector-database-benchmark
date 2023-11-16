"""
The vyos lldp_global fact class
It is in this file the configuration is collected from the device
for a given resource, parsed, and the facts tree is populated
based on the configuration.
"""
from __future__ import annotations
from re import findall, M
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.lldp_global.lldp_global import Lldp_globalArgs

class Lldp_globalFacts(object):
    """ The vyos lldp_global fact class
    """

    def __init__(self, module, subspec='config', options='options'):
        if False:
            for i in range(10):
                print('nop')
        self._module = module
        self.argument_spec = Lldp_globalArgs.argument_spec
        spec = deepcopy(self.argument_spec)
        if subspec:
            if options:
                facts_argument_spec = spec[subspec][options]
            else:
                facts_argument_spec = spec[subspec]
        else:
            facts_argument_spec = spec
        self.generated_spec = utils.generate_dict(facts_argument_spec)

    def populate_facts(self, connection, ansible_facts, data=None):
        if False:
            i = 10
            return i + 15
        ' Populate the facts for lldp_global\n        :param connection: the device connection\n        :param ansible_facts: Facts dictionary\n        :param data: previously collected conf\n        :rtype: dictionary\n        :returns: facts\n        '
        if not data:
            data = connection.get_config()
        objs = {}
        lldp_output = findall('^set service lldp (\\S+)', data, M)
        if lldp_output:
            for item in set(lldp_output):
                lldp_regex = ' %s .+$' % item
                cfg = findall(lldp_regex, data, M)
                obj = self.render_config(cfg)
                if obj:
                    objs.update(obj)
        lldp_service = findall("^set service (lldp)?('lldp')", data, M)
        if lldp_service or lldp_output:
            lldp_obj = {}
            lldp_obj['enable'] = True
            objs.update(lldp_obj)
        facts = {}
        params = utils.validate_config(self.argument_spec, {'config': objs})
        facts['lldp_global'] = utils.remove_empties(params['config'])
        ansible_facts['ansible_network_resources'].update(facts)
        return ansible_facts

    def render_config(self, conf):
        if False:
            while True:
                i = 10
        '\n         Render config as dictionary structure and delete keys\n           from spec for null values\n         :param spec: The facts tree, generated from the argspec\n         :param conf: The configuration\n         :rtype: dictionary\n         :returns: The generated config\n         '
        protocol_conf = '\n'.join(filter(lambda x: 'legacy-protocols' in x, conf))
        att_conf = '\n'.join(filter(lambda x: 'legacy-protocols' not in x, conf))
        config = self.parse_attribs(['snmp', 'address'], att_conf)
        config['legacy_protocols'] = self.parse_protocols(protocol_conf)
        return utils.remove_empties(config)

    def parse_protocols(self, conf):
        if False:
            i = 10
            return i + 15
        protocol_support = None
        if conf:
            protocols = findall('^.*legacy-protocols (.+)', conf, M)
            if protocols:
                protocol_support = []
                for protocol in protocols:
                    protocol_support.append(protocol.strip("'"))
        return protocol_support

    def parse_attribs(self, attribs, conf):
        if False:
            return 10
        config = {}
        for item in attribs:
            value = utils.parse_conf_arg(conf, item)
            if value:
                config[item] = value.strip("'")
            else:
                config[item] = None
        return utils.remove_empties(config)