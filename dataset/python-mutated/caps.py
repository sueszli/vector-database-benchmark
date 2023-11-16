from __future__ import annotations
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.collector import BaseFactCollector

class SystemCapabilitiesFactCollector(BaseFactCollector):
    name = 'caps'
    _fact_ids = set(['system_capabilities', 'system_capabilities_enforced'])

    def collect(self, module=None, collected_facts=None):
        if False:
            for i in range(10):
                print('nop')
        rc = -1
        facts_dict = {'system_capabilities_enforced': 'N/A', 'system_capabilities': 'N/A'}
        if module:
            capsh_path = module.get_bin_path('capsh')
            if capsh_path:
                try:
                    (rc, out, err) = module.run_command([capsh_path, '--print'], errors='surrogate_then_replace', handle_exceptions=False)
                except (IOError, OSError) as e:
                    module.warn('Could not query system capabilities: %s' % str(e))
            if rc == 0:
                enforced_caps = []
                enforced = 'NA'
                for line in out.splitlines():
                    if len(line) < 1:
                        continue
                    if line.startswith('Current:'):
                        if line.split(':')[1].strip() == '=ep':
                            enforced = 'False'
                        else:
                            enforced = 'True'
                            enforced_caps = [i.strip() for i in line.split('=')[1].split(',')]
                facts_dict['system_capabilities_enforced'] = enforced
                facts_dict['system_capabilities'] = enforced_caps
        return facts_dict