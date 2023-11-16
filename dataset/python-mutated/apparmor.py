from __future__ import annotations
import os
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.collector import BaseFactCollector

class ApparmorFactCollector(BaseFactCollector):
    name = 'apparmor'
    _fact_ids = set()

    def collect(self, module=None, collected_facts=None):
        if False:
            i = 10
            return i + 15
        facts_dict = {}
        apparmor_facts = {}
        if os.path.exists('/sys/kernel/security/apparmor'):
            apparmor_facts['status'] = 'enabled'
        else:
            apparmor_facts['status'] = 'disabled'
        facts_dict['apparmor'] = apparmor_facts
        return facts_dict