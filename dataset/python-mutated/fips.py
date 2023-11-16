from __future__ import annotations
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.utils import get_file_content
from ansible.module_utils.facts.collector import BaseFactCollector

class FipsFactCollector(BaseFactCollector):
    name = 'fips'
    _fact_ids = set()

    def collect(self, module=None, collected_facts=None):
        if False:
            for i in range(10):
                print('nop')
        fips_facts = {}
        fips_facts['fips'] = False
        data = get_file_content('/proc/sys/crypto/fips_enabled')
        if data and data == '1':
            fips_facts['fips'] = True
        return fips_facts