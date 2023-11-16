from __future__ import annotations
import json
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.namespace import PrefixFactNamespace
from ansible.module_utils.facts.collector import BaseFactCollector

class OhaiFactCollector(BaseFactCollector):
    """This is a subclass of Facts for including information gathered from Ohai."""
    name = 'ohai'
    _fact_ids = set()

    def __init__(self, collectors=None, namespace=None):
        if False:
            for i in range(10):
                print('nop')
        namespace = PrefixFactNamespace(namespace_name='ohai', prefix='ohai_')
        super(OhaiFactCollector, self).__init__(collectors=collectors, namespace=namespace)

    def find_ohai(self, module):
        if False:
            return 10
        ohai_path = module.get_bin_path('ohai')
        return ohai_path

    def run_ohai(self, module, ohai_path):
        if False:
            return 10
        (rc, out, err) = module.run_command(ohai_path)
        return (rc, out, err)

    def get_ohai_output(self, module):
        if False:
            while True:
                i = 10
        ohai_path = self.find_ohai(module)
        if not ohai_path:
            return None
        (rc, out, err) = self.run_ohai(module, ohai_path)
        if rc != 0:
            return None
        return out

    def collect(self, module=None, collected_facts=None):
        if False:
            i = 10
            return i + 15
        ohai_facts = {}
        if not module:
            return ohai_facts
        ohai_output = self.get_ohai_output(module)
        if ohai_output is None:
            return ohai_facts
        try:
            ohai_facts = json.loads(ohai_output)
        except Exception:
            pass
        return ohai_facts