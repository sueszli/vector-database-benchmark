from __future__ import annotations
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.utils import get_file_content
from ansible.module_utils.facts.collector import BaseFactCollector

class SshPubKeyFactCollector(BaseFactCollector):
    name = 'ssh_pub_keys'
    _fact_ids = set(['ssh_host_pub_keys', 'ssh_host_key_dsa_public', 'ssh_host_key_rsa_public', 'ssh_host_key_ecdsa_public', 'ssh_host_key_ed25519_public'])

    def collect(self, module=None, collected_facts=None):
        if False:
            while True:
                i = 10
        ssh_pub_key_facts = {}
        algos = ('dsa', 'rsa', 'ecdsa', 'ed25519')
        keydirs = ['/etc/ssh', '/etc/openssh', '/etc']
        for keydir in keydirs:
            for algo in algos:
                factname = 'ssh_host_key_%s_public' % algo
                if factname in ssh_pub_key_facts:
                    return ssh_pub_key_facts
                key_filename = '%s/ssh_host_%s_key.pub' % (keydir, algo)
                keydata = get_file_content(key_filename)
                if keydata is not None:
                    (keytype, key) = keydata.split()[0:2]
                    ssh_pub_key_facts[factname] = key
                    ssh_pub_key_facts[factname + '_keytype'] = keytype
        return ssh_pub_key_facts