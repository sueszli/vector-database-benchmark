from __future__ import annotations
import json
from ansible.module_utils.common.json import AnsibleJSONEncoder
from ansible.parsing.vault import VaultLib
from ansible.parsing.yaml.objects import AnsibleVaultEncryptedUnicode
from ansible.utils.unsafe_proxy import wrap_var

class AnsibleJSONDecoder(json.JSONDecoder):
    _vaults = {}

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        kwargs['object_hook'] = self.object_hook
        super(AnsibleJSONDecoder, self).__init__(*args, **kwargs)

    @classmethod
    def set_secrets(cls, secrets):
        if False:
            for i in range(10):
                print('nop')
        cls._vaults['default'] = VaultLib(secrets=secrets)

    def object_hook(self, pairs):
        if False:
            return 10
        for key in pairs:
            value = pairs[key]
            if key == '__ansible_vault':
                value = AnsibleVaultEncryptedUnicode(value)
                if self._vaults:
                    value.vault = self._vaults['default']
                return value
            elif key == '__ansible_unsafe':
                return wrap_var(value)
        return pairs