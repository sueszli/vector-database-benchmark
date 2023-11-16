from typing import Any, List, Tuple, Type
from synapse.types import JsonDict
from synapse.util.module_loader import load_module
from ._base import Config
LDAP_PROVIDER = 'ldap_auth_provider.LdapAuthProvider'

class PasswordAuthProviderConfig(Config):
    section = 'authproviders'

    def read_config(self, config: JsonDict, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Parses the old password auth providers config. The config format looks like this:\n\n        password_providers:\n           # Example config for an LDAP auth provider\n           - module: "ldap_auth_provider.LdapAuthProvider"\n             config:\n               enabled: true\n               uri: "ldap://ldap.example.com:389"\n               start_tls: true\n               base: "ou=users,dc=example,dc=com"\n               attributes:\n                  uid: "cn"\n                  mail: "email"\n                  name: "givenName"\n               #bind_dn:\n               #bind_password:\n               #filter: "(objectClass=posixAccount)"\n\n        We expect admins to use modules for this feature (which is why it doesn\'t appear\n        in the sample config file), but we want to keep support for it around for a bit\n        for backwards compatibility.\n        '
        self.password_providers: List[Tuple[Type, Any]] = []
        providers = []
        ldap_config = config.get('ldap_config', {})
        if ldap_config.get('enabled', False):
            providers.append({'module': LDAP_PROVIDER, 'config': ldap_config})
        providers.extend(config.get('password_providers') or [])
        for (i, provider) in enumerate(providers):
            mod_name = provider['module']
            if mod_name == 'synapse.util.ldap_auth_provider.LdapAuthProvider':
                mod_name = LDAP_PROVIDER
            (provider_class, provider_config) = load_module({'module': mod_name, 'config': provider['config']}, ('password_providers', '<item %i>' % i))
            self.password_providers.append((provider_class, provider_config))