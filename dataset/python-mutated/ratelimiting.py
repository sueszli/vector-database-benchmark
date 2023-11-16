from typing import Any, Dict, Optional, cast
import attr
from synapse.types import JsonDict
from ._base import Config

@attr.s(slots=True, frozen=True, auto_attribs=True)
class RatelimitSettings:
    key: str
    per_second: float
    burst_count: int

    @classmethod
    def parse(cls, config: Dict[str, Any], key: str, defaults: Optional[Dict[str, float]]=None) -> 'RatelimitSettings':
        if False:
            print('Hello World!')
        'Parse config[key] as a new-style rate limiter config.\n\n        The key may refer to a nested dictionary using a full stop (.) to separate\n        each nested key. For example, use the key "a.b.c" to parse the following:\n\n        a:\n          b:\n            c:\n              per_second: 10\n              burst_count: 200\n\n        If this lookup fails, we\'ll fallback to the defaults.\n        '
        defaults = defaults or {'per_second': 0.17, 'burst_count': 3.0}
        rl_config = config
        for part in key.split('.'):
            rl_config = rl_config.get(part, {})
        rl_config = cast(Dict[str, float], rl_config)
        return cls(key=key, per_second=rl_config.get('per_second', defaults['per_second']), burst_count=int(rl_config.get('burst_count', defaults['burst_count'])))

@attr.s(auto_attribs=True)
class FederationRatelimitSettings:
    window_size: int = 1000
    sleep_limit: int = 10
    sleep_delay: int = 500
    reject_limit: int = 50
    concurrent: int = 3

class RatelimitConfig(Config):
    section = 'ratelimiting'

    def read_config(self, config: JsonDict, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        if 'rc_message' in config:
            self.rc_message = RatelimitSettings.parse(config, 'rc_message', defaults={'per_second': 0.2, 'burst_count': 10.0})
        else:
            self.rc_message = RatelimitSettings(key='rc_messages', per_second=config.get('rc_messages_per_second', 0.2), burst_count=config.get('rc_message_burst_count', 10.0))
        if 'rc_federation' in config:
            self.rc_federation = FederationRatelimitSettings(**config['rc_federation'])
        else:
            self.rc_federation = FederationRatelimitSettings(**{k: v for (k, v) in {'window_size': config.get('federation_rc_window_size'), 'sleep_limit': config.get('federation_rc_sleep_limit'), 'sleep_delay': config.get('federation_rc_sleep_delay'), 'reject_limit': config.get('federation_rc_reject_limit'), 'concurrent': config.get('federation_rc_concurrent')}.items() if v is not None})
        self.rc_registration = RatelimitSettings.parse(config, 'rc_registration', {})
        self.rc_registration_token_validity = RatelimitSettings.parse(config, 'rc_registration_token_validity', defaults={'per_second': 0.1, 'burst_count': 5})
        self.rc_login_address = RatelimitSettings.parse(config, 'rc_login.address', defaults={'per_second': 0.003, 'burst_count': 5})
        self.rc_login_account = RatelimitSettings.parse(config, 'rc_login.account', defaults={'per_second': 0.003, 'burst_count': 5})
        self.rc_login_failed_attempts = RatelimitSettings.parse(config, 'rc_login.failed_attempts', {})
        self.federation_rr_transactions_per_room_per_second = config.get('federation_rr_transactions_per_room_per_second', 50)
        self.rc_admin_redaction = None
        if 'rc_admin_redaction' in config:
            self.rc_admin_redaction = RatelimitSettings.parse(config, 'rc_admin_redaction', {})
        self.rc_joins_local = RatelimitSettings.parse(config, 'rc_joins.local', defaults={'per_second': 0.1, 'burst_count': 10})
        self.rc_joins_remote = RatelimitSettings.parse(config, 'rc_joins.remote', defaults={'per_second': 0.01, 'burst_count': 10})
        self.rc_joins_per_room = RatelimitSettings.parse(config, 'rc_joins_per_room', defaults={'per_second': 1, 'burst_count': 10})
        self.rc_key_requests = RatelimitSettings.parse(config, 'rc_key_requests', defaults={'per_second': 20, 'burst_count': 100})
        self.rc_3pid_validation = RatelimitSettings.parse(config, 'rc_3pid_validation', defaults={'per_second': 0.003, 'burst_count': 5})
        self.rc_invites_per_room = RatelimitSettings.parse(config, 'rc_invites.per_room', defaults={'per_second': 0.3, 'burst_count': 10})
        self.rc_invites_per_user = RatelimitSettings.parse(config, 'rc_invites.per_user', defaults={'per_second': 0.003, 'burst_count': 5})
        self.rc_invites_per_issuer = RatelimitSettings.parse(config, 'rc_invites.per_issuer', defaults={'per_second': 0.3, 'burst_count': 10})
        self.rc_third_party_invite = RatelimitSettings.parse(config, 'rc_third_party_invite', defaults={'per_second': 0.0025, 'burst_count': 5})
        self.rc_media_create = RatelimitSettings.parse(config, 'rc_media_create', defaults={'per_second': 10, 'burst_count': 50})