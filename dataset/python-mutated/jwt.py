from typing import Any
from synapse.types import JsonDict
from synapse.util.check_dependencies import check_requirements
from ._base import Config

class JWTConfig(Config):
    section = 'jwt'

    def read_config(self, config: JsonDict, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        jwt_config = config.get('jwt_config', None)
        if jwt_config:
            self.jwt_enabled = jwt_config.get('enabled', False)
            self.jwt_secret = jwt_config['secret']
            self.jwt_algorithm = jwt_config['algorithm']
            self.jwt_subject_claim = jwt_config.get('subject_claim', 'sub')
            self.jwt_issuer = jwt_config.get('issuer')
            self.jwt_audiences = jwt_config.get('audiences')
            check_requirements('jwt')
        else:
            self.jwt_enabled = False
            self.jwt_secret = None
            self.jwt_algorithm = None
            self.jwt_subject_claim = None
            self.jwt_issuer = None
            self.jwt_audiences = None