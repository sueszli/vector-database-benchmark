from typing import Optional, Any, Dict

class StackSettingsSecureConfigValue:
    """A secret Stack config entry."""
    secure: str

    def __init__(self, secure: str):
        if False:
            for i in range(10):
                print('nop')
        self.secure = secure

class StackSettings:
    """A description of the Stack's configuration and encryption metadata."""
    secrets_provider: Optional[str]
    encrypted_key: Optional[str]
    encryption_salt: Optional[str]
    config: Optional[Dict[str, Any]]

    def __init__(self, secrets_provider: Optional[str]=None, encrypted_key: Optional[str]=None, encryption_salt: Optional[str]=None, config: Optional[Dict[str, Any]]=None):
        if False:
            return 10
        self.secrets_provider = secrets_provider
        self.encrypted_key = encrypted_key
        self.encryption_salt = encryption_salt
        self.config = config

    @classmethod
    def _deserialize(cls, data: dict):
        if False:
            for i in range(10):
                print('nop')
        config = data.get('config')
        if config is not None:
            stack_config: Dict[str, Any] = {}
            for (key, val) in config.items():
                if isinstance(val, str):
                    stack_config[key] = val
                elif 'secure' in val:
                    stack_config[key] = StackSettingsSecureConfigValue(**val)
            config = stack_config
        return cls(secrets_provider=data.get('secretsprovider'), encrypted_key=data.get('encryptedkey'), encryption_salt=data.get('encryptionsalt'), config=config)

    def _serialize(self):
        if False:
            while True:
                i = 10
        serializable = {}
        if self.secrets_provider:
            serializable['secretsprovider'] = self.secrets_provider
        if self.encrypted_key:
            serializable['encryptedkey'] = self.encrypted_key
        if self.encryption_salt:
            serializable['encryptionsalt'] = self.encryption_salt
        if self.config:
            config = {}
            for (key, val) in self.config.items():
                if isinstance(val, StackSettingsSecureConfigValue):
                    config[key] = {'secure': val.secure}
                else:
                    config[key] = val
            serializable['config'] = config
        return serializable