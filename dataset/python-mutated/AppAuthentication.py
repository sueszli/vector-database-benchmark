from typing import Dict, Optional, Union
import deprecated
from github.Auth import AppAuth, AppInstallationAuth

@deprecated.deprecated('Use github.Auth.AppInstallationAuth instead')
class AppAuthentication(AppInstallationAuth):

    def __init__(self, app_id: Union[int, str], private_key: str, installation_id: int, token_permissions: Optional[Dict[str, str]]=None):
        if False:
            while True:
                i = 10
        super().__init__(app_auth=AppAuth(app_id, private_key), installation_id=installation_id, token_permissions=token_permissions)