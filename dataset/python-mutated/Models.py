from typing import Optional, Dict, Any, List, Union
from copy import deepcopy
from cura.OAuth2.KeyringAttribute import KeyringAttribute

class BaseModel:

    def __init__(self, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        self.__dict__.update(kwargs)

class OAuth2Settings(BaseModel):
    """OAuth OAuth2Settings data template."""
    CALLBACK_PORT = None
    OAUTH_SERVER_URL = None
    CLIENT_ID = None
    CLIENT_SCOPES = None
    CALLBACK_URL = None
    AUTH_DATA_PREFERENCE_KEY = ''
    AUTH_SUCCESS_REDIRECT = 'https://ultimaker.com'
    AUTH_FAILED_REDIRECT = 'https://ultimaker.com'

class UserProfile(BaseModel):
    """User profile data template."""
    user_id = None
    username = None
    profile_image_url = None
    organization_id = None
    subscriptions = None

class AuthenticationResponse(BaseModel):
    """Authentication data template."""
    success = True
    token_type = None
    expires_in = None
    scope = None
    err_message = None
    received_at = None
    access_token = KeyringAttribute()
    refresh_token = KeyringAttribute()

    def __init__(self, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        self.access_token = kwargs.pop('access_token', None)
        self.refresh_token = kwargs.pop('refresh_token', None)
        super(AuthenticationResponse, self).__init__(**kwargs)

    def dump(self) -> Dict[str, Union[bool, Optional[str]]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Dumps the dictionary of Authentication attributes. KeyringAttributes are transformed to public attributes\n        If the keyring was used, these will have a None value, otherwise they will have the secret value\n\n        :return: Dictionary of Authentication attributes\n        '
        dumped = deepcopy(vars(self))
        dumped['access_token'] = dumped.pop('_access_token')
        dumped['refresh_token'] = dumped.pop('_refresh_token')
        return dumped

class ResponseStatus(BaseModel):
    """Response status template."""
    code = 200
    message = ''

class ResponseData(BaseModel):
    """Response data template."""
    status = None
    data_stream = None
    redirect_uri = None
    content_type = 'text/html'
HTTP_STATUS = {'Possible HTTP responses.OK': ResponseStatus(code=200, message='OK'), 'NOT_FOUND': ResponseStatus(code=404, message='NOT FOUND'), 'REDIRECT': ResponseStatus(code=302, message='REDIRECT')}