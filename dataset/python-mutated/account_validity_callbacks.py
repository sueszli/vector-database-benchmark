import logging
from typing import Awaitable, Callable, List, Optional, Tuple
from twisted.web.http import Request
logger = logging.getLogger(__name__)
IS_USER_EXPIRED_CALLBACK = Callable[[str], Awaitable[Optional[bool]]]
ON_USER_REGISTRATION_CALLBACK = Callable[[str], Awaitable]
ON_LEGACY_SEND_MAIL_CALLBACK = Callable[[str], Awaitable]
ON_LEGACY_RENEW_CALLBACK = Callable[[str], Awaitable[Tuple[bool, bool, int]]]
ON_LEGACY_ADMIN_REQUEST = Callable[[Request], Awaitable]

class AccountValidityModuleApiCallbacks:

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self.is_user_expired_callbacks: List[IS_USER_EXPIRED_CALLBACK] = []
        self.on_user_registration_callbacks: List[ON_USER_REGISTRATION_CALLBACK] = []
        self.on_legacy_send_mail_callback: Optional[ON_LEGACY_SEND_MAIL_CALLBACK] = None
        self.on_legacy_renew_callback: Optional[ON_LEGACY_RENEW_CALLBACK] = None
        self.on_legacy_admin_request_callback: Optional[ON_LEGACY_ADMIN_REQUEST] = None

    def register_callbacks(self, is_user_expired: Optional[IS_USER_EXPIRED_CALLBACK]=None, on_user_registration: Optional[ON_USER_REGISTRATION_CALLBACK]=None, on_legacy_send_mail: Optional[ON_LEGACY_SEND_MAIL_CALLBACK]=None, on_legacy_renew: Optional[ON_LEGACY_RENEW_CALLBACK]=None, on_legacy_admin_request: Optional[ON_LEGACY_ADMIN_REQUEST]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Register callbacks from module for each hook.'
        if is_user_expired is not None:
            self.is_user_expired_callbacks.append(is_user_expired)
        if on_user_registration is not None:
            self.on_user_registration_callbacks.append(on_user_registration)
        if on_legacy_send_mail is not None:
            if self.on_legacy_send_mail_callback is not None:
                raise RuntimeError('Tried to register on_legacy_send_mail twice')
            self.on_legacy_send_mail_callback = on_legacy_send_mail
        if on_legacy_renew is not None:
            if self.on_legacy_renew_callback is not None:
                raise RuntimeError('Tried to register on_legacy_renew twice')
            self.on_legacy_renew_callback = on_legacy_renew
        if on_legacy_admin_request is not None:
            if self.on_legacy_admin_request_callback is not None:
                raise RuntimeError('Tried to register on_legacy_admin_request twice')
            self.on_legacy_admin_request_callback = on_legacy_admin_request