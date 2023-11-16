""" Provide a mixin class to add authorization hooks to a request handler.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from tornado.web import RequestHandler
__all__ = ('AuthRequestHandler',)

class AuthRequestHandler(RequestHandler):
    """ This mixin adds the expected Tornado authorization hooks:

    * get_login_url
    * get_current_user
    * prepare

    All of these delegate to the a :class:`~bokeh.serve.auth_provider.AuthProvider`
    confiured on the Bokeh tornado application.

    """

    def get_login_url(self):
        if False:
            for i in range(10):
                print('nop')
        ' Delegates to``get_login_url`` method of the auth provider, or the\n        ``login_url`` attribute.\n\n        '
        if self.application.auth_provider.get_login_url is not None:
            return self.application.auth_provider.get_login_url(self)
        if self.application.auth_provider.login_url is not None:
            return self.application.auth_provider.login_url
        raise RuntimeError('login_url or get_login_url() must be supplied when authentication hooks are enabled')

    def get_current_user(self):
        if False:
            i = 10
            return i + 15
        ' Delegate to the synchronous ``get_user`` method of the auth\n        provider\n\n        '
        if self.application.auth_provider.get_user is not None:
            return self.application.auth_provider.get_user(self)
        return 'default_user'

    async def prepare(self):
        """ Async counterpart to ``get_current_user``

        """
        if self.application.auth_provider.get_user_async is not None:
            self.current_user = await self.application.auth_provider.get_user_async(self)