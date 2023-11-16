from http import cookiejar
_LOCALHOST = 'localhost'
_LOCALHOST_SUFFIX = '.localhost'

class HTTPieCookiePolicy(cookiejar.DefaultCookiePolicy):

    def return_ok_secure(self, cookie, request):
        if False:
            for i in range(10):
                print('nop')
        'Check whether the given cookie is sent to a secure host.'
        is_secure_protocol = super().return_ok_secure(cookie, request)
        if is_secure_protocol:
            return True
        return self._is_local_host(cookiejar.request_host(request))

    def _is_local_host(self, hostname):
        if False:
            return 10
        return hostname == _LOCALHOST or hostname.endswith(_LOCALHOST_SUFFIX)