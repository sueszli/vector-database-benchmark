import collections
import http
from urllib.parse import urlencode
VERIFY_URL = 'https://www.google.com/recaptcha/api/siteverify'

class RecaptchaError(ValueError):
    pass

class MissingInputSecretError(RecaptchaError):
    pass

class InvalidInputSecretError(RecaptchaError):
    pass

class MissingInputResponseError(RecaptchaError):
    pass

class InvalidInputResponseError(RecaptchaError):
    pass

class UnexpectedError(RecaptchaError):
    pass
ERROR_CODE_MAP = {'missing-input-secret': MissingInputSecretError, 'invalid-input-secret': InvalidInputSecretError, 'missing-input-response': MissingInputResponseError, 'invalid-input-response': InvalidInputResponseError}
ChallengeResponse = collections.namedtuple('ChallengeResponse', ('challenge_ts', 'hostname'))

class Service:

    def __init__(self, request):
        if False:
            for i in range(10):
                print('nop')
        self.request = request

    @property
    def csp_policy(self):
        if False:
            return 10
        return {'script-src': ['{request.scheme}://www.recaptcha.net/recaptcha/', '{request.scheme}://www.gstatic.com/recaptcha/', '{request.scheme}://www.gstatic.cn/recaptcha/'], 'frame-src': ['{request.scheme}://www.recaptcha.net/recaptcha/'], 'style-src': ["'unsafe-inline'"]}

    @property
    def enabled(self):
        if False:
            i = 10
            return i + 15
        settings = self.request.registry.settings
        return bool(settings.get('recaptcha.site_key') and settings.get('recaptcha.secret_key'))

    def verify_response(self, response, remote_ip=None):
        if False:
            print('Hello World!')
        if not self.enabled:
            return
        settings = self.request.registry.settings
        payload = {'secret': settings['recaptcha.secret_key'], 'response': response}
        if remote_ip is not None:
            payload['remoteip'] = remote_ip
        try:
            resp = self.request.http.post(VERIFY_URL, urlencode(payload), headers={'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8'}, timeout=10)
        except Exception as err:
            raise UnexpectedError(str(err))
        try:
            data = resp.json()
        except ValueError:
            raise UnexpectedError('Unexpected data in response body: %s' % str(resp.content, 'utf-8'))
        if 'success' not in data:
            raise UnexpectedError("Missing 'success' key in response: %s" % data)
        if resp.status_code != http.HTTPStatus.OK or not data['success']:
            try:
                error_codes = data['error_codes']
            except KeyError:
                raise UnexpectedError("Response missing 'error-codes' key: %s" % data)
            try:
                exc_tp = ERROR_CODE_MAP[error_codes[0]]
            except KeyError:
                raise UnexpectedError('Unexpected error code: %s' % error_codes[0])
            raise exc_tp
        return ChallengeResponse(data.get('challenge_ts'), data.get('hostname'))

def service_factory(handler, request):
    if False:
        return 10
    return Service(request)

def includeme(config):
    if False:
        for i in range(10):
            print('nop')
    config.register_service_factory(service_factory, name='recaptcha')