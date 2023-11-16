import requests
from ckan.common import config
from ckan.types import Request

def check_recaptcha(request: Request) -> None:
    if False:
        for i in range(10):
            print('nop')
    "Check a user's recaptcha submission is valid, and raise CaptchaError\n    on failure."
    recaptcha_private_key = config.get('ckan.recaptcha.privatekey')
    if not recaptcha_private_key:
        return
    client_ip_address = request.environ.get('REMOTE_ADDR', 'Unknown IP Address')
    recaptcha_response_field = request.form.get('g-recaptcha-response', '')
    recaptcha_server_name = 'https://www.google.com/recaptcha/api/siteverify'
    params = dict(secret=recaptcha_private_key, remoteip=client_ip_address, response=recaptcha_response_field.encode('utf8'))
    timeout = config.get('ckan.requests.timeout')
    response = requests.get(recaptcha_server_name, params, timeout=timeout)
    data = response.json()
    try:
        if not data['success']:
            raise CaptchaError()
    except IndexError:
        raise CaptchaError()

class CaptchaError(ValueError):
    pass