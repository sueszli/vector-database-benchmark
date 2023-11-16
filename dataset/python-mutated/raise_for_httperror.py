"""Raise exception for an HTTP response is an error.

"""
from searx.exceptions import SearxEngineCaptchaException, SearxEngineTooManyRequestsException, SearxEngineAccessDeniedException
from searx import get_setting

def is_cloudflare_challenge(resp):
    if False:
        i = 10
        return i + 15
    if resp.status_code in [429, 503]:
        if '__cf_chl_jschl_tk__=' in resp.text or ('/cdn-cgi/challenge-platform/' in resp.text and 'orchestrate/jsch/v1' in resp.text and ('window._cf_chl_enter(' in resp.text)):
            return True
    if resp.status_code == 403 and '__cf_chl_captcha_tk__=' in resp.text:
        return True
    return False

def is_cloudflare_firewall(resp):
    if False:
        for i in range(10):
            print('nop')
    return resp.status_code == 403 and '<span class="cf-error-code">1020</span>' in resp.text

def raise_for_cloudflare_captcha(resp):
    if False:
        print('Hello World!')
    if resp.headers.get('Server', '').startswith('cloudflare'):
        if is_cloudflare_challenge(resp):
            raise SearxEngineCaptchaException(message='Cloudflare CAPTCHA', suspended_time=get_setting('search.suspended_times.cf_SearxEngineCaptcha'))
        if is_cloudflare_firewall(resp):
            raise SearxEngineAccessDeniedException(message='Cloudflare Firewall', suspended_time=get_setting('search.suspended_times.cf_SearxEngineAccessDenied'))

def raise_for_recaptcha(resp):
    if False:
        return 10
    if resp.status_code == 503 and '"https://www.google.com/recaptcha/' in resp.text:
        raise SearxEngineCaptchaException(message='ReCAPTCHA', suspended_time=get_setting('search.suspended_times.recaptcha_SearxEngineCaptcha'))

def raise_for_captcha(resp):
    if False:
        print('Hello World!')
    raise_for_cloudflare_captcha(resp)
    raise_for_recaptcha(resp)

def raise_for_httperror(resp):
    if False:
        print('Hello World!')
    'Raise exception for an HTTP response is an error.\n\n    Args:\n        resp (requests.Response): Response to check\n\n    Raises:\n        requests.HTTPError: raise by resp.raise_for_status()\n        searx.exceptions.SearxEngineAccessDeniedException: raise when the HTTP status code is 402 or 403.\n        searx.exceptions.SearxEngineTooManyRequestsException: raise when the HTTP status code is 429.\n        searx.exceptions.SearxEngineCaptchaException: raise when if CATPCHA challenge is detected.\n    '
    if resp.status_code and resp.status_code >= 400:
        raise_for_captcha(resp)
        if resp.status_code in (402, 403):
            raise SearxEngineAccessDeniedException(message='HTTP error ' + str(resp.status_code))
        if resp.status_code == 429:
            raise SearxEngineTooManyRequestsException()
        resp.raise_for_status()