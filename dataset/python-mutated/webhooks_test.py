import json
import logging
import os
import socket
from time import sleep
import pytest
import requests
from errbot.backends.test import FullStackTest, testbot
log = logging.getLogger(__name__)
PYTHONOBJECT = ['foo', {'bar': ('baz', None, 1.0, 2)}]
JSONOBJECT = json.dumps(PYTHONOBJECT)
WEBSERVER_PORT = 5000 + os.getpid() % 1000
WEBSERVER_SSL_PORT = WEBSERVER_PORT + 1000

def webserver_ready(host, port):
    if False:
        for i in range(10):
            print('nop')
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((host, port))
        s.shutdown(socket.SHUT_RDWR)
        s.close()
        return True
    except Exception:
        return False
extra_plugin_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'webhooks_plugin')

def wait_for_server(port: int):
    if False:
        return 10
    failure_count = 10
    while not webserver_ready('localhost', port):
        waiting_time = 1.0 / failure_count
        log.info('Webserver not ready yet, sleeping for %f second.', waiting_time)
        sleep(waiting_time)
        failure_count -= 1
        if failure_count == 0:
            raise TimeoutError('Could not start the internal Webserver to test.')

@pytest.fixture
def webhook_testbot(request, testbot):
    if False:
        i = 10
        return i + 15
    testbot.push_message("!plugin config Webserver {'HOST': 'localhost', 'PORT': %s, 'SSL': None}" % WEBSERVER_PORT)
    log.info(testbot.pop_message())
    wait_for_server(WEBSERVER_PORT)
    return testbot

def test_not_configured_url_returns_404(webhook_testbot):
    if False:
        i = 10
        return i + 15
    assert requests.post('http://localhost:{}/randomness_blah'.format(WEBSERVER_PORT), "{'toto': 'titui'}").status_code == 404

def test_webserver_plugin_ok(webhook_testbot):
    if False:
        while True:
            i = 10
    assert '/echo' in webhook_testbot.exec_command('!webstatus')

def test_trailing_no_slash_ok(webhook_testbot):
    if False:
        for i in range(10):
            print('nop')
    assert requests.post('http://localhost:{}/echo'.format(WEBSERVER_PORT), JSONOBJECT).text == repr(json.loads(JSONOBJECT))

def test_trailing_slash_also_ok(webhook_testbot):
    if False:
        return 10
    assert requests.post('http://localhost:{}/echo/'.format(WEBSERVER_PORT), JSONOBJECT).text == repr(json.loads(JSONOBJECT))

def test_json_is_automatically_decoded(webhook_testbot):
    if False:
        while True:
            i = 10
    assert requests.post('http://localhost:{}/webhook1'.format(WEBSERVER_PORT), JSONOBJECT).text == repr(json.loads(JSONOBJECT))

def test_json_on_custom_url_is_automatically_decoded(webhook_testbot):
    if False:
        print('Hello World!')
    assert requests.post('http://localhost:{}/custom_webhook'.format(WEBSERVER_PORT), JSONOBJECT).text == repr(json.loads(JSONOBJECT))

def test_post_form_on_webhook_without_form_param_is_automatically_decoded(webhook_testbot):
    if False:
        print('Hello World!')
    assert requests.post('http://localhost:{}/webhook1'.format(WEBSERVER_PORT), data=JSONOBJECT).text == repr(json.loads(JSONOBJECT))

def test_post_form_on_webhook_with_custom_url_and_without_form_param_is_automatically_decoded(webhook_testbot):
    if False:
        i = 10
        return i + 15
    assert requests.post('http://localhost:{}/custom_webhook'.format(WEBSERVER_PORT), data=JSONOBJECT).text == repr(json.loads(JSONOBJECT))

def test_webhooks_with_form_parameter_decode_json_automatically(webhook_testbot):
    if False:
        for i in range(10):
            print('nop')
    form = {'form': JSONOBJECT}
    assert requests.post('http://localhost:{}/form'.format(WEBSERVER_PORT), data=form).text == repr(json.loads(JSONOBJECT))

def test_webhooks_with_form_parameter_on_custom_url_decode_json_automatically(webhook_testbot):
    if False:
        print('Hello World!')
    form = {'form': JSONOBJECT}
    assert requests.post('http://localhost:{}/custom_form'.format(WEBSERVER_PORT), data=form).text, repr(json.loads(JSONOBJECT))

def test_webhooks_with_raw_request(webhook_testbot):
    if False:
        print('Hello World!')
    form = {'form': JSONOBJECT}
    assert 'LocalProxy' in requests.post('http://localhost:{}/raw'.format(WEBSERVER_PORT), data=form).text

def test_webhooks_with_naked_decorator_raw_request(webhook_testbot):
    if False:
        i = 10
        return i + 15
    form = {'form': JSONOBJECT}
    assert 'LocalProxy' in requests.post('http://localhost:{}/raw2'.format(WEBSERVER_PORT), data=form).text

def test_generate_certificate_creates_usable_cert(webhook_testbot):
    if False:
        return 10
    d = webhook_testbot.bot.bot_config.BOT_DATA_DIR
    key_path = os.sep.join((d, 'webserver_key.pem'))
    cert_path = os.sep.join((d, 'webserver_certificate.pem'))
    assert 'Generating' in webhook_testbot.exec_command('!generate_certificate', timeout=1)
    assert 'successfully generated' in webhook_testbot.pop_message(timeout=60)
    assert 'is recommended' in webhook_testbot.pop_message(timeout=1)
    assert key_path in webhook_testbot.pop_message(timeout=1)
    webserver_config = {'HOST': 'localhost', 'PORT': WEBSERVER_PORT, 'SSL': {'certificate': cert_path, 'key': key_path, 'host': 'localhost', 'port': WEBSERVER_SSL_PORT, 'enabled': True}}
    webhook_testbot.push_message('!plugin config Webserver {!r}'.format(webserver_config))
    assert 'Plugin configuration done.' in webhook_testbot.pop_message(timeout=2)
    wait_for_server(WEBSERVER_SSL_PORT)
    requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
    assert requests.post('https://localhost:{}/webhook1'.format(WEBSERVER_SSL_PORT), JSONOBJECT, verify=False).text == repr(json.loads(JSONOBJECT))

def test_custom_headers_and_status_codes(webhook_testbot):
    if False:
        i = 10
        return i + 15
    assert requests.post('http://localhost:{}/webhook6'.format(WEBSERVER_PORT)).headers['X-Powered-By'] == 'Errbot'
    assert requests.post('http://localhost:{}/webhook7'.format(WEBSERVER_PORT)).status_code == 403

def test_lambda_webhook(webhook_testbot):
    if False:
        return 10
    assert requests.post('http://localhost:{}/lambda'.format(WEBSERVER_PORT)).status_code == 200