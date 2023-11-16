import logging
import re
import sys
import ssl
from unittest.mock import call, Mock, ANY
import pytest
import apprise
from apprise.plugins.NotifyMQTT import NotifyMQTT
logging.disable(logging.CRITICAL)

@pytest.fixture
def mqtt_client_mock(mocker):
    if False:
        i = 10
        return i + 15
    '\n    Mocks an MQTT client and response and returns the mocked client.\n    '
    if 'paho' not in sys.modules:
        raise pytest.skip('Requires that `paho-mqtt` is installed')
    publish_result = Mock(**{'rc': 0, 'is_published.return_value': True})
    mock_client = Mock(**{'connect.return_value': 0, 'reconnect.return_value': 0, 'is_connected.return_value': True, 'publish.return_value': publish_result})
    mocker.patch('paho.mqtt.client.Client', return_value=mock_client)
    return mock_client

@pytest.mark.skipif('paho' in sys.modules, reason='Requires that `paho-mqtt` is NOT installed')
def test_plugin_mqtt_paho_import_error():
    if False:
        while True:
            i = 10
    '\n    Verify `NotifyMQTT` is disabled when `paho.mqtt.client` fails loading.\n    '
    obj = apprise.Apprise.instantiate('mqtt://user:pass@localhost/my/topic')
    assert obj is None

def test_plugin_mqtt_default_success(mqtt_client_mock):
    if False:
        return 10
    '\n    Verify `NotifyMQTT` succeeds and has appropriate default settings.\n    '
    obj = apprise.Apprise.instantiate('mqtt://localhost:1234/my/topic', suppress_exceptions=False)
    assert isinstance(obj, NotifyMQTT)
    assert len(obj) == 1
    assert obj.url().startswith('mqtt://localhost:1234/my/topic')
    assert re.search('qos=0', obj.url())
    assert re.search('version=v3.1.1', obj.url())
    assert re.search('session=no', obj.url())
    assert re.search('client_id=', obj.url()) is None
    assert obj.notify(body='test=test') is True
    assert obj.notify(body='foo=bar') is True
    assert mqtt_client_mock.mock_calls == [call.max_inflight_messages_set(200), call.connect('localhost', port=1234, keepalive=30), call.loop_start(), call.is_connected(), call.publish('my/topic', payload='test=test', qos=0, retain=False), call.publish().is_published(), call.is_connected(), call.publish('my/topic', payload='foo=bar', qos=0, retain=False), call.publish().is_published()]

def test_plugin_mqtt_multiple_topics_success(mqtt_client_mock):
    if False:
        while True:
            i = 10
    '\n    Verify submission to multiple MQTT topics.\n    '
    obj = apprise.Apprise.instantiate('mqtt://localhost/my/topic,my/other/topic', suppress_exceptions=False)
    assert len(obj) == 2
    assert isinstance(obj, NotifyMQTT)
    assert obj.url().startswith('mqtt://localhost')
    assert re.search('my/topic', obj.url())
    assert re.search('my/other/topic', obj.url())
    assert obj.notify(body='test=test') is True
    assert mqtt_client_mock.mock_calls == [call.max_inflight_messages_set(200), call.connect('localhost', port=1883, keepalive=30), call.loop_start(), call.is_connected(), call.publish('my/topic', payload='test=test', qos=0, retain=False), call.publish().is_published(), call.is_connected(), call.publish('my/other/topic', payload='test=test', qos=0, retain=False), call.publish().is_published()]

def test_plugin_mqtt_to_success(mqtt_client_mock):
    if False:
        i = 10
        return i + 15
    '\n    Verify `NotifyMQTT` succeeds with the `to=` parameter.\n    '
    obj = apprise.Apprise.instantiate('mqtt://localhost?to=my/topic', suppress_exceptions=False)
    assert isinstance(obj, NotifyMQTT)
    assert obj.url().startswith('mqtt://localhost/my/topic')
    assert re.search('qos=0', obj.url())
    assert re.search('version=v3.1.1', obj.url())
    assert obj.notify(body='test=test') is True

def test_plugin_mqtt_valid_settings_success(mqtt_client_mock):
    if False:
        i = 10
        return i + 15
    '\n    Verify settings as URL parameters will be accepted.\n    '
    obj = apprise.Apprise.instantiate('mqtt://localhost/my/topic?qos=1&version=v3.1', suppress_exceptions=False)
    assert isinstance(obj, NotifyMQTT)
    assert obj.url().startswith('mqtt://localhost')
    assert re.search('qos=1', obj.url())
    assert re.search('version=v3.1', obj.url())

def test_plugin_mqtt_invalid_settings_failure(mqtt_client_mock):
    if False:
        i = 10
        return i + 15
    '\n    Verify notifier instantiation croaks on invalid settings.\n    '
    with pytest.raises(TypeError):
        apprise.Apprise.instantiate('mqtt://localhost?version=v1.0.0.0', suppress_exceptions=False)
    with pytest.raises(TypeError):
        apprise.Apprise.instantiate('mqtt://localhost?qos=123', suppress_exceptions=False)
    with pytest.raises(TypeError):
        apprise.Apprise.instantiate('mqtt://localhost?qos=invalid', suppress_exceptions=False)

def test_plugin_mqtt_bad_url_failure(mqtt_client_mock):
    if False:
        i = 10
        return i + 15
    '\n    Verify notifier is disabled when using an invalid URL.\n    '
    obj = apprise.Apprise.instantiate('mqtt://', suppress_exceptions=False)
    assert obj is None

def test_plugin_mqtt_no_topic_failure(mqtt_client_mock):
    if False:
        print('Hello World!')
    '\n    Verify notification fails when no topic is given.\n    '
    obj = apprise.Apprise.instantiate('mqtt://localhost', suppress_exceptions=False)
    assert isinstance(obj, NotifyMQTT)
    assert obj.notify(body='test=test') is False

def test_plugin_mqtt_tls_connect_success(mqtt_client_mock):
    if False:
        i = 10
        return i + 15
    '\n    Verify TLS encrypted connections work.\n    '
    obj = apprise.Apprise.instantiate('mqtts://user:pass@localhost/my/topic', suppress_exceptions=False)
    assert isinstance(obj, NotifyMQTT)
    assert obj.url().startswith('mqtts://user:pass@localhost/my/topic')
    assert obj.notify(body='test=test') is True
    assert mqtt_client_mock.mock_calls == [call.max_inflight_messages_set(200), call.username_pw_set('user', password='pass'), call.tls_set(ca_certs=ANY, certfile=None, keyfile=None, cert_reqs=ssl.VerifyMode.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS, ciphers=None), call.tls_insecure_set(False), call.connect('localhost', port=8883, keepalive=30), call.loop_start(), call.is_connected(), call.publish('my/topic', payload='test=test', qos=0, retain=False), call.publish().is_published()]

def test_plugin_mqtt_tls_no_certificates_failure(mqtt_client_mock, mocker):
    if False:
        i = 10
        return i + 15
    '\n    Verify TLS does not work without access to CA root certificates.\n    '
    mocker.patch.object(NotifyMQTT, 'CA_CERTIFICATE_FILE_LOCATIONS', [])
    obj = apprise.Apprise.instantiate('mqtts://user:pass@localhost/my/topic', suppress_exceptions=False)
    assert isinstance(obj, NotifyMQTT)
    logger: Mock = mocker.spy(obj, 'logger')
    assert obj.notify(body='test=test') is False
    assert logger.mock_calls == [call.error('MQTT secure communication can not be verified, CA certificates file missing')]

def test_plugin_mqtt_tls_no_verify_success(mqtt_client_mock):
    if False:
        while True:
            i = 10
    '\n    Verify TLS encrypted connections work with `verify=False`.\n    '
    obj = apprise.Apprise.instantiate('mqtts://user:pass@localhost/my/topic?verify=False', suppress_exceptions=False)
    assert isinstance(obj, NotifyMQTT)
    assert obj.notify(body='test=test') is True
    assert call.tls_insecure_set(True) in mqtt_client_mock.mock_calls

def test_plugin_mqtt_session_client_id_success(mqtt_client_mock):
    if False:
        return 10
    '\n    Verify handling `session=yes` and `client_id=` works.\n    '
    obj = apprise.Apprise.instantiate('mqtt://user@localhost/my/topic?session=yes&client_id=apprise', suppress_exceptions=False)
    assert isinstance(obj, NotifyMQTT)
    assert obj.url().startswith('mqtt://user@localhost')
    assert re.search('my/topic', obj.url())
    assert re.search('client_id=apprise', obj.url())
    assert re.search('session=yes', obj.url())
    assert obj.notify(body='test=test') is True

def test_plugin_mqtt_connect_failure(mqtt_client_mock):
    if False:
        while True:
            i = 10
    '\n    Verify `NotifyMQTT` fails when MQTT `connect()` fails.\n    '
    mqtt_client_mock.connect.return_value = 2
    obj = apprise.Apprise.instantiate('mqtt://localhost/my/topic', suppress_exceptions=False)
    assert obj.notify(body='test=test') is False

def test_plugin_mqtt_reconnect_failure(mqtt_client_mock):
    if False:
        print('Hello World!')
    '\n    Verify `NotifyMQTT` fails when MQTT `reconnect()` fails.\n    '
    mqtt_client_mock.reconnect.return_value = 2
    mqtt_client_mock.is_connected.return_value = False
    obj = apprise.Apprise.instantiate('mqtt://localhost/my/topic', suppress_exceptions=False)
    assert obj.notify(body='test=test') is False

def test_plugin_mqtt_publish_failure(mqtt_client_mock):
    if False:
        i = 10
        return i + 15
    '\n    Verify `NotifyMQTT` fails when MQTT `publish()` fails.\n    '
    mqtt_response = mqtt_client_mock.publish.return_value
    mqtt_response.rc = 2
    obj = apprise.Apprise.instantiate('mqtt://localhost/my/topic', suppress_exceptions=False)
    assert obj.notify(body='test=test') is False

def test_plugin_mqtt_exception_failure(mqtt_client_mock):
    if False:
        print('Hello World!')
    '\n    Verify `NotifyMQTT` fails when an exception happens.\n    '
    obj = apprise.Apprise.instantiate('mqtt://localhost/my/topic', suppress_exceptions=False)
    mqtt_client_mock.connect.return_value = None
    for side_effect in (ValueError, ConnectionError, ssl.CertificateError):
        mqtt_client_mock.connect.side_effect = side_effect
        assert obj.notify(body='test=test') is False

def test_plugin_mqtt_not_published_failure(mqtt_client_mock, mocker):
    if False:
        return 10
    '\n    Verify `NotifyMQTT` fails there if the message has not been published.\n    '
    mocker.patch.object(NotifyMQTT, 'socket_read_timeout', 0.00025)
    mocker.patch.object(NotifyMQTT, 'mqtt_block_time_sec', 0)
    mqtt_response = mqtt_client_mock.publish.return_value
    mqtt_response.is_published.return_value = False
    obj = apprise.Apprise.instantiate('mqtt://localhost/my/topic', suppress_exceptions=False)
    assert obj.notify(body='test=test') is False

def test_plugin_mqtt_not_published_recovery_success(mqtt_client_mock):
    if False:
        for i in range(10):
            print('nop')
    '\n    Verify `NotifyMQTT` success after recovering from is_published==False.\n    '
    mqtt_response = mqtt_client_mock.publish.return_value
    mqtt_response.is_published.return_value = None
    mqtt_response.is_published.side_effect = (False, True)
    obj = apprise.Apprise.instantiate('mqtt://localhost/my/topic', suppress_exceptions=False)
    assert obj.notify(body='test=test') is True