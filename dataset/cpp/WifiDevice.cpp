#include <WiFi.h>
#include "WifiDevice.h"
#include "../PreferencesKeys.h"
#include "../Logger.h"
#include "../MqttTopics.h"
#include "espMqttClient.h"
#include "../RestartReason.h"

RTC_NOINIT_ATTR char WiFiDevice_reconfdetect[17];

WifiDevice::WifiDevice(const String& hostname, Preferences* _preferences, const IPConfiguration* ipConfiguration)
: NetworkDevice(hostname, ipConfiguration)
{
    _startAp = strcmp(WiFiDevice_reconfdetect, "reconfigure_wifi") == 0;

    _restartOnDisconnect = _preferences->getBool(preference_restart_on_disconnect);

    size_t caLength = _preferences->getString(preference_mqtt_ca,_ca,TLS_CA_MAX_SIZE);
    size_t crtLength = _preferences->getString(preference_mqtt_crt,_cert,TLS_CERT_MAX_SIZE);
    size_t keyLength = _preferences->getString(preference_mqtt_key,_key,TLS_KEY_MAX_SIZE);

    _useEncryption = caLength > 1;  // length is 1 when empty

    if(_useEncryption)
    {
        Log->println(F("MQTT over TLS."));
        Log->println(_ca);
        _mqttClientSecure = new espMqttClientSecure(espMqttClientTypes::UseInternalTask::NO);
        _mqttClientSecure->setCACert(_ca);
        if(crtLength > 1 && keyLength > 1) // length is 1 when empty
        {
            Log->println(F("MQTT with client certificate."));
            Log->println(_cert);
            Log->println(_key);
            _mqttClientSecure->setCertificate(_cert);
            _mqttClientSecure->setPrivateKey(_key);
        }
    } else
    {
        Log->println(F("MQTT without TLS."));
        _mqttClient = new espMqttClient(espMqttClientTypes::UseInternalTask::NO);
    }

    if(_preferences->getBool(preference_mqtt_log_enabled))
    {
        _path = new char[200];
        memset(_path, 0, sizeof(_path));

        String pathStr = _preferences->getString(preference_mqtt_lock_path);
        pathStr.concat(mqtt_topic_log);
        strcpy(_path, pathStr.c_str());
        Log = new MqttLogger(this, _path, MqttLoggerMode::MqttAndSerial);
    }
}

const String WifiDevice::deviceName() const
{
    return "Built-in Wifi";
}

void WifiDevice::initialize()
{
    std::vector<const char *> wm_menu;
    wm_menu.push_back("wifi");
    wm_menu.push_back("exit");
    // reduced tieout if ESP is set to restart on disconnect
    _wm.setConfigPortalTimeout(_restartOnDisconnect ? 60 * 3 : 60 * 30);
    _wm.setShowInfoUpdate(false);
    _wm.setMenu(wm_menu);
    _wm.setHostname(_hostname);

    if(!_ipConfiguration->dhcpEnabled())
    {
        _wm.setSTAStaticIPConfig(_ipConfiguration->ipAddress(), _ipConfiguration->defaultGateway(), _ipConfiguration->subnet(), _ipConfiguration->dnsServer());
    }

    _wm.setAPCallback(clearRtcInitVar);

    bool res = false;

    if(_startAp)
    {
        Log->println(F("Opening WiFi configuration portal."));
        res = _wm.startConfigPortal();
    }
    else
    {
        res = _wm.autoConnect(); // password protected ap
    }

    if(!res) {
        Log->println(F("Failed to connect. Wait for ESP restart."));
        delay(1000);
        restartEsp(RestartReason::WifiInitFailed);
    }
    else {
        Log->print(F("WiFi connected: "));
        Log->println(WiFi.localIP().toString());
    }

    if(_restartOnDisconnect)
    {
        WiFi.onEvent([&](WiFiEvent_t event, WiFiEventInfo_t info)
        {
            if(event == ARDUINO_EVENT_WIFI_STA_DISCONNECTED)
            {
                onDisconnected();
            }
        });
    }
}

void WifiDevice::reconfigure()
{
    strcpy(WiFiDevice_reconfdetect, "reconfigure_wifi");
    delay(200);
    restartEsp(RestartReason::ReconfigureWifi);
}

void WifiDevice::printError()
{
//    if(_wifiClientSecure != nullptr)
//    {
//        char lastError[100];
//        _wifiClientSecure->lastError(lastError,100);
//        Log->println(lastError);
//    }
    Log->print(F("Free Heap: "));
    Log->println(ESP.getFreeHeap());
}

bool WifiDevice::supportsEncryption()
{
    return true;
}

bool WifiDevice::isConnected()
{
    return WiFi.isConnected();
}

ReconnectStatus WifiDevice::reconnect()
{
    delay(3000);
    return isConnected() ? ReconnectStatus::Success : ReconnectStatus::Failure;
}

void WifiDevice::update()
{
    if(_mqttEnabled)
    {
        if (_useEncryption)
        {
            _mqttClientSecure->loop();
        } else
        {
            _mqttClient->loop();
        }
    }
}

void WifiDevice::onDisconnected()
{
    if(millis() > 60000)
    {
        restartEsp(RestartReason::RestartOnDisconnectWatchdog);
    }
}

int8_t WifiDevice::signalStrength()
{
    return WiFi.RSSI();
}

void WifiDevice::clearRtcInitVar(WiFiManager *)
{
    memset(WiFiDevice_reconfdetect, 0, sizeof WiFiDevice_reconfdetect);
}

void WifiDevice::mqttSetClientId(const char *clientId)
{
    if(_useEncryption)
    {
        _mqttClientSecure->setClientId(clientId);
    }
    else
    {
        _mqttClient->setClientId(clientId);
    }
}

void WifiDevice::mqttSetCleanSession(bool cleanSession)
{
    if(_useEncryption)
    {
        _mqttClientSecure->setCleanSession(cleanSession);
    }
    else
    {
        _mqttClient->setCleanSession(cleanSession);
    }
}

uint16_t WifiDevice::mqttPublish(const char *topic, uint8_t qos, bool retain, const char *payload)
{
    if(_useEncryption)
    {
        return _mqttClientSecure->publish(topic, qos, retain, payload);
    }
    else
    {
        return _mqttClient->publish(topic, qos, retain, payload);
    }
}

uint16_t WifiDevice::mqttPublish(const char *topic, uint8_t qos, bool retain, const uint8_t *payload, size_t length)
{
    if(_useEncryption)
    {
        return _mqttClientSecure->publish(topic, qos, retain, payload, length);
    }
    else
    {
        return _mqttClient->publish(topic, qos, retain, payload, length);
    }
}

bool WifiDevice::mqttConnected() const
{
    if(_useEncryption)
    {
        return _mqttClientSecure->connected();
    }
    else
    {
        return _mqttClient->connected();
    }
}

void WifiDevice::mqttSetServer(const char *host, uint16_t port)
{
    if(_useEncryption)
    {
        _mqttClientSecure->setServer(host, port);
    }
    else
    {
        _mqttClient->setServer(host, port);
    }
}

bool WifiDevice::mqttConnect()
{
    if(_useEncryption)
    {
        return _mqttClientSecure->connect();
    }
    else
    {
        return _mqttClient->connect();
    }
}

bool WifiDevice::mqttDisonnect(bool force)
{
    if(_useEncryption)
    {
        return _mqttClientSecure->disconnect(force);
    }
    else
    {
        return _mqttClient->disconnect(force);
    }
}

void WifiDevice::setWill(const char *topic, uint8_t qos, bool retain, const char *payload)
{
    if(_useEncryption)
    {
        _mqttClientSecure->setWill(topic, qos, retain, payload);
    }
    else
    {
        _mqttClient->setWill(topic, qos, retain, payload);
    }
}

void WifiDevice::mqttSetCredentials(const char *username, const char *password)
{
    if(_useEncryption)
    {
        _mqttClientSecure->setCredentials(username, password);
    }
    else
    {
        _mqttClient->setCredentials(username, password);
    }
}

void WifiDevice::mqttOnMessage(espMqttClientTypes::OnMessageCallback callback)
{
    if(_useEncryption)
    {
        _mqttClientSecure->onMessage(callback);
    }
    else
    {
        _mqttClient->onMessage(callback);
    }
}


void WifiDevice::mqttOnConnect(espMqttClientTypes::OnConnectCallback callback)
{
    if(_useEncryption)
    {
        _mqttClientSecure->onConnect(callback);
    }
    else
    {
        _mqttClient->onConnect(callback);
    }
}

void WifiDevice::mqttOnDisconnect(espMqttClientTypes::OnDisconnectCallback callback)
{
    if(_useEncryption)
    {
        _mqttClientSecure->onDisconnect(callback);
    }
    else
    {
        _mqttClient->onDisconnect(callback);
    }
}


uint16_t WifiDevice::mqttSubscribe(const char *topic, uint8_t qos)
{
    if(_useEncryption)
    {
        return _mqttClientSecure->subscribe(topic, qos);
    }
    else
    {
        return _mqttClient->subscribe(topic, qos);
    }
}

void WifiDevice::disableMqtt()
{
    if (_useEncryption)
    {
        _mqttClientSecure->disconnect();
    } else
    {
        _mqttClient->disconnect();
    }

    _mqttEnabled = false;
}
