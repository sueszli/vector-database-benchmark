from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTShadowClient
import json
import psutil
import argparse
import logging
import time

def configureParser():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--endpoint', action='store', required=True, dest='host', help='Your AWS IoT custom endpoint')
    parser.add_argument('-r', '--rootCA', action='store', required=True, dest='rootCAPath', help='Root CA file path')
    parser.add_argument('-c', '--cert', action='store', required=True, dest='certificatePath', help='Certificate file path')
    parser.add_argument('-k', '--key', action='store', required=True, dest='privateKeyPath', help='Private key file path')
    parser.add_argument('-p', '--port', action='store', dest='port', type=int, default=8883, help='Port number override')
    parser.add_argument('-n', '--thingName', action='store', required=True, dest='thingName', help='Targeted thing name')
    parser.add_argument('-d', '--requestDelay', action='store', dest='requestDelay', type=float, default=1, help='Time between requests (in seconds)')
    parser.add_argument('-v', '--enableLogging', action='store_true', dest='enableLogging', help='Enable logging for the AWS IoT Device SDK for Python')
    return parser

class PerformanceShadowClient:

    def __init__(self, thingName, host, port, rootCAPath, privateKeyPath, certificatePath, requestDelay):
        if False:
            print('Hello World!')
        self.thingName = thingName
        self.host = host
        self.port = port
        self.rootCAPath = rootCAPath
        self.privateKeyPath = privateKeyPath
        self.certificatePath = certificatePath
        self.requestDelay = requestDelay

    def run(self):
        if False:
            i = 10
            return i + 15
        print('Connecting MQTT client for {}...'.format(self.thingName))
        mqttClient = self.configureMQTTClient()
        mqttClient.connect()
        print('MQTT client for {} connected'.format(self.thingName))
        deviceShadowHandler = mqttClient.createShadowHandlerWithName(self.thingName, True)
        print('Running performance shadow client for {}...\n'.format(self.thingName))
        while True:
            performance = self.readPerformance()
            print('[{}]'.format(self.thingName))
            print('CPU:\t{}%'.format(performance['cpu']))
            print('Memory:\t{}%\n'.format(performance['memory']))
            payload = {'state': {'reported': performance}}
            deviceShadowHandler.shadowUpdate(json.dumps(payload), self.shadowUpdateCallback, 5)
            time.sleep(args.requestDelay)

    def configureMQTTClient(self):
        if False:
            for i in range(10):
                print('nop')
        mqttClient = AWSIoTMQTTShadowClient(self.thingName)
        mqttClient.configureEndpoint(self.host, self.port)
        mqttClient.configureCredentials(self.rootCAPath, self.privateKeyPath, self.certificatePath)
        mqttClient.configureAutoReconnectBackoffTime(1, 32, 20)
        mqttClient.configureConnectDisconnectTimeout(10)
        mqttClient.configureMQTTOperationTimeout(5)
        return mqttClient

    def readPerformance(self):
        if False:
            print('Hello World!')
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        timestamp = time.time()
        return {'cpu': cpu, 'memory': memory, 'timestamp': timestamp}

    def shadowUpdateCallback(self, payload, responseStatus, token):
        if False:
            return 10
        print('[{}]'.format(self.thingName))
        print('Update request {} {}\n'.format(token, responseStatus))

def configureLogging():
    if False:
        i = 10
        return i + 15
    logger = logging.getLogger('AWSIoTPythonSDK.core')
    logger.setLevel(logging.DEBUG)
    streamHandler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
if __name__ == '__main__':
    parser = configureParser()
    args = parser.parse_args()
    if args.enableLogging:
        configureLogging()
    thingClient = PerformanceShadowClient(args.thingName, args.host, args.port, args.rootCAPath, args.privateKeyPath, args.certificatePath, args.requestDelay)
    thingClient.run()