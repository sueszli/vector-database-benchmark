import paho.mqtt.client as mqtt
import random
import json
prediction = 'light_bright'
broker = 'test.mosquitto.org'
port = 1883
brightness = 100
topicOnOff = '/Licht/Wohnzimmer/OnOff'
topicBrightDim = '/Licht/Wohnzimmer/BrightDim'
topicBrightnessStatus = '/Licht/Wohnzimmer/brightness'
payloadOn = json.dumps({'on': True})
payloadOff = json.dumps({'on': False})
payloadBrighter = json.dumps({'brightness': brightness})
payloadDim = json.dumps({'brightness': brightness})
client = mqtt.Client()
client.subscribe(topicBrightnessStatus)

def on_message(client, userdata, message):
    if False:
        return 10
    print('Received message on topic:', message.topic)
    print('Message payload:', message.payload)
client.on_message = on_message
client.connect(broker, port)
client.loop_start()
client.publish(topicOnOff, payloadOn)
print('Client loop started.')
input('Press Enter to stop the client.')
client.loop_stop()