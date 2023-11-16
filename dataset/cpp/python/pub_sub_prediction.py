import paho.mqtt.client as mqtt
import random
import json

prediction="light_bright"
#MQTT parameterers
broker = 'test.mosquitto.org'
port = 1883

brightness=100
#Topics
topicOnOff = "/Licht/Wohnzimmer/OnOff"
topicBrightDim = "/Licht/Wohnzimmer/BrightDim"
topicBrightnessStatus="/Licht/Wohnzimmer/brightness"
#payoloads
payloadOn = json.dumps({"on": True})
payloadOff =json.dumps({"on": False})
payloadBrighter =json.dumps({"brightness":brightness})
payloadDim =json.dumps({"brightness":brightness})

# Create a new MQTT client
client = mqtt.Client()
# Subscribe to a topic
client.subscribe(topicBrightnessStatus)
# Define the on_message callback function
def on_message(client, userdata, message):
  print("Received message on topic:", message.topic)
  print("Message payload:", message.payload)

# Set the on_message callback
client.on_message = on_message
# Set parameters
client.connect(broker, port)
# Start the MQTT client loop
client.loop_start()

client.publish(topicOnOff, payloadOn)
# Code after the loop_start() method will be executed here
print("Client loop started.")

# Wait for user input
input("Press Enter to stop the client.")

# Stop the MQTT client loop
client.loop_stop()






