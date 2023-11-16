from __future__ import print_function
import json
import thread
import time
from socket import error as socket_error
import paho.mqtt.client as mqtt
from pokemongo_bot.event_manager import EventHandler
DEBUG_ON = False

class MyMQTTClass:
    MAX_RESULTS = 50

    def __init__(self, bot, clientid=None):
        if False:
            while True:
                i = 10
        self.bot = bot
        self.client_id = clientid
        self.bot.mqtt_pokemon_list = []
        self._mqttc = None

    def mqtt_on_connect(self, mqttc, obj, flags, rc):
        if False:
            print('Hello World!')
        if rc is 0:
            self._mqttc.subscribe('pgo/#', 1)
        if DEBUG_ON:
            print('rc: ' + str(rc))

    def mqtt_on_message(self, mqttc, obj, msg):
        if False:
            return 10
        if DEBUG_ON:
            print('on message: {}'.format(msg.payload))
        pokemon = json.loads(msg.payload)
        if pokemon and 'encounter_id' in pokemon:
            new_list = [x for x in self.bot.mqtt_pokemon_list if x['encounter_id'] is pokemon['encounter_id']]
            if not (new_list and len(new_list) > 0):
                if len(self.bot.mqtt_pokemon_list) > self.MAX_RESULTS:
                    del self.bot.mqtt_pokemon_list[:]
                self.bot.mqtt_pokemon_list.append(pokemon)

    def on_disconnect(self, client, userdata, rc):
        if False:
            while True:
                i = 10
        self._mqttc.unsubscribe('pgo/#')
        if DEBUG_ON:
            print('on_disconnect')
            if rc != 0:
                print('Unexpected disconnection.')

    def mqtt_on_publish(self, mqttc, obj, mid):
        if False:
            while True:
                i = 10
        if DEBUG_ON:
            print('mid: ' + str(mid))

    def mqtt_on_subscribe(self, mqttc, obj, mid, granted_qos):
        if False:
            return 10
        if DEBUG_ON:
            print('Subscribed: ' + str(mid) + ' ' + str(granted_qos))

    def publish(self, channel, message):
        if False:
            i = 10
            return i + 15
        if self._mqttc:
            try:
                self._mqttc.publish(channel, message)
            except UnicodeDecodeError:
                pass

    def initialize(self):
        if False:
            while True:
                i = 10
        try:
            if DEBUG_ON:
                print('connect again')
            self._mqttc = mqtt.Client(None)
            self._mqttc.on_message = self.mqtt_on_message
            self._mqttc.on_connect = self.mqtt_on_connect
            self._mqttc.on_subscribe = self.mqtt_on_subscribe
            self._mqttc.on_publish = self.mqtt_on_publish
            self._mqttc.on_disconnect = self.on_disconnect
        except TypeError:
            print('Connect to mqtter error')
            return

    def run(self):
        if False:
            print('Hello World!')
        try:
            self._mqttc.connect('broker.pikabot.org', 1883, 20)
        except:
            print('Error occured in social handler')
        while True:
            try:
                self._mqttc.loop_forever(timeout=30.0, max_packets=100, retry_first_connection=False)
                print('Oops disconnected ?')
                time.sleep(20)
            except UnicodeDecodeError:
                time.sleep(1)
            except Exception as e:
                print(e)
                time.sleep(10)

class SocialHandler(EventHandler):

    def __init__(self, bot):
        if False:
            print('Hello World!')
        super(SocialHandler, self).__init__()
        self.bot = bot
        self.mqttc = None

    def handle_event(self, event, sender, level, formatted_msg, data):
        if False:
            return 10
        if self.mqttc is None:
            try:
                if DEBUG_ON:
                    print('need connect')
                self.mqttc = MyMQTTClass(self.bot, self.bot.config.client_id)
                self.mqttc.initialize()
                self.bot.mqttc = self.mqttc
                thread.start_new_thread(self.mqttc.run)
            except socket_error as serr:
                self.mqttc = None
                return
        if event == 'catchable_pokemon' and 'pokemon_id' in data:
            data_string = '%s, %s, %s, %s, %s' % (str(data['latitude']), str(data['longitude']), str(data['pokemon_id']), str(data['expiration_timestamp_ms']), str(data['pokemon_name']))
            self.mqttc.publish('pgomapcatch/all/catchable/' + str(data['pokemon_id']), data_string)
            json_data = json.dumps(data)
            self.mqttc.publish('pgo/all/catchable/' + str(data['pokemon_id']), json_data)