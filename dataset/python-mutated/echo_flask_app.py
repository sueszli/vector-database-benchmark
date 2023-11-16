from flask import request, Flask
from st2reactor.sensor.base import Sensor

class EchoFlaskSensor(Sensor):

    def __init__(self, sensor_service, config):
        if False:
            for i in range(10):
                print('nop')
        super(EchoFlaskSensor, self).__init__(sensor_service=sensor_service, config=config)
        self._host = '127.0.0.1'
        self._port = 5000
        self._path = '/echo'
        self._log = self._sensor_service.get_logger(__name__)
        self._app = Flask(__name__)

    def setup(self):
        if False:
            return 10
        pass

    def run(self):
        if False:
            for i in range(10):
                print('nop')

        @self._app.route(self._path, methods=['POST'])
        def echo():
            if False:
                while True:
                    i = 10
            payload = request.get_json(force=True)
            self._sensor_service.dispatch(trigger='examples.echoflasksensor', payload=payload)
            return request.data
        self._log.info('Listening for payload on http://{}:{}{}'.format(self._host, self._port, self._path))
        self._app.run(host=self._host, port=self._port, threaded=False)

    def cleanup(self):
        if False:
            while True:
                i = 10
        pass

    def add_trigger(self, trigger):
        if False:
            print('Hello World!')
        pass

    def update_trigger(self, trigger):
        if False:
            while True:
                i = 10
        pass

    def remove_trigger(self, trigger):
        if False:
            return 10
        pass