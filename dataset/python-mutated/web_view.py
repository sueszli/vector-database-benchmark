import abc
import json
import time
from queue import Queue
from flask import Response, jsonify, render_template, request
distinct_colors = ['#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#000075', '#808080', '#ffffff', '#000000']

class WebView(abc.ABC):

    def __init__(self, main_ui, web_app, route_prefix):
        if False:
            for i in range(10):
                print('nop')
        self.main_ui = main_ui
        self.app = web_app
        self.eventQueue = Queue()
        self.latlons = main_ui.image_manager.load_latlons()
        self.register_routes(route_prefix)
        self.route_prefix = route_prefix

    @abc.abstractclassmethod
    def sync_to_client(self):
        if False:
            i = 10
            return i + 15
        pass

    @abc.abstractclassmethod
    def process_client_message(self, data):
        if False:
            while True:
                i = 10
        pass

    def template_name(self):
        if False:
            i = 10
            return i + 15
        class_name = type(self).__name__
        return class_name

    def register_routes(self, route):
        if False:
            while True:
                i = 10

        def send_main_page():
            if False:
                i = 10
                return i + 15
            template = self.template_name()
            return render_template(f'{template}.html', class_name=template)
        self.app.add_url_rule(route, route + '_index', send_main_page)

        def postdata():
            if False:
                i = 10
                return i + 15
            data = request.get_json()
            if data['event'] != 'init':
                self.process_client_message(data)
            self.main_ui.sync_to_client()
            return jsonify(success=True)
        self.app.add_url_rule(route + '/postdata', route + '_postdata', postdata, methods=['POST'])

        def stream():
            if False:
                while True:
                    i = 10

            def eventStream():
                if False:
                    i = 10
                    return i + 15
                while True:
                    msg = self.eventQueue.get()
                    yield msg
            return Response(eventStream(), mimetype='text/event-stream')
        self.app.add_url_rule(route + '/stream', route + '_stream', stream)

    def send_sse_message(self, data, event_type='sync'):
        if False:
            print('Hello World!')
        data['time'] = time.time()
        sse_string = f'event: {event_type}\ndata: {json.dumps(data)}\n\n'
        self.eventQueue.put(sse_string)