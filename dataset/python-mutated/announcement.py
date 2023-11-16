import argparse
import datetime
import json
import os
from tornado import escape, ioloop, web
from jupyterhub.services.auth import HubAuthenticated

class AnnouncementRequestHandler(HubAuthenticated, web.RequestHandler):
    """Dynamically manage page announcements"""

    def initialize(self, storage):
        if False:
            for i in range(10):
                print('nop')
        'Create storage for announcement text'
        self.storage = storage

    @web.authenticated
    def post(self):
        if False:
            print('Hello World!')
        'Update announcement'
        user = self.get_current_user()
        doc = escape.json_decode(self.request.body)
        self.storage['announcement'] = doc['announcement']
        self.storage['timestamp'] = datetime.datetime.now().isoformat()
        self.storage['user'] = user['name']
        self.write_to_json(self.storage)

    def get(self):
        if False:
            while True:
                i = 10
        'Retrieve announcement'
        self.write_to_json(self.storage)

    @web.authenticated
    def delete(self):
        if False:
            for i in range(10):
                print('nop')
        'Clear announcement'
        self.storage['announcement'] = ''
        self.write_to_json(self.storage)

    def write_to_json(self, doc):
        if False:
            print('Hello World!')
        'Write dictionary document as JSON'
        self.set_header('Content-Type', 'application/json; charset=UTF-8')
        self.write(escape.utf8(json.dumps(doc)))

def main():
    if False:
        print('Hello World!')
    args = parse_arguments()
    application = create_application(**vars(args))
    application.listen(args.port)
    ioloop.IOLoop.current().start()

def parse_arguments():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-prefix', '-a', default=os.environ.get('JUPYTERHUB_SERVICE_PREFIX', '/'), help='application API prefix')
    parser.add_argument('--port', '-p', default=8888, help='port for API to listen on', type=int)
    return parser.parse_args()

def create_application(api_prefix='/', handler=AnnouncementRequestHandler, **kwargs):
    if False:
        while True:
            i = 10
    storage = dict(announcement='', timestamp='', user='')
    return web.Application([(api_prefix, handler, dict(storage=storage))])
if __name__ == '__main__':
    main()