import random
import os
from plugins.plugin import Plugin

class ImageRandomizer(Plugin):
    name = 'ImageRandomizer'
    optname = 'imgrand'
    desc = 'Replaces images with a random one from a specified directory'
    version = '0.1'

    def initialize(self, options):
        if False:
            print('Hello World!')
        self.options = options
        self.img_dir = options.img_dir

    def responseheaders(self, response, request):
        if False:
            while True:
                i = 10
        "Kill the image skipping that's in place for speed reasons"
        if request.isImageRequest:
            request.isImageRequest = False
            request.isImage = True
            self.imageType = response.responseHeaders.getRawHeaders('content-type')[0].split('/')[1].upper()

    def response(self, response, request, data):
        if False:
            return 10
        try:
            isImage = getattr(request, 'isImage')
        except AttributeError:
            isImage = False
        if isImage:
            try:
                img = random.choice(os.listdir(self.options.img_dir))
                with open(os.path.join(self.options.img_dir, img), 'rb') as img_file:
                    data = img_file.read()
                    self.clientlog.info('Replaced image with {}'.format(img), extra=request.clientInfo)
                    return {'response': response, 'request': request, 'data': data}
            except Exception as e:
                self.clientlog.info('Error: {}'.format(e), extra=request.clientInfo)

    def options(self, options):
        if False:
            while True:
                i = 10
        options.add_argument('--img-dir', type=str, metavar='DIRECTORY', help='Directory with images')