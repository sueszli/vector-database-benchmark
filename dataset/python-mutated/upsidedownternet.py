from cStringIO import StringIO
from plugins.plugin import Plugin
from PIL import Image, ImageFile

class Upsidedownternet(Plugin):
    name = 'Upsidedownternet'
    optname = 'upsidedownternet'
    desc = 'Flips images 180 degrees'
    version = '0.1'

    def initialize(self, options):
        if False:
            i = 10
            return i + 15
        self.options = options

    def responseheaders(self, response, request):
        if False:
            for i in range(10):
                print('nop')
        "Kill the image skipping that's in place for speed reasons"
        if request.isImageRequest:
            request.isImageRequest = False
            request.isImage = True
            self.imageType = response.responseHeaders.getRawHeaders('content-type')[0].split('/')[1].upper()

    def response(self, response, request, data):
        if False:
            for i in range(10):
                print('nop')
        try:
            isImage = getattr(request, 'isImage')
        except AttributeError:
            isImage = False
        if isImage:
            try:
                p = ImageFile.Parser()
                p.feed(data)
                im = p.close()
                im = im.transpose(Image.ROTATE_180)
                output = StringIO()
                im.save(output, format=self.imageType)
                data = output.getvalue()
                output.close()
                self.clientlog.info('Flipped image', extra=request.clientInfo)
            except Exception as e:
                self.clientlog.info('Error: {}'.format(e), extra=request.clientInfo)
        return {'response': response, 'request': request, 'data': data}