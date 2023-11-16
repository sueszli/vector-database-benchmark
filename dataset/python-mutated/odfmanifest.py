import zipfile
from xml.sax import make_parser, handler
from xml.sax.xmlreader import InputSource
import io
MANIFESTNS = 'urn:oasis:names:tc:opendocument:xmlns:manifest:1.0'

class ODFManifestHandler(handler.ContentHandler):
    """ The ODFManifestHandler parses a manifest file and produces a list of
        content """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.manifest = {}
        self.elements = {(MANIFESTNS, 'file-entry'): (self.s_file_entry, self.donothing)}

    def handle_starttag(self, tag, method, attrs):
        if False:
            return 10
        method(tag, attrs)

    def handle_endtag(self, tag, method):
        if False:
            return 10
        method(tag)

    def startElementNS(self, tag, qname, attrs):
        if False:
            print('Hello World!')
        method = self.elements.get(tag, (None, None))[0]
        if method:
            self.handle_starttag(tag, method, attrs)
        else:
            self.unknown_starttag(tag, attrs)

    def endElementNS(self, tag, qname):
        if False:
            while True:
                i = 10
        method = self.elements.get(tag, (None, None))[1]
        if method:
            self.handle_endtag(tag, method)
        else:
            self.unknown_endtag(tag)

    def unknown_starttag(self, tag, attrs):
        if False:
            while True:
                i = 10
        pass

    def unknown_endtag(self, tag):
        if False:
            print('Hello World!')
        pass

    def donothing(self, tag, attrs=None):
        if False:
            i = 10
            return i + 15
        pass

    def s_file_entry(self, tag, attrs):
        if False:
            for i in range(10):
                print('nop')
        m = attrs.get((MANIFESTNS, 'media-type'), 'application/octet-stream')
        p = attrs.get((MANIFESTNS, 'full-path'))
        self.manifest[p] = {'media-type': m, 'full-path': p}

def manifestlist(manifestxml):
    if False:
        for i in range(10):
            print('nop')
    odhandler = ODFManifestHandler()
    parser = make_parser()
    parser.setFeature(handler.feature_namespaces, 1)
    parser.setContentHandler(odhandler)
    parser.setErrorHandler(handler.ErrorHandler())
    inpsrc = InputSource()
    inpsrc.setByteStream(io.BytesIO(manifestxml))
    parser.setFeature(handler.feature_external_ges, False)
    parser.parse(inpsrc)
    return odhandler.manifest

def odfmanifest(odtfile):
    if False:
        print('Hello World!')
    z = zipfile.ZipFile(odtfile)
    manifest = z.read('META-INF/manifest.xml')
    z.close()
    return manifestlist(manifest)
if __name__ == '__main__':
    import sys
    result = odfmanifest(sys.argv[1])
    for file in result.values():
        print('%-40s %-40s' % (file['media-type'], file['full-path']))