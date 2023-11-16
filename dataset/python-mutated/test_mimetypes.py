import base64
import unittest
from odoo.tools.mimetypes import guess_mimetype
PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVQI12P4//8/AAX+Av7czFnnAAAAAElFTkSuQmCC'
GIF = 'R0lGODdhAQABAIAAAP///////ywAAAAAAQABAAACAkQBADs='
BMP = 'Qk1+AAAAAAAAAHoAAABsAAAAAQAAAAEAAAABABgAAAAAAAQAAAATCwAAEwsAAAAAAAAAAAAAQkdScwAAAAAAAAAAAA\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAD///8A'
JPG = '/9j/4AAQSkZJRgABAQEASABIAAD//gATQ3JlYXRlZCB3aXRoIEdJTVD/2wBDAP\n//////////////////////////////////////////////////////////////////////////////////////2wBDAf///////\n///////////////////////////////////////////////////////////////////////////////wgARCAABAAEDAREAAhEB\nAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAAAv/EABQBAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEAAhADEAAAAUf/xAAUEAEAAAAAAAA\nAAAAAAAAAAAAA/9oACAEBAAEFAn//xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/AX//xAAUEQEAAAAAAAAAAAAAAAAAAA\nAA/9oACAECAQE/AX//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAY/An//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBA\nAE/IX//2gAMAwEAAgADAAAAEB//xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/EH//xAAUEQEAAAAAAAAAAAAAAAAAAAAA\n/9oACAECAQE/EH//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/EH//2Q=='

class test_guess_mimetype(unittest.TestCase):

    def test_default_mimetype_empty(self):
        if False:
            while True:
                i = 10
        mimetype = guess_mimetype('')
        self.assertIn(mimetype, ('application/octet-stream', 'application/x-empty'))

    def test_default_mimetype(self):
        if False:
            while True:
                i = 10
        mimetype = guess_mimetype('', default='test')
        self.assertIn(mimetype, ('test', 'application/x-empty'))

    def test_mimetype_octet_stream(self):
        if False:
            return 10
        mimetype = guess_mimetype('\x00')
        self.assertEqual(mimetype, 'application/octet-stream')

    def test_mimetype_png(self):
        if False:
            print('Hello World!')
        content = base64.b64decode(PNG)
        mimetype = guess_mimetype(content, default='test')
        self.assertEqual(mimetype, 'image/png')

    def test_mimetype_bmp(self):
        if False:
            print('Hello World!')
        content = base64.b64decode(BMP)
        mimetype = guess_mimetype(content, default='test')
        self.assertRegexpMatches(mimetype, 'image/.*\\bbmp')

    def test_mimetype_jpg(self):
        if False:
            print('Hello World!')
        content = base64.b64decode(JPG)
        mimetype = guess_mimetype(content, default='test')
        self.assertEqual(mimetype, 'image/jpeg')

    def test_mimetype_gif(self):
        if False:
            print('Hello World!')
        content = base64.b64decode(GIF)
        mimetype = guess_mimetype(content, default='test')
        self.assertEqual(mimetype, 'image/gif')
if __name__ == '__main__':
    unittest.main()