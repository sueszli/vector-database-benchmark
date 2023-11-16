__license__ = 'GPL 3'
__copyright__ = '2010, Hiroshi Miura <miurahr@linux.com>'
__docformat__ = 'restructuredtext en'
'\nDecode unicode text to an ASCII representation of the text in Vietnamese.\n\n'
from calibre.ebooks.unihandecode.unidecoder import Unidecoder
from calibre.ebooks.unihandecode.vncodepoints import CODEPOINTS as HANCODES
from calibre.ebooks.unihandecode.unicodepoints import CODEPOINTS

class Vndecoder(Unidecoder):
    codepoints = {}

    def __init__(self):
        if False:
            return 10
        self.codepoints = CODEPOINTS
        self.codepoints.update(HANCODES)