__license__ = 'GPL 3'
__copyright__ = '2010, Hiroshi Miura <miurahr@linux.com>'
__docformat__ = 'restructuredtext en'
'\nDecode unicode text to an ASCII representation of the text for Japanese.\n Translate unicode string to ASCII roman string.\n\nAPI is based on the python unidecode,\nwhich is based on Ruby gem (http://rubyforge.org/projects/unidecode/)\nand  perl module Text::Unidecode\n(http://search.cpan.org/~sburke/Text-Unidecode-0.04/).\n\nThis functionality is owned by Kakasi Japanese processing engine.\n\nCopyright (c) 2010 Hiroshi Miura\n'
import re
from calibre.ebooks.unihandecode.unidecoder import Unidecoder
from calibre.ebooks.unihandecode.unicodepoints import CODEPOINTS
from calibre.ebooks.unihandecode.jacodepoints import CODEPOINTS as JACODES
from calibre.ebooks.unihandecode.pykakasi.kakasi import kakasi

class Jadecoder(Unidecoder):
    kakasi = None
    codepoints = {}

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.codepoints = CODEPOINTS
        self.codepoints.update(JACODES)
        self.kakasi = kakasi()

    def decode(self, text):
        if False:
            for i in range(10):
                print('nop')
        try:
            result = self.kakasi.do(text)
            return re.sub('[^\x00-\x7f]', lambda x: self.replace_point(x.group()), result)
        except:
            return re.sub('[^\x00-\x7f]', lambda x: self.replace_point(x.group()), text)