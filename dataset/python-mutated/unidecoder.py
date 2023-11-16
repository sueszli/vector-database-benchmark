__license__ = 'GPL 3'
__copyright__ = '2010, Hiroshi Miura <miurahr@linux.com>'
__docformat__ = 'restructuredtext en'
'\nDecode unicode text to an ASCII representation of the text in Chinese.\nTransliterate unicode characters to ASCII based on chinese pronounce.\n\nDerived from John Schember\'s unidecode library. Which was created\nas part of calibre.\n\nCopyright(c) 2009, John Schember <john@nachtimwald.com>\n\nBased on the ruby unidecode gem (http://rubyforge.org/projects/unidecode/) which\nis based on the perl module Text::Unidecode\n(http://search.cpan.org/~sburke/Text-Unidecode-0.04/). More information about\nunidecode can be found at\nhttp://interglacial.com/~sburke/tpj/as_html/tpj22.html.\n\nThe major differences between this implementation and others is it\'s written in\npython and it uses a single dictionary instead of loading the code group files\nas needed.\n\n\nCopyright (c) 2007 Russell Norris\n\nPermission is hereby granted, free of charge, to any person\nobtaining a copy of this software and associated documentation\nfiles (the "Software"), to deal in the Software without\nrestriction, including without limitation the rights to use,\ncopy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the\nSoftware is furnished to do so, subject to the following\nconditions:\n\nThe above copyright notice and this permission notice shall be\nincluded in all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,\nEXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES\nOF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND\nNONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT\nHOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,\nWHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING\nFROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR\nOTHER DEALINGS IN THE SOFTWARE.\n\n\nCopyright 2001, Sean M. Burke <sburke@cpan.org>, all rights reserved.\n\nThe programs and documentation in this dist are distributed in the\nhope that they will be useful, but without any warranty; without even\nthe implied warranty of merchantability or fitness for a particular\npurpose.\n\nThis library is free software; you can redistribute it and/or modify\nit under the same terms as Perl itself.\n'
import re
from calibre.ebooks.unihandecode.unicodepoints import CODEPOINTS
from calibre.ebooks.unihandecode.zhcodepoints import CODEPOINTS as HANCODES

class Unidecoder:
    codepoints = {}

    def __init__(self):
        if False:
            return 10
        self.codepoints = CODEPOINTS
        self.codepoints.update(HANCODES)

    def decode(self, text):
        if False:
            while True:
                i = 10
        return re.sub('[^\x00-\x7f]', lambda x: self.replace_point(x.group()), text)

    def replace_point(self, codepoint):
        if False:
            while True:
                i = 10
        '\n        Returns the replacement character or ? if none can be found.\n        '
        try:
            return self.codepoints[self.code_group(codepoint)][self.grouped_point(codepoint)]
        except:
            return '?'

    def code_group(self, character):
        if False:
            for i in range(10):
                print('nop')
        '\n        Find what group character is a part of.\n        '
        if not isinstance(character, str):
            character = str(character, 'utf-8')
        return 'x%02x' % (ord(character) >> 8)

    def grouped_point(self, character):
        if False:
            while True:
                i = 10
        '\n        Return the location the replacement character is in the list for a\n        the group character is a part of.\n        '
        if not isinstance(character, str):
            character = str(character, 'utf-8')
        return ord(character) & 255