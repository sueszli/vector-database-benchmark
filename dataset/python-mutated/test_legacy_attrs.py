"""
Python Markdown

A Python implementation of John Gruber's Markdown.

Documentation: https://python-markdown.github.io/
GitHub: https://github.com/Python-Markdown/markdown/
PyPI: https://pypi.org/project/Markdown/

Started by Manfred Stienstra (http://www.dwerg.net/).
Maintained for a few years by Yuri Takhteyev (http://www.freewisdom.org).
Currently maintained by Waylan Limberg (https://github.com/waylan),
Dmitry Shachnev (https://github.com/mitya57) and Isaac Muse (https://github.com/facelessuser).

Copyright 2007-2023 The Python Markdown Project (v. 1.7 and later)
Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
Copyright 2004 Manfred Stienstra (the original version)

License: BSD (see LICENSE.md for details).
"""
from markdown.test_tools import TestCase

class TestLegacyAtrributes(TestCase):
    maxDiff = None

    def testLegacyAttrs(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders(self.dedent('\n                # Header {@id=inthebeginning}\n\n                Now, let\'s try something *inline{@class=special}*, to see if it works\n\n                @id=TABLE.OF.CONTENTS}\n\n\n                * {@id=TABLEOFCONTENTS}\n\n\n                Or in the middle of the text {@id=TABLEOFCONTENTS}\n\n                {@id=tableofcontents}\n\n                [![{@style=float: left; margin: 10px; border:\n                none;}](http://fourthought.com/images/ftlogo.png "Fourthought\n                logo")](http://fourthought.com/)\n\n                ![img{@id=foo}][img]\n\n                [img]: http://example.com/i.jpg\n            '), self.dedent('\n                <h1 id="inthebeginning">Header </h1>\n                <p>Now, let\'s try something <em class="special">inline</em>, to see if it works</p>\n                <p>@id=TABLE.OF.CONTENTS}</p>\n                <ul>\n                <li id="TABLEOFCONTENTS"></li>\n                </ul>\n                <p id="TABLEOFCONTENTS">Or in the middle of the text </p>\n                <p id="tableofcontents"></p>\n                <p><a href="http://fourthought.com/"><img alt="" src="http://fourthought.com/images/ftlogo.png" style="float: left; margin: 10px; border: none;" title="Fourthought logo" /></a></p>\n                <p><img alt="img" id="foo" src="http://example.com/i.jpg" /></p>\n            '), extensions=['legacy_attrs'])