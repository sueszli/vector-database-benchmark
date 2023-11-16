"""
Math extension for Python-Markdown
==================================

Adds support for displaying math formulas using
[MathJax](https://www.mathjax.org/).

Author: 2015-2017, Dmitry Shachnev <mitya57@gmail.com>.
"""
'\nCopyright 2015-2017 Dmitry Shachnev <mitya57@gmail.com>.\nAll rights reserved.\n\nRedistribution and use in source and binary forms, with or without\nmodification, are permitted provided that the following conditions\nare met:\n\n1. Redistributions of source code must retain the above copyright\n   notice, this list of conditions and the following disclaimer.\n2. Redistributions in binary form must reproduce the above copyright\n   notice, this list of conditions and the following disclaimer in the\n   documentation and/or other materials provided with the distribution.\n3. Neither the name of the author nor the names of its contributors\n   may be used to endorse or promote products derived from this software\n   without specific prior written permission.\n\nTHIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"\nAND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE\nIMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE\nARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE\nFOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL\nDAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS\nOR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)\nHOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT\nLIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY\nOUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF\nSUCH DAMAGE.\n'
from markdown.extensions import Extension
from markdown.inlinepatterns import Pattern
from markdown.util import AtomicString, etree

class MathExtension(Extension):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.config = {'enable_dollar_delimiter': [False, 'Enable single-dollar delimiter'], 'add_preview': [False, 'Add a preview node before each math node']}
        super(MathExtension, self).__init__(*args, **kwargs)

    def extendMarkdown(self, md, md_globals):
        if False:
            return 10

        def _wrap_node(node, preview_text, wrapper_tag):
            if False:
                while True:
                    i = 10
            if not self.getConfig('add_preview'):
                return node
            preview = etree.Element('span', {'class': 'MathJax_Preview'})
            preview.text = AtomicString(preview_text)
            wrapper = etree.Element(wrapper_tag)
            wrapper.extend([preview, node])
            return wrapper

        def handle_match_inline(m):
            if False:
                while True:
                    i = 10
            node = etree.Element('script')
            node.set('type', 'math/tex')
            node.text = AtomicString(m.group(3))
            return _wrap_node(node, ''.join(m.group(2, 3, 4)), 'span')

        def handle_match(m):
            if False:
                print('Hello World!')
            node = etree.Element('script')
            node.set('type', 'math/tex; mode=display')
            if '\\begin' in m.group(2):
                node.text = AtomicString(''.join(m.group(2, 4, 5)))
                return _wrap_node(node, ''.join(m.group(1, 2, 4, 5, 6)), 'div')
            else:
                node.text = AtomicString(m.group(3))
                return _wrap_node(node, ''.join(m.group(2, 3, 4)), 'div')
        inlinemathpatterns = (Pattern('(?<!\\\\|\\$)(\\$)([^\\$]+)(\\$)'), Pattern('(?<!\\\\)(\\\\\\()(.+?)(\\\\\\))'))
        mathpatterns = (Pattern('(?<!\\\\)(\\$\\$)([^\\$]+)(\\$\\$)'), Pattern('(?<!\\\\)(\\\\\\[)(.+?)(\\\\\\])'), Pattern('(?<!\\\\)(\\\\begin{([a-z]+?\\*?)})(.+?)(\\\\end{\\3})'))
        if not self.getConfig('enable_dollar_delimiter'):
            inlinemathpatterns = inlinemathpatterns[1:]
        for (i, pattern) in enumerate(inlinemathpatterns):
            pattern.handleMatch = handle_match_inline
            md.inlinePatterns.add('math-inline-%d' % i, pattern, '<escape')
        for (i, pattern) in enumerate(mathpatterns):
            pattern.handleMatch = handle_match
            md.inlinePatterns.add('math-%d' % i, pattern, '<escape')

def makeExtension(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return MathExtension(*args, **kwargs)