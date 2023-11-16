import unittest
from mkdocs.structure.toc import get_toc
from mkdocs.tests.base import dedent, get_markdown_toc

class TableOfContentsTests(unittest.TestCase):

    def test_indented_toc(self):
        if False:
            return 10
        md = dedent('\n            # Heading 1\n            ## Heading 2\n            ### Heading 3\n            ')
        expected = dedent('\n            Heading 1 - #heading-1\n                Heading 2 - #heading-2\n                    Heading 3 - #heading-3\n            ')
        toc = get_toc(get_markdown_toc(md))
        self.assertEqual(str(toc).strip(), expected)
        self.assertEqual(len(toc), 1)

    def test_indented_toc_html(self):
        if False:
            for i in range(10):
                print('nop')
        md = dedent('\n            # Heading 1\n            ## <code>Heading</code> 2\n            ## Heading 3\n            ')
        expected = dedent('\n            Heading 1 - #heading-1\n                Heading 2 - #heading-2\n                Heading 3 - #heading-3\n            ')
        toc = get_toc(get_markdown_toc(md))
        self.assertEqual(str(toc).strip(), expected)
        self.assertEqual(len(toc), 1)

    def test_flat_toc(self):
        if False:
            i = 10
            return i + 15
        md = dedent('\n            # Heading 1\n            # Heading 2\n            # Heading 3\n            ')
        expected = dedent('\n            Heading 1 - #heading-1\n            Heading 2 - #heading-2\n            Heading 3 - #heading-3\n            ')
        toc = get_toc(get_markdown_toc(md))
        self.assertEqual(str(toc).strip(), expected)
        self.assertEqual(len(toc), 3)

    def test_flat_h2_toc(self):
        if False:
            for i in range(10):
                print('nop')
        md = dedent('\n            ## Heading 1\n            ## Heading 2\n            ## Heading 3\n            ')
        expected = dedent('\n            Heading 1 - #heading-1\n            Heading 2 - #heading-2\n            Heading 3 - #heading-3\n            ')
        toc = get_toc(get_markdown_toc(md))
        self.assertEqual(str(toc).strip(), expected)
        self.assertEqual(len(toc), 3)

    def test_mixed_toc(self):
        if False:
            return 10
        md = dedent('\n            # Heading 1\n            ## Heading 2\n            # Heading 3\n            ### Heading 4\n            ### Heading 5\n            ')
        expected = dedent('\n            Heading 1 - #heading-1\n                Heading 2 - #heading-2\n            Heading 3 - #heading-3\n                Heading 4 - #heading-4\n                Heading 5 - #heading-5\n            ')
        toc = get_toc(get_markdown_toc(md))
        self.assertEqual(str(toc).strip(), expected)
        self.assertEqual(len(toc), 2)

    def test_mixed_html(self):
        if False:
            while True:
                i = 10
        md = dedent('\n            # Heading 1\n            ## Heading 2\n            # Heading 3\n            ### Heading 4\n            ### <a>Heading 5</a>\n            ')
        expected = dedent('\n            Heading 1 - #heading-1\n                Heading 2 - #heading-2\n            Heading 3 - #heading-3\n                Heading 4 - #heading-4\n                Heading 5 - #heading-5\n            ')
        toc = get_toc(get_markdown_toc(md))
        self.assertEqual(str(toc).strip(), expected)
        self.assertEqual(len(toc), 2)

    def test_nested_anchor(self):
        if False:
            print('Hello World!')
        md = dedent('\n            # Heading 1\n            ## Heading 2\n            # Heading 3\n            ### Heading 4\n            ### <a href="/">Heading 5</a>\n            ')
        expected = dedent('\n            Heading 1 - #heading-1\n                Heading 2 - #heading-2\n            Heading 3 - #heading-3\n                Heading 4 - #heading-4\n                Heading 5 - #heading-5\n            ')
        toc = get_toc(get_markdown_toc(md))
        self.assertEqual(str(toc).strip(), expected)
        self.assertEqual(len(toc), 2)

    def test_entityref(self):
        if False:
            i = 10
            return i + 15
        md = dedent('\n            # Heading & 1\n            ## Heading > 2\n            ### Heading < 3\n            ')
        expected = dedent('\n            Heading &amp; 1 - #heading-1\n                Heading &gt; 2 - #heading-2\n                    Heading &lt; 3 - #heading-3\n            ')
        toc = get_toc(get_markdown_toc(md))
        self.assertEqual(str(toc).strip(), expected)
        self.assertEqual(len(toc), 1)

    def test_charref(self):
        if False:
            print('Hello World!')
        md = '# &#64;Header'
        expected = '&#64;Header - #header'
        toc = get_toc(get_markdown_toc(md))
        self.assertEqual(str(toc).strip(), expected)
        self.assertEqual(len(toc), 1)

    def test_level(self):
        if False:
            return 10
        md = dedent('\n            # Heading 1\n            ## Heading 1.1\n            ### Heading 1.1.1\n            ### Heading 1.1.2\n            ## Heading 1.2\n            ')
        toc = get_toc(get_markdown_toc(md))

        def get_level_sequence(items):
            if False:
                return 10
            for item in items:
                yield item.level
                yield from get_level_sequence(item.children)
        self.assertEqual(tuple(get_level_sequence(toc)), (1, 2, 3, 3, 2))