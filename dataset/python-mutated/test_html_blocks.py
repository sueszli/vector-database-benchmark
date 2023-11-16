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
import markdown

class TestHTMLBlocks(TestCase):

    def test_raw_paragraph(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders('<p>A raw paragraph.</p>', '<p>A raw paragraph.</p>')

    def test_raw_skip_inline_markdown(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders('<p>A *raw* paragraph.</p>', '<p>A *raw* paragraph.</p>')

    def test_raw_indent_one_space(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(' <p>A *raw* paragraph.</p>', '<p>A *raw* paragraph.</p>')

    def test_raw_indent_two_spaces(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders('  <p>A *raw* paragraph.</p>', '<p>A *raw* paragraph.</p>')

    def test_raw_indent_three_spaces(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders('   <p>A *raw* paragraph.</p>', '<p>A *raw* paragraph.</p>')

    def test_raw_indent_four_spaces(self):
        if False:
            return 10
        self.assertMarkdownRenders('    <p>code block</p>', self.dedent('\n                <pre><code>&lt;p&gt;code block&lt;/p&gt;\n                </code></pre>\n                '))

    def test_raw_span(self):
        if False:
            return 10
        self.assertMarkdownRenders('<span>*inline*</span>', '<p><span><em>inline</em></span></p>')

    def test_code_span(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders('`<p>code span</p>`', '<p><code>&lt;p&gt;code span&lt;/p&gt;</code></p>')

    def test_code_span_open_gt(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders('*bar* `<` *foo*', '<p><em>bar</em> <code>&lt;</code> <em>foo</em></p>')

    def test_raw_empty(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders('<p></p>', '<p></p>')

    def test_raw_empty_space(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders('<p> </p>', '<p> </p>')

    def test_raw_empty_newline(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders('<p>\n</p>', '<p>\n</p>')

    def test_raw_empty_blank_line(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders('<p>\n\n</p>', '<p>\n\n</p>')

    def test_raw_uppercase(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders('<DIV>*foo*</DIV>', '<DIV>*foo*</DIV>')

    def test_raw_uppercase_multiline(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                <DIV>\n                *foo*\n                </DIV>\n                '), self.dedent('\n                <DIV>\n                *foo*\n                </DIV>\n                '))

    def test_multiple_raw_single_line(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders('<p>*foo*</p><div>*bar*</div>', self.dedent('\n                <p>*foo*</p>\n                <div>*bar*</div>\n                '))

    def test_multiple_raw_single_line_with_pi(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders("<p>*foo*</p><?php echo '>'; ?>", self.dedent("\n                <p>*foo*</p>\n                <?php echo '>'; ?>\n                "))

    def test_multiline_raw(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                <p>\n                    A raw paragraph\n                    with multiple lines.\n                </p>\n                '), self.dedent('\n                <p>\n                    A raw paragraph\n                    with multiple lines.\n                </p>\n                '))

    def test_blank_lines_in_raw(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                <p>\n\n                    A raw paragraph...\n\n                    with many blank lines.\n\n                </p>\n                '), self.dedent('\n                <p>\n\n                    A raw paragraph...\n\n                    with many blank lines.\n\n                </p>\n                '))

    def test_raw_surrounded_by_Markdown(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                Some *Markdown* text.\n\n                <p>*Raw* HTML.</p>\n\n                More *Markdown* text.\n                '), self.dedent('\n                <p>Some <em>Markdown</em> text.</p>\n                <p>*Raw* HTML.</p>\n\n                <p>More <em>Markdown</em> text.</p>\n                '))

    def test_raw_surrounded_by_text_without_blank_lines(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders(self.dedent('\n                Some *Markdown* text.\n                <p>*Raw* HTML.</p>\n                More *Markdown* text.\n                '), self.dedent('\n                <p>Some <em>Markdown</em> text.</p>\n                <p>*Raw* HTML.</p>\n                <p>More <em>Markdown</em> text.</p>\n                '))

    def test_multiline_markdown_with_code_span(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders(self.dedent('\n                A paragraph with a block-level\n                `<p>code span</p>`, which is\n                at the start of a line.\n                '), self.dedent('\n                <p>A paragraph with a block-level\n                <code>&lt;p&gt;code span&lt;/p&gt;</code>, which is\n                at the start of a line.</p>\n                '))

    def test_raw_block_preceded_by_markdown_code_span_with_unclosed_block_tag(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                A paragraph with a block-level code span: `<div>`.\n\n                <p>*not markdown*</p>\n\n                This is *markdown*\n                '), self.dedent('\n                <p>A paragraph with a block-level code span: <code>&lt;div&gt;</code>.</p>\n                <p>*not markdown*</p>\n\n                <p>This is <em>markdown</em></p>\n                '))

    def test_raw_one_line_followed_by_text(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders('<p>*foo*</p>*bar*', self.dedent('\n                <p>*foo*</p>\n                <p><em>bar</em></p>\n                '))

    def test_raw_one_line_followed_by_span(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders('<p>*foo*</p><span>*bar*</span>', self.dedent('\n                <p>*foo*</p>\n                <p><span><em>bar</em></span></p>\n                '))

    def test_raw_with_markdown_blocks(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                <div>\n                    Not a Markdown paragraph.\n\n                    * Not a list item.\n                    * Another non-list item.\n\n                    Another non-Markdown paragraph.\n                </div>\n                '), self.dedent('\n                <div>\n                    Not a Markdown paragraph.\n\n                    * Not a list item.\n                    * Another non-list item.\n\n                    Another non-Markdown paragraph.\n                </div>\n                '))

    def test_adjacent_raw_blocks(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n                <p>A raw paragraph.</p>\n                <p>A second raw paragraph.</p>\n                '), self.dedent('\n                <p>A raw paragraph.</p>\n                <p>A second raw paragraph.</p>\n                '))

    def test_adjacent_raw_blocks_with_blank_lines(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                <p>A raw paragraph.</p>\n\n                <p>A second raw paragraph.</p>\n                '), self.dedent('\n                <p>A raw paragraph.</p>\n\n                <p>A second raw paragraph.</p>\n                '))

    def test_nested_raw_one_line(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders('<div><p>*foo*</p></div>', '<div><p>*foo*</p></div>')

    def test_nested_raw_block(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders(self.dedent('\n                <div>\n                <p>A raw paragraph.</p>\n                </div>\n                '), self.dedent('\n                <div>\n                <p>A raw paragraph.</p>\n                </div>\n                '))

    def test_nested_indented_raw_block(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                <div>\n                    <p>A raw paragraph.</p>\n                </div>\n                '), self.dedent('\n                <div>\n                    <p>A raw paragraph.</p>\n                </div>\n                '))

    def test_nested_raw_blocks(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders(self.dedent('\n                <div>\n                <p>A raw paragraph.</p>\n                <p>A second raw paragraph.</p>\n                </div>\n                '), self.dedent('\n                <div>\n                <p>A raw paragraph.</p>\n                <p>A second raw paragraph.</p>\n                </div>\n                '))

    def test_nested_raw_blocks_with_blank_lines(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                <div>\n\n                <p>A raw paragraph.</p>\n\n                <p>A second raw paragraph.</p>\n\n                </div>\n                '), self.dedent('\n                <div>\n\n                <p>A raw paragraph.</p>\n\n                <p>A second raw paragraph.</p>\n\n                </div>\n                '))

    def test_nested_inline_one_line(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders('<p><em>foo</em><br></p>', '<p><em>foo</em><br></p>')

    def test_raw_nested_inline(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders(self.dedent('\n                <div>\n                    <p>\n                        <span>*text*</span>\n                    </p>\n                </div>\n                '), self.dedent('\n                <div>\n                    <p>\n                        <span>*text*</span>\n                    </p>\n                </div>\n                '))

    def test_raw_nested_inline_with_blank_lines(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n                <div>\n\n                    <p>\n\n                        <span>*text*</span>\n\n                    </p>\n\n                </div>\n                '), self.dedent('\n                <div>\n\n                    <p>\n\n                        <span>*text*</span>\n\n                    </p>\n\n                </div>\n                '))

    def test_raw_html5(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                <section>\n                    <header>\n                        <hgroup>\n                            <h1>Hello :-)</h1>\n                        </hgroup>\n                    </header>\n                    <figure>\n                        <img src="image.png" alt="" />\n                        <figcaption>Caption</figcaption>\n                    </figure>\n                    <footer>\n                        <p>Some footer</p>\n                    </footer>\n                </section>\n                '), self.dedent('\n                <section>\n                    <header>\n                        <hgroup>\n                            <h1>Hello :-)</h1>\n                        </hgroup>\n                    </header>\n                    <figure>\n                        <img src="image.png" alt="" />\n                        <figcaption>Caption</figcaption>\n                    </figure>\n                    <footer>\n                        <p>Some footer</p>\n                    </footer>\n                </section>\n                '))

    def test_raw_pre_tag(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent("\n                Preserve whitespace in raw html\n\n                <pre>\n                class Foo():\n                    bar = 'bar'\n\n                    @property\n                    def baz(self):\n                        return self.bar\n                </pre>\n                "), self.dedent("\n                <p>Preserve whitespace in raw html</p>\n                <pre>\n                class Foo():\n                    bar = 'bar'\n\n                    @property\n                    def baz(self):\n                        return self.bar\n                </pre>\n                "))

    def test_raw_pre_tag_nested_escaped_html(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                <pre>\n                &lt;p&gt;foo&lt;/p&gt;\n                </pre>\n                '), self.dedent('\n                <pre>\n                &lt;p&gt;foo&lt;/p&gt;\n                </pre>\n                '))

    def test_raw_p_no_end_tag(self):
        if False:
            return 10
        self.assertMarkdownRenders('<p>*text*', '<p>*text*')

    def test_raw_multiple_p_no_end_tag(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent("\n                <p>*text*'\n\n                <p>more *text*\n                "), self.dedent("\n                <p>*text*'\n\n                <p>more *text*\n                "))

    def test_raw_p_no_end_tag_followed_by_blank_line(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders(self.dedent("\n                <p>*raw text*'\n\n                Still part of *raw* text.\n                "), self.dedent("\n                <p>*raw text*'\n\n                Still part of *raw* text.\n                "))

    def test_raw_nested_p_no_end_tag(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders('<div><p>*text*</div>', '<div><p>*text*</div>')

    def test_raw_open_bracket_only(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders('<', '<p>&lt;</p>')

    def test_raw_open_bracket_followed_by_space(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders('< foo', '<p>&lt; foo</p>')

    def test_raw_missing_close_bracket(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders('<foo', '<p>&lt;foo</p>')

    def test_raw_unclosed_tag_in_code_span(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                `<div`.\n\n                <div>\n                hello\n                </div>\n                '), self.dedent('\n                <p><code>&lt;div</code>.</p>\n                <div>\n                hello\n                </div>\n                '))

    def test_raw_unclosed_tag_in_code_span_space(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                ` <div `.\n\n                <div>\n                hello\n                </div>\n                '), self.dedent('\n                <p><code>&lt;div</code>.</p>\n                <div>\n                hello\n                </div>\n                '))

    def test_raw_attributes(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders('<p id="foo", class="bar baz", style="margin: 15px; line-height: 1.5; text-align: center;">text</p>', '<p id="foo", class="bar baz", style="margin: 15px; line-height: 1.5; text-align: center;">text</p>')

    def test_raw_attributes_nested(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                <div id="foo, class="bar", style="background: #ffe7e8; border: 2px solid #e66465;">\n                    <p id="baz", style="margin: 15px; line-height: 1.5; text-align: center;">\n                        <img scr="../foo.jpg" title="with \'quoted\' text." valueless_attr weirdness="<i>foo</i>" />\n                    </p>\n                </div>\n                '), self.dedent('\n                <div id="foo, class="bar", style="background: #ffe7e8; border: 2px solid #e66465;">\n                    <p id="baz", style="margin: 15px; line-height: 1.5; text-align: center;">\n                        <img scr="../foo.jpg" title="with \'quoted\' text." valueless_attr weirdness="<i>foo</i>" />\n                    </p>\n                </div>\n                '))

    def test_raw_comment_one_line(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders('<!-- *foo* -->', '<!-- *foo* -->')

    def test_raw_comment_one_line_with_tag(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders('<!-- <tag> -->', '<!-- <tag> -->')

    def test_comment_in_code_span(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders('`<!-- *foo* -->`', '<p><code>&lt;!-- *foo* --&gt;</code></p>')

    def test_raw_comment_one_line_followed_by_text(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders('<!-- *foo* -->*bar*', self.dedent('\n                <!-- *foo* -->\n                <p><em>bar</em></p>\n                '))

    def test_raw_comment_one_line_followed_by_html(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders('<!-- *foo* --><p>*bar*</p>', self.dedent('\n                <!-- *foo* -->\n                <p>*bar*</p>\n                '))

    def test_raw_comment_trailing_whitespace(self):
        if False:
            return 10
        self.assertMarkdownRenders('<!-- *foo* --> ', '<!-- *foo* -->')

    def test_bogus_comment(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders('<!*foo*>', '<!--*foo*-->')

    def test_raw_multiline_comment(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                <!--\n                *foo*\n                -->\n                '), self.dedent('\n                <!--\n                *foo*\n                -->\n                '))

    def test_raw_multiline_comment_with_tag(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders(self.dedent('\n                <!--\n                <tag>\n                -->\n                '), self.dedent('\n                <!--\n                <tag>\n                -->\n                '))

    def test_raw_multiline_comment_first_line(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders(self.dedent('\n                <!-- *foo*\n                -->\n                '), self.dedent('\n                <!-- *foo*\n                -->\n                '))

    def test_raw_multiline_comment_last_line(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                <!--\n                *foo* -->\n                '), self.dedent('\n                <!--\n                *foo* -->\n                '))

    def test_raw_comment_with_blank_lines(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                <!--\n\n                *foo*\n\n                -->\n                '), self.dedent('\n                <!--\n\n                *foo*\n\n                -->\n                '))

    def test_raw_comment_with_blank_lines_with_tag(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                <!--\n\n                <tag>\n\n                -->\n                '), self.dedent('\n                <!--\n\n                <tag>\n\n                -->\n                '))

    def test_raw_comment_with_blank_lines_first_line(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n                <!-- *foo*\n\n                -->\n                '), self.dedent('\n                <!-- *foo*\n\n                -->\n                '))

    def test_raw_comment_with_blank_lines_last_line(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                <!--\n\n                *foo* -->\n                '), self.dedent('\n                <!--\n\n                *foo* -->\n                '))

    def test_raw_comment_indented(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                <!--\n\n                    *foo*\n\n                -->\n                '), self.dedent('\n                <!--\n\n                    *foo*\n\n                -->\n                '))

    def test_raw_comment_indented_with_tag(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                <!--\n\n                    <tag>\n\n                -->\n                '), self.dedent('\n                <!--\n\n                    <tag>\n\n                -->\n                '))

    def test_raw_comment_nested(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n                <div>\n                <!-- *foo* -->\n                </div>\n                '), self.dedent('\n                <div>\n                <!-- *foo* -->\n                </div>\n                '))

    def test_comment_in_code_block(self):
        if False:
            return 10
        self.assertMarkdownRenders('    <!-- *foo* -->', self.dedent('\n                <pre><code>&lt;!-- *foo* --&gt;\n                </code></pre>\n                '))

    def test_unclosed_comment_(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                <!-- unclosed comment\n\n                *not* a comment\n                '), self.dedent('\n                <p>&lt;!-- unclosed comment</p>\n                <p><em>not</em> a comment</p>\n                '))

    def test_raw_processing_instruction_one_line(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders("<?php echo '>'; ?>", "<?php echo '>'; ?>")

    def test_raw_processing_instruction_one_line_followed_by_text(self):
        if False:
            return 10
        self.assertMarkdownRenders("<?php echo '>'; ?>*bar*", self.dedent("\n                <?php echo '>'; ?>\n                <p><em>bar</em></p>\n                "))

    def test_raw_multiline_processing_instruction(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent("\n                <?php\n                echo '>';\n                ?>\n                "), self.dedent("\n                <?php\n                echo '>';\n                ?>\n                "))

    def test_raw_processing_instruction_with_blank_lines(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders(self.dedent("\n                <?php\n\n                echo '>';\n\n                ?>\n                "), self.dedent("\n                <?php\n\n                echo '>';\n\n                ?>\n                "))

    def test_raw_processing_instruction_indented(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders(self.dedent("\n                <?php\n\n                    echo '>';\n\n                ?>\n                "), self.dedent("\n                <?php\n\n                    echo '>';\n\n                ?>\n                "))

    def test_raw_processing_instruction_code_span(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders(self.dedent('\n                `<?php`\n\n                <div>\n                foo\n                </div>\n                '), self.dedent('\n                <p><code>&lt;?php</code></p>\n                <div>\n                foo\n                </div>\n                '))

    def test_raw_declaration_one_line(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders('<!DOCTYPE html>', '<!DOCTYPE html>')

    def test_raw_declaration_one_line_followed_by_text(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders('<!DOCTYPE html>*bar*', self.dedent('\n                <!DOCTYPE html>\n                <p><em>bar</em></p>\n                '))

    def test_raw_multiline_declaration(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n                <!DOCTYPE html PUBLIC\n                  "-//W3C//DTD XHTML 1.1//EN"\n                  "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">\n                '), self.dedent('\n                <!DOCTYPE html PUBLIC\n                  "-//W3C//DTD XHTML 1.1//EN"\n                  "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">\n                '))

    def test_raw_declaration_code_span(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                `<!`\n\n                <div>\n                foo\n                </div>\n                '), self.dedent('\n                <p><code>&lt;!</code></p>\n                <div>\n                foo\n                </div>\n                '))

    def test_raw_cdata_one_line(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders('<![CDATA[ document.write(">"); ]]>', '<![CDATA[ document.write(">"); ]]>')

    def test_raw_cdata_one_line_followed_by_text(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders('<![CDATA[ document.write(">"); ]]>*bar*', self.dedent('\n                <![CDATA[ document.write(">"); ]]>\n                <p><em>bar</em></p>\n                '))

    def test_raw_multiline_cdata(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                <![CDATA[\n                document.write(">");\n                ]]>\n                '), self.dedent('\n                <![CDATA[\n                document.write(">");\n                ]]>\n                '))

    def test_raw_cdata_with_blank_lines(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders(self.dedent('\n                <![CDATA[\n\n                document.write(">");\n\n                ]]>\n                '), self.dedent('\n                <![CDATA[\n\n                document.write(">");\n\n                ]]>\n                '))

    def test_raw_cdata_indented(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders(self.dedent('\n                <![CDATA[\n\n                    document.write(">");\n\n                ]]>\n                '), self.dedent('\n                <![CDATA[\n\n                    document.write(">");\n\n                ]]>\n                '))

    def test_raw_cdata_code_span(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                `<![`\n\n                <div>\n                foo\n                </div>\n                '), self.dedent('\n                <p><code>&lt;![</code></p>\n                <div>\n                foo\n                </div>\n                '))

    def test_charref(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders('&sect;', '<p>&sect;</p>')

    def test_nested_charref(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders('<p>&sect;</p>', '<p>&sect;</p>')

    def test_entityref(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders('&#167;', '<p>&#167;</p>')

    def test_nested_entityref(self):
        if False:
            return 10
        self.assertMarkdownRenders('<p>&#167;</p>', '<p>&#167;</p>')

    def test_amperstand(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders('AT&T & AT&amp;T', '<p>AT&amp;T &amp; AT&amp;T</p>')

    def test_startendtag(self):
        if False:
            return 10
        self.assertMarkdownRenders('<hr>', '<hr>')

    def test_startendtag_with_attrs(self):
        if False:
            return 10
        self.assertMarkdownRenders('<hr id="foo" class="bar">', '<hr id="foo" class="bar">')

    def test_startendtag_with_space(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders('<hr >', '<hr >')

    def test_closed_startendtag(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders('<hr />', '<hr />')

    def test_closed_startendtag_without_space(self):
        if False:
            return 10
        self.assertMarkdownRenders('<hr/>', '<hr/>')

    def test_closed_startendtag_with_attrs(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders('<hr id="foo" class="bar" />', '<hr id="foo" class="bar" />')

    def test_nested_startendtag(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders('<div><hr></div>', '<div><hr></div>')

    def test_nested_closed_startendtag(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders('<div><hr /></div>', '<div><hr /></div>')

    def test_auto_links_dont_break_parser(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                <https://example.com>\n\n                <email@example.com>\n                '), '<p><a href="https://example.com">https://example.com</a></p>\n<p><a href="&#109;&#97;&#105;&#108;&#116;&#111;&#58;&#101;&#109;&#97;&#105;&#108;&#64;&#101;&#120;&#97;&#109;&#112;&#108;&#101;&#46;&#99;&#111;&#109;">&#101;&#109;&#97;&#105;&#108;&#64;&#101;&#120;&#97;&#109;&#112;&#108;&#101;&#46;&#99;&#111;&#109;</a></p>')

    def test_text_links_ignored(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                https://example.com\n\n                email@example.com\n                '), self.dedent('\n                <p>https://example.com</p>\n                <p>email@example.com</p>\n                '))

    def text_invalid_tags(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n                <some [weird](http://example.com) stuff>\n\n                <some>> <<unbalanced>> <<brackets>\n                '), self.dedent('\n                <p><some <a href="http://example.com">weird</a> stuff></p>\n                <p><some>&gt; &lt;<unbalanced>&gt; &lt;<brackets></p>\n                '))

    def test_script_tags(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n                <script>\n                *random stuff* <div> &amp;\n                </script>\n\n                <style>\n                **more stuff**\n                </style>\n                '), self.dedent('\n                <script>\n                *random stuff* <div> &amp;\n                </script>\n\n                <style>\n                **more stuff**\n                </style>\n                '))

    def test_unclosed_script_tag(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                <script>\n                *random stuff* <div> &amp;\n\n                Still part of the *script* tag\n                '), self.dedent('\n                <script>\n                *random stuff* <div> &amp;\n\n                Still part of the *script* tag\n                '))

    def test_inline_script_tags(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders(self.dedent('\n                Text `<script>` more *text*.\n\n                <div>\n                *foo*\n                </div>\n\n                <div>\n\n                bar\n\n                </div>\n\n                A new paragraph with a closing `</script>` tag.\n                '), self.dedent('\n                <p>Text <code>&lt;script&gt;</code> more <em>text</em>.</p>\n                <div>\n                *foo*\n                </div>\n\n                <div>\n\n                bar\n\n                </div>\n\n                <p>A new paragraph with a closing <code>&lt;/script&gt;</code> tag.</p>\n                '))

    def test_hr_only_start(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders(self.dedent('\n                *emphasis1*\n                <hr>\n                *emphasis2*\n                '), self.dedent('\n                <p><em>emphasis1</em></p>\n                <hr>\n                <p><em>emphasis2</em></p>\n                '))

    def test_hr_self_close(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n                *emphasis1*\n                <hr/>\n                *emphasis2*\n                '), self.dedent('\n                <p><em>emphasis1</em></p>\n                <hr/>\n                <p><em>emphasis2</em></p>\n                '))

    def test_hr_start_and_end(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders(self.dedent('\n                *emphasis1*\n                <hr></hr>\n                *emphasis2*\n                '), self.dedent('\n                <p><em>emphasis1</em></p>\n                <hr>\n                <p></hr>\n                <em>emphasis2</em></p>\n                '))

    def test_hr_only_end(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n                *emphasis1*\n                </hr>\n                *emphasis2*\n                '), self.dedent('\n                <p><em>emphasis1</em>\n                </hr>\n                <em>emphasis2</em></p>\n                '))

    def test_hr_with_content(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n                *emphasis1*\n                <hr>\n                **content**\n                </hr>\n                *emphasis2*\n                '), self.dedent('\n                <p><em>emphasis1</em></p>\n                <hr>\n                <p><strong>content</strong>\n                </hr>\n                <em>emphasis2</em></p>\n                '))

    def test_placeholder_in_source(self):
        if False:
            i = 10
            return i + 15
        md = markdown.Markdown()
        md.htmlStash.store('foo')
        placeholder = md.htmlStash.get_placeholder(md.htmlStash.html_counter + 1)
        result = md.postprocessors['raw_html'].run(placeholder)
        self.assertEqual(placeholder, result)