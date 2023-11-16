"""Tests for HTMLParser.py."""
import html.parser
import pprint
import unittest

class EventCollector(html.parser.HTMLParser):

    def __init__(self, *args, **kw):
        if False:
            for i in range(10):
                print('nop')
        self.events = []
        self.append = self.events.append
        html.parser.HTMLParser.__init__(self, *args, **kw)

    def get_events(self):
        if False:
            while True:
                i = 10
        L = []
        prevtype = None
        for event in self.events:
            type = event[0]
            if type == prevtype == 'data':
                L[-1] = ('data', L[-1][1] + event[1])
            else:
                L.append(event)
            prevtype = type
        self.events = L
        return L

    def handle_starttag(self, tag, attrs):
        if False:
            while True:
                i = 10
        self.append(('starttag', tag, attrs))

    def handle_startendtag(self, tag, attrs):
        if False:
            return 10
        self.append(('startendtag', tag, attrs))

    def handle_endtag(self, tag):
        if False:
            i = 10
            return i + 15
        self.append(('endtag', tag))

    def handle_comment(self, data):
        if False:
            for i in range(10):
                print('nop')
        self.append(('comment', data))

    def handle_charref(self, data):
        if False:
            while True:
                i = 10
        self.append(('charref', data))

    def handle_data(self, data):
        if False:
            return 10
        self.append(('data', data))

    def handle_decl(self, data):
        if False:
            while True:
                i = 10
        self.append(('decl', data))

    def handle_entityref(self, data):
        if False:
            for i in range(10):
                print('nop')
        self.append(('entityref', data))

    def handle_pi(self, data):
        if False:
            return 10
        self.append(('pi', data))

    def unknown_decl(self, decl):
        if False:
            return 10
        self.append(('unknown decl', decl))

class EventCollectorExtra(EventCollector):

    def handle_starttag(self, tag, attrs):
        if False:
            print('Hello World!')
        EventCollector.handle_starttag(self, tag, attrs)
        self.append(('starttag_text', self.get_starttag_text()))

class EventCollectorCharrefs(EventCollector):

    def handle_charref(self, data):
        if False:
            while True:
                i = 10
        self.fail('This should never be called with convert_charrefs=True')

    def handle_entityref(self, data):
        if False:
            i = 10
            return i + 15
        self.fail('This should never be called with convert_charrefs=True')

class TestCaseBase(unittest.TestCase):

    def get_collector(self):
        if False:
            i = 10
            return i + 15
        return EventCollector(convert_charrefs=False)

    def _run_check(self, source, expected_events, collector=None):
        if False:
            for i in range(10):
                print('nop')
        if collector is None:
            collector = self.get_collector()
        parser = collector
        for s in source:
            parser.feed(s)
        parser.close()
        events = parser.get_events()
        if events != expected_events:
            self.fail('received events did not match expected events' + '\nSource:\n' + repr(source) + '\nExpected:\n' + pprint.pformat(expected_events) + '\nReceived:\n' + pprint.pformat(events))

    def _run_check_extra(self, source, events):
        if False:
            print('Hello World!')
        self._run_check(source, events, EventCollectorExtra(convert_charrefs=False))

class HTMLParserTestCase(TestCaseBase):

    def test_processing_instruction_only(self):
        if False:
            i = 10
            return i + 15
        self._run_check('<?processing instruction>', [('pi', 'processing instruction')])
        self._run_check('<?processing instruction ?>', [('pi', 'processing instruction ?')])

    def test_simple_html(self):
        if False:
            for i in range(10):
                print('nop')
        self._run_check("\n<!DOCTYPE html PUBLIC 'foo'>\n<HTML>&entity;&#32;\n<!--comment1a\n-></foo><bar>&lt;<?pi?></foo<bar\ncomment1b-->\n<Img sRc='Bar' isMAP>sample\ntext\n&#x201C;\n<!--comment2a-- --comment2b-->\n</Html>\n", [('data', '\n'), ('decl', "DOCTYPE html PUBLIC 'foo'"), ('data', '\n'), ('starttag', 'html', []), ('entityref', 'entity'), ('charref', '32'), ('data', '\n'), ('comment', 'comment1a\n-></foo><bar>&lt;<?pi?></foo<bar\ncomment1b'), ('data', '\n'), ('starttag', 'img', [('src', 'Bar'), ('ismap', None)]), ('data', 'sample\ntext\n'), ('charref', 'x201C'), ('data', '\n'), ('comment', 'comment2a-- --comment2b'), ('data', '\n'), ('endtag', 'html'), ('data', '\n')])

    def test_malformatted_charref(self):
        if False:
            while True:
                i = 10
        self._run_check('<p>&#bad;</p>', [('starttag', 'p', []), ('data', '&#bad;'), ('endtag', 'p')])
        self._run_check(['<div>&#bad;</div>'], [('starttag', 'div', []), ('data', '&#bad;'), ('endtag', 'div')])

    def test_unclosed_entityref(self):
        if False:
            while True:
                i = 10
        self._run_check('&entityref foo', [('entityref', 'entityref'), ('data', ' foo')])

    def test_bad_nesting(self):
        if False:
            i = 10
            return i + 15
        self._run_check('<a><b></a></b>', [('starttag', 'a', []), ('starttag', 'b', []), ('endtag', 'a'), ('endtag', 'b')])

    def test_bare_ampersands(self):
        if False:
            while True:
                i = 10
        self._run_check('this text & contains & ampersands &', [('data', 'this text & contains & ampersands &')])

    def test_bare_pointy_brackets(self):
        if False:
            while True:
                i = 10
        self._run_check('this < text > contains < bare>pointy< brackets', [('data', 'this < text > contains < bare>pointy< brackets')])

    def test_starttag_end_boundary(self):
        if False:
            while True:
                i = 10
        self._run_check("<a b='<'>", [('starttag', 'a', [('b', '<')])])
        self._run_check("<a b='>'>", [('starttag', 'a', [('b', '>')])])

    def test_buffer_artefacts(self):
        if False:
            i = 10
            return i + 15
        output = [('starttag', 'a', [('b', '<')])]
        self._run_check(["<a b='<'>"], output)
        self._run_check(['<a ', "b='<'>"], output)
        self._run_check(['<a b', "='<'>"], output)
        self._run_check(['<a b=', "'<'>"], output)
        self._run_check(["<a b='<", "'>"], output)
        self._run_check(["<a b='<'", '>'], output)
        output = [('starttag', 'a', [('b', '>')])]
        self._run_check(["<a b='>'>"], output)
        self._run_check(['<a ', "b='>'>"], output)
        self._run_check(['<a b', "='>'>"], output)
        self._run_check(['<a b=', "'>'>"], output)
        self._run_check(["<a b='>", "'>"], output)
        self._run_check(["<a b='>'", '>'], output)
        output = [('comment', 'abc')]
        self._run_check(['', '<!--abc-->'], output)
        self._run_check(['<', '!--abc-->'], output)
        self._run_check(['<!', '--abc-->'], output)
        self._run_check(['<!-', '-abc-->'], output)
        self._run_check(['<!--', 'abc-->'], output)
        self._run_check(['<!--a', 'bc-->'], output)
        self._run_check(['<!--ab', 'c-->'], output)
        self._run_check(['<!--abc', '-->'], output)
        self._run_check(['<!--abc-', '->'], output)
        self._run_check(['<!--abc--', '>'], output)
        self._run_check(['<!--abc-->', ''], output)

    def test_valid_doctypes(self):
        if False:
            i = 10
            return i + 15
        dtds = ['HTML', 'HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd"', 'HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd"', 'html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd"', 'html PUBLIC "-//W3C//DTD XHTML 1.0 Frameset//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-frameset.dtd"', 'math PUBLIC "-//W3C//DTD MathML 2.0//EN" "http://www.w3.org/Math/DTD/mathml2/mathml2.dtd"', 'html PUBLIC "-//W3C//DTD XHTML 1.1 plus MathML 2.0 plus SVG 1.1//EN" "http://www.w3.org/2002/04/xhtml-math-svg/xhtml-math-svg.dtd"', 'svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd"', 'html PUBLIC "-//IETF//DTD HTML 2.0//EN"', 'html PUBLIC "-//W3C//DTD HTML 3.2 Final//EN"']
        for dtd in dtds:
            self._run_check('<!DOCTYPE %s>' % dtd, [('decl', 'DOCTYPE ' + dtd)])

    def test_startendtag(self):
        if False:
            return 10
        self._run_check('<p/>', [('startendtag', 'p', [])])
        self._run_check('<p></p>', [('starttag', 'p', []), ('endtag', 'p')])
        self._run_check("<p><img src='foo' /></p>", [('starttag', 'p', []), ('startendtag', 'img', [('src', 'foo')]), ('endtag', 'p')])

    def test_get_starttag_text(self):
        if False:
            i = 10
            return i + 15
        s = '<foo:bar   \n   one="1"\ttwo=2   >'
        self._run_check_extra(s, [('starttag', 'foo:bar', [('one', '1'), ('two', '2')]), ('starttag_text', s)])

    def test_cdata_content(self):
        if False:
            i = 10
            return i + 15
        contents = ['<!-- not a comment --> &not-an-entity-ref;', "<not a='start tag'>", '<a href="" /> <p> <span></span>', 'foo = "</scr" + "ipt>";', 'foo = "</SCRIPT" + ">";', 'foo = <\n/script> ', '<!-- document.write("</scr" + "ipt>"); -->', '\n//<![CDATA[\ndocument.write(\'<s\'+\'cript type="text/javascript" src="http://www.example.org/r=\'+new Date().getTime()+\'"><\\/s\'+\'cript>\');\n//]]>', '\n<!-- //\nvar foo = 3.14;\n// -->\n', 'foo = "</sty" + "le>";', '<!-- ☃ -->']
        elements = ['script', 'style', 'SCRIPT', 'STYLE', 'Script', 'Style']
        for content in contents:
            for element in elements:
                element_lower = element.lower()
                s = '<{element}>{content}</{element}>'.format(element=element, content=content)
                self._run_check(s, [('starttag', element_lower, []), ('data', content), ('endtag', element_lower)])

    def test_cdata_with_closing_tags(self):
        if False:
            for i in range(10):
                print('nop')

        class Collector(EventCollector):

            def get_events(self):
                if False:
                    print('Hello World!')
                return self.events
        content = '<!-- not a comment --> &not-an-entity-ref;\n                  <a href="" /> </p><p> <span></span></style>\n                  \'</script\' + \'>\''
        for element in [' script', 'script ', ' script ', '\nscript', 'script\n', '\nscript\n']:
            element_lower = element.lower().strip()
            s = '<script>{content}</{element}>'.format(element=element, content=content)
            self._run_check(s, [('starttag', element_lower, []), ('data', content), ('endtag', element_lower)], collector=Collector(convert_charrefs=False))

    def test_comments(self):
        if False:
            print('Hello World!')
        html = "<!-- I'm a valid comment --><!--me too!--><!------><!----><!----I have many hyphens----><!-- I have a > in the middle --><!-- and I have -- in the middle! -->"
        expected = [('comment', " I'm a valid comment "), ('comment', 'me too!'), ('comment', '--'), ('comment', ''), ('comment', '--I have many hyphens--'), ('comment', ' I have a > in the middle '), ('comment', ' and I have -- in the middle! ')]
        self._run_check(html, expected)

    def test_condcoms(self):
        if False:
            for i in range(10):
                print('nop')
        html = "<!--[if IE & !(lte IE 8)]>aren't<![endif]--><!--[if IE 8]>condcoms<![endif]--><!--[if lte IE 7]>pretty?<![endif]-->"
        expected = [('comment', "[if IE & !(lte IE 8)]>aren't<![endif]"), ('comment', '[if IE 8]>condcoms<![endif]'), ('comment', '[if lte IE 7]>pretty?<![endif]')]
        self._run_check(html, expected)

    def test_convert_charrefs(self):
        if False:
            while True:
                i = 10
        collector = lambda : EventCollectorCharrefs()
        self.assertTrue(collector().convert_charrefs)
        charrefs = ['&quot;', '&#34;', '&#x22;', '&quot', '&#34', '&#x22']
        expected = [('starttag', 'a', [('href', 'foo"zar')]), ('data', 'a"z'), ('endtag', 'a')]
        for charref in charrefs:
            self._run_check('<a href="foo{0}zar">a{0}z</a>'.format(charref), expected, collector=collector())
        expected = [('data', '"'), ('starttag', 'a', [('x', '"'), ('y', '"X'), ('z', 'X"')]), ('data', '"'), ('endtag', 'a'), ('data', '"')]
        for charref in charrefs:
            self._run_check('{0}<a x="{0}" y="{0}X" z="X{0}">{0}</a>{0}'.format(charref), expected, collector=collector())
        for charref in charrefs:
            text = 'X'.join([charref] * 3)
            expected = [('data', '"'), ('starttag', 'script', []), ('data', text), ('endtag', 'script'), ('data', '"'), ('starttag', 'style', []), ('data', text), ('endtag', 'style'), ('data', '"')]
            self._run_check('{1}<script>{0}</script>{1}<style>{0}</style>{1}'.format(text, charref), expected, collector=collector())
        html = '&quo &# &#x'
        for x in range(1, len(html)):
            self._run_check(html[:x], [('data', html[:x])], collector=collector())
        self._run_check('no charrefs here', [('data', 'no charrefs here')], collector=collector())

    def test_tolerant_parsing(self):
        if False:
            print('Hello World!')
        self._run_check('<html <html>te>>xt&a<<bc</a></html>\n<img src="URL><//img></html</html>', [('starttag', 'html', [('<html', None)]), ('data', 'te>>xt'), ('entityref', 'a'), ('data', '<'), ('starttag', 'bc<', [('a', None)]), ('endtag', 'html'), ('data', '\n<img src="URL>'), ('comment', '/img'), ('endtag', 'html<')])

    def test_starttag_junk_chars(self):
        if False:
            i = 10
            return i + 15
        self._run_check('</>', [])
        self._run_check('</$>', [('comment', '$')])
        self._run_check('</', [('data', '</')])
        self._run_check('</a', [('data', '</a')])
        self._run_check('<a<a>', [('starttag', 'a<a', [])])
        self._run_check('</a<a>', [('endtag', 'a<a')])
        self._run_check('<!', [('data', '<!')])
        self._run_check('<a', [('data', '<a')])
        self._run_check("<a foo='bar'", [('data', "<a foo='bar'")])
        self._run_check("<a foo='bar", [('data', "<a foo='bar")])
        self._run_check("<a foo='>'", [('data', "<a foo='>'")])
        self._run_check("<a foo='>", [('data', "<a foo='>")])
        self._run_check('<a$>', [('starttag', 'a$', [])])
        self._run_check('<a$b>', [('starttag', 'a$b', [])])
        self._run_check('<a$b/>', [('startendtag', 'a$b', [])])
        self._run_check('<a$b  >', [('starttag', 'a$b', [])])
        self._run_check('<a$b  />', [('startendtag', 'a$b', [])])

    def test_slashes_in_starttag(self):
        if False:
            for i in range(10):
                print('nop')
        self._run_check('<a foo="var"/>', [('startendtag', 'a', [('foo', 'var')])])
        html = '<img width=902 height=250px src="/sites/default/files/images/homepage/foo.jpg" /*what am I doing here*/ />'
        expected = [('startendtag', 'img', [('width', '902'), ('height', '250px'), ('src', '/sites/default/files/images/homepage/foo.jpg'), ('*what', None), ('am', None), ('i', None), ('doing', None), ('here*', None)])]
        self._run_check(html, expected)
        html = '<a / /foo/ / /=/ / /bar/ / /><a / /foo/ / /=/ / /bar/ / >'
        expected = [('startendtag', 'a', [('foo', None), ('=', None), ('bar', None)]), ('starttag', 'a', [('foo', None), ('=', None), ('bar', None)])]
        self._run_check(html, expected)
        html = '<meta><meta / ><meta // ><meta / / ><meta/><meta /><meta //><meta//>'
        expected = [('starttag', 'meta', []), ('starttag', 'meta', []), ('starttag', 'meta', []), ('starttag', 'meta', []), ('startendtag', 'meta', []), ('startendtag', 'meta', []), ('startendtag', 'meta', []), ('startendtag', 'meta', [])]
        self._run_check(html, expected)

    def test_declaration_junk_chars(self):
        if False:
            i = 10
            return i + 15
        self._run_check('<!DOCTYPE foo $ >', [('decl', 'DOCTYPE foo $ ')])

    def test_illegal_declarations(self):
        if False:
            while True:
                i = 10
        self._run_check('<!spacer type="block" height="25">', [('comment', 'spacer type="block" height="25"')])

    def test_invalid_end_tags(self):
        if False:
            i = 10
            return i + 15
        html = '<br></label</p><br></div end tmAd-leaderBoard><br></<h4><br></li class="unit"><br></li\r\n\t\t\t\t\t\t</ul><br></><br>'
        expected = [('starttag', 'br', []), ('endtag', 'label<'), ('starttag', 'br', []), ('endtag', 'div'), ('starttag', 'br', []), ('comment', '<h4'), ('starttag', 'br', []), ('endtag', 'li'), ('starttag', 'br', []), ('endtag', 'li'), ('starttag', 'br', []), ('starttag', 'br', [])]
        self._run_check(html, expected)

    def test_broken_invalid_end_tag(self):
        if False:
            return 10
        html = '<b>This</b attr=">"> confuses the parser'
        expected = [('starttag', 'b', []), ('data', 'This'), ('endtag', 'b'), ('data', '"> confuses the parser')]
        self._run_check(html, expected)

    def test_correct_detection_of_start_tags(self):
        if False:
            i = 10
            return i + 15
        html = '<div style=""    ><b>The <a href="some_url">rain</a> <br /> in <span>Spain</span></b></div>'
        expected = [('starttag', 'div', [('style', '')]), ('starttag', 'b', []), ('data', 'The '), ('starttag', 'a', [('href', 'some_url')]), ('data', 'rain'), ('endtag', 'a'), ('data', ' '), ('startendtag', 'br', []), ('data', ' in '), ('starttag', 'span', []), ('data', 'Spain'), ('endtag', 'span'), ('endtag', 'b'), ('endtag', 'div')]
        self._run_check(html, expected)
        html = '<div style="", foo = "bar" ><b>The <a href="some_url">rain</a>'
        expected = [('starttag', 'div', [('style', ''), (',', None), ('foo', 'bar')]), ('starttag', 'b', []), ('data', 'The '), ('starttag', 'a', [('href', 'some_url')]), ('data', 'rain'), ('endtag', 'a')]
        self._run_check(html, expected)

    def test_EOF_in_charref(self):
        if False:
            while True:
                i = 10
        data = [('a&', [('data', 'a&')]), ('a&b', [('data', 'ab')]), ('a&b ', [('data', 'a'), ('entityref', 'b'), ('data', ' ')]), ('a&b;', [('data', 'a'), ('entityref', 'b')])]
        for (html, expected) in data:
            self._run_check(html, expected)

    def test_broken_comments(self):
        if False:
            for i in range(10):
                print('nop')
        html = '<! not really a comment ><! not a comment either --><! -- close enough --><!><!<-- this was an empty comment><!!! another bogus comment !!!>'
        expected = [('comment', ' not really a comment '), ('comment', ' not a comment either --'), ('comment', ' -- close enough --'), ('comment', ''), ('comment', '<-- this was an empty comment'), ('comment', '!! another bogus comment !!!')]
        self._run_check(html, expected)

    def test_broken_condcoms(self):
        if False:
            return 10
        html = '<![if !(IE)]>broken condcom<![endif]><![if ! IE]><link href="favicon.tiff"/><![endif]><![if !IE 6]><img src="firefox.png" /><![endif]><![if !ie 6]><b>foo</b><![endif]><![if (!IE)|(lt IE 9)]><img src="mammoth.bmp" /><![endif]>'
        expected = [('unknown decl', 'if !(IE)'), ('data', 'broken condcom'), ('unknown decl', 'endif'), ('unknown decl', 'if ! IE'), ('startendtag', 'link', [('href', 'favicon.tiff')]), ('unknown decl', 'endif'), ('unknown decl', 'if !IE 6'), ('startendtag', 'img', [('src', 'firefox.png')]), ('unknown decl', 'endif'), ('unknown decl', 'if !ie 6'), ('starttag', 'b', []), ('data', 'foo'), ('endtag', 'b'), ('unknown decl', 'endif'), ('unknown decl', 'if (!IE)|(lt IE 9)'), ('startendtag', 'img', [('src', 'mammoth.bmp')]), ('unknown decl', 'endif')]
        self._run_check(html, expected)

    def test_convert_charrefs_dropped_text(self):
        if False:
            for i in range(10):
                print('nop')
        parser = EventCollector(convert_charrefs=True)
        parser.feed('foo <a>link</a> bar &amp; baz')
        self.assertEqual(parser.get_events(), [('data', 'foo '), ('starttag', 'a', []), ('data', 'link'), ('endtag', 'a'), ('data', ' bar & baz')])

class AttributesTestCase(TestCaseBase):

    def test_attr_syntax(self):
        if False:
            return 10
        output = [('starttag', 'a', [('b', 'v'), ('c', 'v'), ('d', 'v'), ('e', None)])]
        self._run_check('<a b=\'v\' c="v" d=v e>', output)
        self._run_check('<a  b = \'v\' c = "v" d = v e>', output)
        self._run_check('<a\nb\n=\n\'v\'\nc\n=\n"v"\nd\n=\nv\ne>', output)
        self._run_check('<a\tb\t=\t\'v\'\tc\t=\t"v"\td\t=\tv\te>', output)

    def test_attr_values(self):
        if False:
            for i in range(10):
                print('nop')
        self._run_check('<a b=\'xxx\n\txxx\' c="yyy\t\nyyy" d=\'\txyz\n\'>', [('starttag', 'a', [('b', 'xxx\n\txxx'), ('c', 'yyy\t\nyyy'), ('d', '\txyz\n')])])
        self._run_check('<a b=\'\' c="">', [('starttag', 'a', [('b', ''), ('c', '')])])
        self._run_check('<e a=rgb(1,2,3)>', [('starttag', 'e', [('a', 'rgb(1,2,3)')])])
        self._run_check('<a href=mailto:xyz@example.com>', [('starttag', 'a', [('href', 'mailto:xyz@example.com')])])

    def test_attr_nonascii(self):
        if False:
            while True:
                i = 10
        self._run_check('<img src=/foo/bar.png alt=中文>', [('starttag', 'img', [('src', '/foo/bar.png'), ('alt', '中文')])])
        self._run_check("<a title='テスト' href='テスト.html'>", [('starttag', 'a', [('title', 'テスト'), ('href', 'テスト.html')])])
        self._run_check('<a title="テスト" href="テスト.html">', [('starttag', 'a', [('title', 'テスト'), ('href', 'テスト.html')])])

    def test_attr_entity_replacement(self):
        if False:
            while True:
                i = 10
        self._run_check("<a b='&amp;&gt;&lt;&quot;&apos;'>", [('starttag', 'a', [('b', '&><"\'')])])

    def test_attr_funky_names(self):
        if False:
            for i in range(10):
                print('nop')
        self._run_check("<a a.b='v' c:d=v e-f=v>", [('starttag', 'a', [('a.b', 'v'), ('c:d', 'v'), ('e-f', 'v')])])

    def test_entityrefs_in_attributes(self):
        if False:
            for i in range(10):
                print('nop')
        self._run_check("<html foo='&euro;&amp;&#97;&#x61;&unsupported;'>", [('starttag', 'html', [('foo', '€&aa&unsupported;')])])

    def test_attr_funky_names2(self):
        if False:
            for i in range(10):
                print('nop')
        self._run_check('<a $><b $=%><c \\=/>', [('starttag', 'a', [('$', None)]), ('starttag', 'b', [('$', '%')]), ('starttag', 'c', [('\\', '/')])])

    def test_entities_in_attribute_value(self):
        if False:
            while True:
                i = 10
        for entity in ['&', '&amp;', '&#38;', '&#x26;']:
            self._run_check('<a href="%s">' % entity, [('starttag', 'a', [('href', '&')])])
            self._run_check("<a href='%s'>" % entity, [('starttag', 'a', [('href', '&')])])
            self._run_check('<a href=%s>' % entity, [('starttag', 'a', [('href', '&')])])

    def test_malformed_attributes(self):
        if False:
            i = 10
            return i + 15
        html = "<a href=test'style='color:red;bad1'>test - bad1</a><a href=test'+style='color:red;ba2'>test - bad2</a><a href=test'&nbsp;style='color:red;bad3'>test - bad3</a><a href = test'&nbsp;style='color:red;bad4'  >test - bad4</a>"
        expected = [('starttag', 'a', [('href', "test'style='color:red;bad1'")]), ('data', 'test - bad1'), ('endtag', 'a'), ('starttag', 'a', [('href', "test'+style='color:red;ba2'")]), ('data', 'test - bad2'), ('endtag', 'a'), ('starttag', 'a', [('href', "test'\xa0style='color:red;bad3'")]), ('data', 'test - bad3'), ('endtag', 'a'), ('starttag', 'a', [('href', "test'\xa0style='color:red;bad4'")]), ('data', 'test - bad4'), ('endtag', 'a')]
        self._run_check(html, expected)

    def test_malformed_adjacent_attributes(self):
        if False:
            i = 10
            return i + 15
        self._run_check('<x><y z=""o"" /></x>', [('starttag', 'x', []), ('startendtag', 'y', [('z', ''), ('o""', None)]), ('endtag', 'x')])
        self._run_check('<x><y z="""" /></x>', [('starttag', 'x', []), ('startendtag', 'y', [('z', ''), ('""', None)]), ('endtag', 'x')])

    def test_adjacent_attributes(self):
        if False:
            return 10
        self._run_check('<a width="100%"cellspacing=0>', [('starttag', 'a', [('width', '100%'), ('cellspacing', '0')])])
        self._run_check('<a id="foo"class="bar">', [('starttag', 'a', [('id', 'foo'), ('class', 'bar')])])

    def test_missing_attribute_value(self):
        if False:
            print('Hello World!')
        self._run_check('<a v=>', [('starttag', 'a', [('v', '')])])

    def test_javascript_attribute_value(self):
        if False:
            while True:
                i = 10
        self._run_check("<a href=javascript:popup('/popup/help.html')>", [('starttag', 'a', [('href', "javascript:popup('/popup/help.html')")])])

    def test_end_tag_in_attribute_value(self):
        if False:
            return 10
        self._run_check('<a href=\'http://www.example.org/">;\'>spam</a>', [('starttag', 'a', [('href', 'http://www.example.org/">;')]), ('data', 'spam'), ('endtag', 'a')])

    def test_with_unquoted_attributes(self):
        if False:
            return 10
        html = "<html><body bgcolor=d0ca90 text='181008'><table cellspacing=0 cellpadding=1 width=100% ><tr><td align=left><font size=-1>- <a href=/rabota/><span class=en> software-and-i</span></a>- <a href='/1/'><span class=en> library</span></a></table>"
        expected = [('starttag', 'html', []), ('starttag', 'body', [('bgcolor', 'd0ca90'), ('text', '181008')]), ('starttag', 'table', [('cellspacing', '0'), ('cellpadding', '1'), ('width', '100%')]), ('starttag', 'tr', []), ('starttag', 'td', [('align', 'left')]), ('starttag', 'font', [('size', '-1')]), ('data', '- '), ('starttag', 'a', [('href', '/rabota/')]), ('starttag', 'span', [('class', 'en')]), ('data', ' software-and-i'), ('endtag', 'span'), ('endtag', 'a'), ('data', '- '), ('starttag', 'a', [('href', '/1/')]), ('starttag', 'span', [('class', 'en')]), ('data', ' library'), ('endtag', 'span'), ('endtag', 'a'), ('endtag', 'table')]
        self._run_check(html, expected)

    def test_comma_between_attributes(self):
        if False:
            return 10
        html = '<div class=bar,baz=asd><div class="bar",baz="asd"><div class=bar, baz=asd,><div class="bar", baz="asd",><div class="bar",><div class=,bar baz=,asd><div class=,"bar" baz=,"asd"><div ,class=bar ,baz=asd><div class,="bar" baz,="asd">'
        expected = [('starttag', 'div', [('class', 'bar,baz=asd')]), ('starttag', 'div', [('class', 'bar'), (',baz', 'asd')]), ('starttag', 'div', [('class', 'bar,'), ('baz', 'asd,')]), ('starttag', 'div', [('class', 'bar'), (',', None), ('baz', 'asd'), (',', None)]), ('starttag', 'div', [('class', 'bar'), (',', None)]), ('starttag', 'div', [('class', ',bar'), ('baz', ',asd')]), ('starttag', 'div', [('class', ',"bar"'), ('baz', ',"asd"')]), ('starttag', 'div', [(',class', 'bar'), (',baz', 'asd')]), ('starttag', 'div', [('class,', 'bar'), ('baz,', 'asd')])]
        self._run_check(html, expected)

    def test_weird_chars_in_unquoted_attribute_values(self):
        if False:
            print('Hello World!')
        self._run_check('<form action=bogus|&#()value>', [('starttag', 'form', [('action', 'bogus|&#()value')])])
if __name__ == '__main__':
    unittest.main()