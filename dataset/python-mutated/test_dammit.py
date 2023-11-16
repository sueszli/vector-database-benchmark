import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import EntitySubstitution, EncodingDetector, UnicodeDammit

class TestUnicodeDammit(object):
    """Standalone tests of UnicodeDammit."""

    def test_unicode_input(self):
        if False:
            print('Hello World!')
        markup = "I'm already Unicode! ☃"
        dammit = UnicodeDammit(markup)
        assert dammit.unicode_markup == markup

    @pytest.mark.parametrize('smart_quotes_to,expect_converted', [(None, '‘’“”'), ('xml', '&#x2018;&#x2019;&#x201C;&#x201D;'), ('html', '&lsquo;&rsquo;&ldquo;&rdquo;'), ('ascii', "''" + '""')])
    def test_smart_quotes_to(self, smart_quotes_to, expect_converted):
        if False:
            for i in range(10):
                print('nop')
        'Verify the functionality of the smart_quotes_to argument\n        to the UnicodeDammit constructor.'
        markup = b'<foo>\x91\x92\x93\x94</foo>'
        converted = UnicodeDammit(markup, known_definite_encodings=['windows-1252'], smart_quotes_to=smart_quotes_to).unicode_markup
        assert converted == '<foo>{}</foo>'.format(expect_converted)

    def test_detect_utf8(self):
        if False:
            return 10
        utf8 = b'Sacr\xc3\xa9 bleu! \xe2\x98\x83'
        dammit = UnicodeDammit(utf8)
        assert dammit.original_encoding.lower() == 'utf-8'
        assert dammit.unicode_markup == 'Sacré bleu! ☃'

    def test_convert_hebrew(self):
        if False:
            while True:
                i = 10
        hebrew = b'\xed\xe5\xec\xf9'
        dammit = UnicodeDammit(hebrew, ['iso-8859-8'])
        assert dammit.original_encoding.lower() == 'iso-8859-8'
        assert dammit.unicode_markup == 'םולש'

    def test_dont_see_smart_quotes_where_there_are_none(self):
        if False:
            while True:
                i = 10
        utf_8 = b'\xe3\x82\xb1\xe3\x83\xbc\xe3\x82\xbf\xe3\x82\xa4 Watch'
        dammit = UnicodeDammit(utf_8)
        assert dammit.original_encoding.lower() == 'utf-8'
        assert dammit.unicode_markup.encode('utf-8') == utf_8

    def test_ignore_inappropriate_codecs(self):
        if False:
            i = 10
            return i + 15
        utf8_data = 'Räksmörgås'.encode('utf-8')
        dammit = UnicodeDammit(utf8_data, ['iso-8859-8'])
        assert dammit.original_encoding.lower() == 'utf-8'

    def test_ignore_invalid_codecs(self):
        if False:
            for i in range(10):
                print('nop')
        utf8_data = 'Räksmörgås'.encode('utf-8')
        for bad_encoding in ['.utf8', '...', 'utF---16.!']:
            dammit = UnicodeDammit(utf8_data, [bad_encoding])
            assert dammit.original_encoding.lower() == 'utf-8'

    def test_exclude_encodings(self):
        if False:
            return 10
        utf8_data = 'Räksmörgås'.encode('utf-8')
        dammit = UnicodeDammit(utf8_data, exclude_encodings=['utf-8'])
        assert dammit.original_encoding.lower() == 'windows-1252'
        dammit = UnicodeDammit(utf8_data, exclude_encodings=['utf-8', 'windows-1252'])
        assert dammit.original_encoding == None

class TestEncodingDetector(object):

    def test_encoding_detector_replaces_junk_in_encoding_name_with_replacement_character(self):
        if False:
            while True:
                i = 10
        detected = EncodingDetector(b'<?xml version="1.0" encoding="UTF-\xdb" ?>')
        encodings = list(detected.encodings)
        assert 'utf-�' in encodings

    def test_detect_html5_style_meta_tag(self):
        if False:
            i = 10
            return i + 15
        for data in (b'<html><meta charset="euc-jp" /></html>', b"<html><meta charset='euc-jp' /></html>", b'<html><meta charset=euc-jp /></html>', b'<html><meta charset=euc-jp/></html>'):
            dammit = UnicodeDammit(data, is_html=True)
            assert 'euc-jp' == dammit.original_encoding

    def test_last_ditch_entity_replacement(self):
        if False:
            i = 10
            return i + 15
        doc = b'\xef\xbb\xbf<?xml version="1.0" encoding="UTF-8"?>\n<html><b>\xd8\xa8\xd8\xaa\xd8\xb1</b>\n<i>\xc8\xd2\xd1\x90\xca\xd1\xed\xe4</i></html>'
        chardet = bs4.dammit.chardet_dammit
        logging.disable(logging.WARNING)
        try:

            def noop(str):
                if False:
                    print('Hello World!')
                return None
            bs4.dammit.chardet_dammit = noop
            dammit = UnicodeDammit(doc)
            assert True == dammit.contains_replacement_characters
            assert '�' in dammit.unicode_markup
            soup = BeautifulSoup(doc, 'html.parser')
            assert soup.contains_replacement_characters
        finally:
            logging.disable(logging.NOTSET)
            bs4.dammit.chardet_dammit = chardet

    def test_byte_order_mark_removed(self):
        if False:
            for i in range(10):
                print('nop')
        data = b'\xff\xfe<\x00a\x00>\x00\xe1\x00\xe9\x00<\x00/\x00a\x00>\x00'
        dammit = UnicodeDammit(data)
        assert '<a>áé</a>' == dammit.unicode_markup
        assert 'utf-16le' == dammit.original_encoding

    def test_known_definite_versus_user_encodings(self):
        if False:
            while True:
                i = 10
        data = b'\xff\xfe<\x00a\x00>\x00\xe1\x00\xe9\x00<\x00/\x00a\x00>\x00'
        dammit = UnicodeDammit(data)
        before = UnicodeDammit(data, known_definite_encodings=['utf-16'])
        assert 'utf-16' == before.original_encoding
        after = UnicodeDammit(data, user_encodings=['utf-8'])
        assert 'utf-16le' == after.original_encoding
        assert ['utf-16le'] == [x[0] for x in dammit.tried_encodings]
        hebrew = b'\xed\xe5\xec\xf9'
        dammit = UnicodeDammit(hebrew, known_definite_encodings=['utf-8'], user_encodings=['iso-8859-8'])
        assert 'iso-8859-8' == dammit.original_encoding
        assert ['utf-8', 'iso-8859-8'] == [x[0] for x in dammit.tried_encodings]

    def test_deprecated_override_encodings(self):
        if False:
            i = 10
            return i + 15
        hebrew = b'\xed\xe5\xec\xf9'
        dammit = UnicodeDammit(hebrew, known_definite_encodings=['shift-jis'], override_encodings=['utf-8'], user_encodings=['iso-8859-8'])
        assert 'iso-8859-8' == dammit.original_encoding
        assert ['shift-jis', 'utf-8', 'iso-8859-8'] == [x[0] for x in dammit.tried_encodings]

    def test_detwingle(self):
        if False:
            return 10
        utf8 = ('☃' * 3).encode('utf8')
        windows_1252 = '“Hi, I like Windows!”'.encode('windows_1252')
        doc = utf8 + windows_1252 + utf8
        with pytest.raises(UnicodeDecodeError):
            doc.decode('utf8')
        fixed = UnicodeDammit.detwingle(doc)
        assert '☃☃☃“Hi, I like Windows!”☃☃☃' == fixed.decode('utf8')

    def test_detwingle_ignores_multibyte_characters(self):
        if False:
            while True:
                i = 10
        for tricky_unicode_char in ('œ', 'ₓ', 'ð\x90\x90\x93'):
            input = tricky_unicode_char.encode('utf8')
            assert input.endswith(b'\x93')
            output = UnicodeDammit.detwingle(input)
            assert output == input

    def test_find_declared_encoding(self):
        if False:
            for i in range(10):
                print('nop')
        html_unicode = '<html><head><meta charset="utf-8"></head></html>'
        html_bytes = html_unicode.encode('ascii')
        xml_unicode = '<?xml version="1.0" encoding="ISO-8859-1" ?>'
        xml_bytes = xml_unicode.encode('ascii')
        m = EncodingDetector.find_declared_encoding
        assert m(html_unicode, is_html=False) is None
        assert 'utf-8' == m(html_unicode, is_html=True)
        assert 'utf-8' == m(html_bytes, is_html=True)
        assert 'iso-8859-1' == m(xml_unicode)
        assert 'iso-8859-1' == m(xml_bytes)
        spacer = b' ' * 5000
        assert m(spacer + html_bytes) is None
        assert m(spacer + xml_bytes) is None
        assert m(spacer + html_bytes, is_html=True, search_entire_document=True) == 'utf-8'
        assert m(xml_bytes, search_entire_document=True) == 'iso-8859-1'
        assert m(b' ' + xml_bytes, search_entire_document=True) == 'iso-8859-1'
        assert m(b'a' + xml_bytes, search_entire_document=True) is None

class TestEntitySubstitution(object):
    """Standalone tests of the EntitySubstitution class."""

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        self.sub = EntitySubstitution

    @pytest.mark.parametrize('original,substituted', [('foo∀☃õbar', 'foo&forall;☃&otilde;bar'), ('‘’foo“”', '&lsquo;&rsquo;foo&ldquo;&rdquo;')])
    def test_substitute_html(self, original, substituted):
        if False:
            return 10
        assert self.sub.substitute_html(original) == substituted

    def test_html5_entity(self):
        if False:
            print('Hello World!')
        for (entity, u) in (('&models;', '⊧'), ('&Nfr;', '𝔑'), ('&ngeqq;', '≧̸'), ('&not;', '¬'), ('&Not;', '⫬'), '||', ('fj', 'fj'), ('&gt;', '>'), ('&lt;', '<'), ('&amp;', '&')):
            template = '3 %s 4'
            raw = template % u
            with_entities = template % entity
            assert self.sub.substitute_html(raw) == with_entities

    def test_html5_entity_with_variation_selector(self):
        if False:
            for i in range(10):
                print('nop')
        data = 'fjords ⊔ penguins'
        markup = 'fjords &sqcup; penguins'
        assert self.sub.substitute_html(data) == markup
        data = 'fjords ⊔︀ penguins'
        markup = 'fjords &sqcups; penguins'
        assert self.sub.substitute_html(data) == markup

    def test_xml_converstion_includes_no_quotes_if_make_quoted_attribute_is_false(self):
        if False:
            i = 10
            return i + 15
        s = 'Welcome to "my bar"'
        assert self.sub.substitute_xml(s, False) == s

    def test_xml_attribute_quoting_normally_uses_double_quotes(self):
        if False:
            return 10
        assert self.sub.substitute_xml('Welcome', True) == '"Welcome"'
        assert self.sub.substitute_xml("Bob's Bar", True) == '"Bob\'s Bar"'

    def test_xml_attribute_quoting_uses_single_quotes_when_value_contains_double_quotes(self):
        if False:
            while True:
                i = 10
        s = 'Welcome to "my bar"'
        assert self.sub.substitute_xml(s, True) == '\'Welcome to "my bar"\''

    def test_xml_attribute_quoting_escapes_single_quotes_when_value_contains_both_single_and_double_quotes(self):
        if False:
            i = 10
            return i + 15
        s = 'Welcome to "Bob\'s Bar"'
        assert self.sub.substitute_xml(s, True) == '"Welcome to &quot;Bob\'s Bar&quot;"'

    def test_xml_quotes_arent_escaped_when_value_is_not_being_quoted(self):
        if False:
            return 10
        quoted = 'Welcome to "Bob\'s Bar"'
        assert self.sub.substitute_xml(quoted) == quoted

    def test_xml_quoting_handles_angle_brackets(self):
        if False:
            i = 10
            return i + 15
        assert self.sub.substitute_xml('foo<bar>') == 'foo&lt;bar&gt;'

    def test_xml_quoting_handles_ampersands(self):
        if False:
            while True:
                i = 10
        assert self.sub.substitute_xml('AT&T') == 'AT&amp;T'

    def test_xml_quoting_including_ampersands_when_they_are_part_of_an_entity(self):
        if False:
            i = 10
            return i + 15
        assert self.sub.substitute_xml('&Aacute;T&T') == '&amp;Aacute;T&amp;T'

    def test_xml_quoting_ignoring_ampersands_when_they_are_part_of_an_entity(self):
        if False:
            return 10
        assert self.sub.substitute_xml_containing_entities('&Aacute;T&T') == '&Aacute;T&amp;T'

    def test_quotes_not_html_substituted(self):
        if False:
            while True:
                i = 10
        "There's no need to do this except inside attribute values."
        text = 'Bob\'s "bar"'
        assert self.sub.substitute_html(text) == text