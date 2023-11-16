import re
from io import StringIO
re_control_chars_1 = re.compile('(_x[0-9a-fA-F]{4}_)')
re_control_chars_2 = re.compile('([\\x00-\\x08\\x0b-\\x1f])')
xml_escapes = re.compile('["&<>\n]')

class XMLwriter(object):
    """
    Simple XML writer class.

    """

    def __init__(self):
        if False:
            print('Hello World!')
        self.fh = None
        self.internal_fh = False

    def _set_filehandle(self, filehandle):
        if False:
            while True:
                i = 10
        self.fh = filehandle
        self.internal_fh = False

    def _set_xml_writer(self, filename):
        if False:
            return 10
        if isinstance(filename, StringIO):
            self.internal_fh = False
            self.fh = filename
        else:
            self.internal_fh = True
            self.fh = open(filename, 'w', encoding='utf-8')

    def _xml_close(self):
        if False:
            print('Hello World!')
        if self.internal_fh:
            self.fh.close()

    def _xml_declaration(self):
        if False:
            return 10
        self.fh.write('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n')

    def _xml_start_tag(self, tag, attributes=[]):
        if False:
            i = 10
            return i + 15
        for (key, value) in attributes:
            value = self._escape_attributes(value)
            tag += ' %s="%s"' % (key, value)
        self.fh.write('<%s>' % tag)

    def _xml_start_tag_unencoded(self, tag, attributes=[]):
        if False:
            return 10
        for (key, value) in attributes:
            tag += ' %s="%s"' % (key, value)
        self.fh.write('<%s>' % tag)

    def _xml_end_tag(self, tag):
        if False:
            return 10
        self.fh.write('</%s>' % tag)

    def _xml_empty_tag(self, tag, attributes=[]):
        if False:
            while True:
                i = 10
        for (key, value) in attributes:
            value = self._escape_attributes(value)
            tag += ' %s="%s"' % (key, value)
        self.fh.write('<%s/>' % tag)

    def _xml_empty_tag_unencoded(self, tag, attributes=[]):
        if False:
            while True:
                i = 10
        for (key, value) in attributes:
            tag += ' %s="%s"' % (key, value)
        self.fh.write('<%s/>' % tag)

    def _xml_data_element(self, tag, data, attributes=[]):
        if False:
            for i in range(10):
                print('nop')
        end_tag = tag
        for (key, value) in attributes:
            value = self._escape_attributes(value)
            tag += ' %s="%s"' % (key, value)
        data = self._escape_data(data)
        data = self._escape_control_characters(data)
        self.fh.write('<%s>%s</%s>' % (tag, data, end_tag))

    def _xml_string_element(self, index, attributes=[]):
        if False:
            print('Hello World!')
        attr = ''
        for (key, value) in attributes:
            value = self._escape_attributes(value)
            attr += ' %s="%s"' % (key, value)
        self.fh.write('<c%s t="s"><v>%d</v></c>' % (attr, index))

    def _xml_si_element(self, string, attributes=[]):
        if False:
            i = 10
            return i + 15
        attr = ''
        for (key, value) in attributes:
            value = self._escape_attributes(value)
            attr += ' %s="%s"' % (key, value)
        string = self._escape_data(string)
        self.fh.write('<si><t%s>%s</t></si>' % (attr, string))

    def _xml_rich_si_element(self, string):
        if False:
            for i in range(10):
                print('nop')
        self.fh.write('<si>%s</si>' % string)

    def _xml_number_element(self, number, attributes=[]):
        if False:
            for i in range(10):
                print('nop')
        attr = ''
        for (key, value) in attributes:
            value = self._escape_attributes(value)
            attr += ' %s="%s"' % (key, value)
        self.fh.write('<c%s><v>%.16G</v></c>' % (attr, number))

    def _xml_formula_element(self, formula, result, attributes=[]):
        if False:
            return 10
        attr = ''
        for (key, value) in attributes:
            value = self._escape_attributes(value)
            attr += ' %s="%s"' % (key, value)
        self.fh.write('<c%s><f>%s</f><v>%s</v></c>' % (attr, self._escape_data(formula), self._escape_data(result)))

    def _xml_inline_string(self, string, preserve, attributes=[]):
        if False:
            for i in range(10):
                print('nop')
        attr = ''
        t_attr = ''
        if preserve:
            t_attr = ' xml:space="preserve"'
        for (key, value) in attributes:
            value = self._escape_attributes(value)
            attr += ' %s="%s"' % (key, value)
        string = self._escape_data(string)
        self.fh.write('<c%s t="inlineStr"><is><t%s>%s</t></is></c>' % (attr, t_attr, string))

    def _xml_rich_inline_string(self, string, attributes=[]):
        if False:
            for i in range(10):
                print('nop')
        attr = ''
        for (key, value) in attributes:
            value = self._escape_attributes(value)
            attr += ' %s="%s"' % (key, value)
        self.fh.write('<c%s t="inlineStr"><is>%s</is></c>' % (attr, string))

    def _escape_attributes(self, attribute):
        if False:
            while True:
                i = 10
        try:
            if not xml_escapes.search(attribute):
                return attribute
        except TypeError:
            return attribute
        attribute = attribute.replace('&', '&amp;').replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '&#xA;')
        return attribute

    def _escape_data(self, data):
        if False:
            print('Hello World!')
        try:
            if not xml_escapes.search(data):
                return data
        except TypeError:
            return data
        data = data.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        return data

    @staticmethod
    def _escape_control_characters(data):
        if False:
            i = 10
            return i + 15
        try:
            data = re_control_chars_1.sub('_x005F\\1', data)
        except TypeError:
            return data
        data = re_control_chars_2.sub(lambda match: '_x%04X_' % ord(match.group(1)), data)
        data = data.replace('\ufffe', '_xFFFE_').replace('\uffff', '_xFFFF_')
        return data