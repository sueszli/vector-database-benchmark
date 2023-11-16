from . import xmlwriter
from .utility import preserve_whitespace

class SharedStrings(xmlwriter.XMLwriter):
    """
    A class for writing the Excel XLSX sharedStrings file.

    """

    def __init__(self):
        if False:
            while True:
                i = 10
        '\n        Constructor.\n\n        '
        super(SharedStrings, self).__init__()
        self.string_table = None

    def _assemble_xml_file(self):
        if False:
            while True:
                i = 10
        self._xml_declaration()
        self._write_sst()
        self._write_sst_strings()
        self._xml_end_tag('sst')
        self._xml_close()

    def _write_sst(self):
        if False:
            return 10
        xmlns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
        attributes = [('xmlns', xmlns), ('count', self.string_table.count), ('uniqueCount', self.string_table.unique_count)]
        self._xml_start_tag('sst', attributes)

    def _write_sst_strings(self):
        if False:
            print('Hello World!')
        for string in self.string_table.string_array:
            self._write_si(string)

    def _write_si(self, string):
        if False:
            return 10
        attributes = []
        string = self._escape_control_characters(string)
        if preserve_whitespace(string):
            attributes.append(('xml:space', 'preserve'))
        if string.startswith('<r>') and string.endswith('</r>'):
            self._xml_rich_si_element(string)
        else:
            self._xml_si_element(string, attributes)

class SharedStringTable(object):
    """
    A class to track Excel shared strings between worksheets.

    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.count = 0
        self.unique_count = 0
        self.string_table = {}
        self.string_array = []

    def _get_shared_string_index(self, string):
        if False:
            print('Hello World!')
        ' " Get the index of the string in the Shared String table.'
        if string not in self.string_table:
            index = self.unique_count
            self.string_table[string] = index
            self.count += 1
            self.unique_count += 1
            return index
        else:
            index = self.string_table[string]
            self.count += 1
            return index

    def _get_shared_string(self, index):
        if False:
            return 10
        ' " Get a shared string from the index.'
        return self.string_array[index]

    def _sort_string_data(self):
        if False:
            print('Hello World!')
        ' " Sort the shared string data and convert from dict to list.'
        self.string_array = sorted(self.string_table, key=self.string_table.__getitem__)
        self.string_table = {}