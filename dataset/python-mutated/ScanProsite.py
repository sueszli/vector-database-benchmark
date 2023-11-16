"""Code for calling and parsing ScanProsite from ExPASy."""
from urllib.request import urlopen
from urllib.parse import urlencode
from xml.sax import handler
from xml.sax.expatreader import ExpatParser

class Record(list):
    """Represents search results returned by ScanProsite.

    This record is a list containing the search results returned by
    ScanProsite. The record also contains the data members n_match,
    n_seq, capped, and warning.
    """

    def __init__(self):
        if False:
            return 10
        'Initialize the class.'
        self.n_match = None
        self.n_seq = None
        self.capped = None
        self.warning = None

def scan(seq='', mirror='https://prosite.expasy.org', output='xml', **keywords):
    if False:
        for i in range(10):
            print('nop')
    'Execute a ScanProsite search.\n\n    Arguments:\n     - mirror:   The ScanProsite mirror to be used\n                 (default: https://prosite.expasy.org).\n     - seq:      The query sequence, or UniProtKB (Swiss-Prot,\n                 TrEMBL) accession\n     - output:   Format of the search results\n                 (default: xml)\n\n    Further search parameters can be passed as keywords; see the\n    documentation for programmatic access to ScanProsite at\n    https://prosite.expasy.org/scanprosite/scanprosite_doc.html\n    for a description of such parameters.\n\n    This function returns a handle to the search results returned by\n    ScanProsite. Search results in the XML format can be parsed into a\n    Python object, by using the Bio.ExPASy.ScanProsite.read function.\n\n    '
    parameters = {'seq': seq, 'output': output}
    for (key, value) in keywords.items():
        if value is not None:
            parameters[key] = value
    command = urlencode(parameters)
    url = f'{mirror}/cgi-bin/prosite/PSScan.cgi?{command}'
    handle = urlopen(url)
    return handle

def read(handle):
    if False:
        for i in range(10):
            print('nop')
    'Parse search results returned by ScanProsite into a Python object.'
    content_handler = ContentHandler()
    saxparser = Parser()
    saxparser.setContentHandler(content_handler)
    saxparser.parse(handle)
    record = content_handler.record
    return record

class Parser(ExpatParser):
    """Process the result from a ScanProsite search (PRIVATE)."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the class.'
        ExpatParser.__init__(self)
        self.firsttime = True

    def feed(self, data, isFinal=0):
        if False:
            print('Hello World!')
        'Raise an Error if plain text is received in the data.\n\n        This is to show the Error messages returned by ScanProsite.\n        '
        if self.firsttime:
            if data[:5].decode('utf-8') != '<?xml':
                raise ValueError(data)
        self.firsttime = False
        return ExpatParser.feed(self, data, isFinal)

class ContentHandler(handler.ContentHandler):
    """Process and fill in the records, results of the search (PRIVATE)."""
    integers = ('start', 'stop')
    strings = ('sequence_ac', 'sequence_id', 'sequence_db', 'signature_ac', 'level', 'level_tag')

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the class.'
        self.element = []

    def startElement(self, name, attrs):
        if False:
            print('Hello World!')
        'Define the beginning of a record and stores the search record.'
        self.element.append(name)
        self.content = ''
        if self.element == ['matchset']:
            self.record = Record()
            self.record.n_match = int(attrs['n_match'])
            self.record.n_seq = int(attrs['n_seq'])
        elif self.element == ['matchset', 'match']:
            match = {}
            self.record.append(match)

    def endElement(self, name):
        if False:
            print('Hello World!')
        'Define the end of the search record.'
        assert name == self.element.pop()
        if self.element == ['matchset', 'match']:
            match = self.record[-1]
            if name in ContentHandler.integers:
                match[name] = int(self.content)
            elif name in ContentHandler.strings:
                match[name] = self.content
            else:
                match[name] = self.content

    def characters(self, content):
        if False:
            i = 10
            return i + 15
        'Store the record content.'
        self.content += content