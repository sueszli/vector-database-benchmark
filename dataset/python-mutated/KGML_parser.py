"""Classes and functions to parse a KGML pathway map.

The KGML pathway map is parsed into the object structure defined in
KGML_Pathway.py in this module.

Classes:
 - KGMLParser - Parses KGML file

Functions:
 - read - Returns a single Pathway object, using KGMLParser internally

"""
from xml.etree import ElementTree
from io import StringIO
from Bio.KEGG.KGML.KGML_pathway import Component, Entry, Graphics
from Bio.KEGG.KGML.KGML_pathway import Pathway, Reaction, Relation

def read(handle):
    if False:
        i = 10
        return i + 15
    'Parse a single KEGG Pathway from given file handle.\n\n    Returns a single Pathway object.  There should be one and only\n    one pathway in each file, but there may well be pathological\n    examples out there.\n    '
    pathways = parse(handle)
    try:
        pathway = next(pathways)
    except StopIteration:
        raise ValueError('No pathways found in handle') from None
    try:
        next(pathways)
        raise ValueError('More than one pathway found in handle')
    except StopIteration:
        pass
    return pathway

def parse(handle):
    if False:
        return 10
    'Return an iterator over Pathway elements.\n\n    Arguments:\n     - handle - file handle to a KGML file for parsing, or a KGML string\n\n    This is a generator for the return of multiple Pathway objects.\n\n    '
    try:
        handle.read(0)
    except AttributeError:
        try:
            handle = StringIO(handle)
        except TypeError:
            raise TypeError('An XML-containing handle or an XML string must be provided') from None
    for (event, elem) in ElementTree.iterparse(handle, events=('start', 'end')):
        if event == 'end' and elem.tag == 'pathway':
            yield KGMLParser(elem).parse()
            elem.clear()

class KGMLParser:
    """Parses a KGML XML Pathway entry into a Pathway object.

    Example: Read and parse large metabolism file

    >>> from Bio.KEGG.KGML.KGML_parser import read
    >>> pathway = read(open('KEGG/ko01100.xml', 'r'))
    >>> print(len(pathway.entries))
    3628
    >>> print(len(pathway.reactions))
    1672
    >>> print(len(pathway.maps))
    149

    >>> pathway = read(open('KEGG/ko00010.xml', 'r'))
    >>> print(pathway) #doctest: +NORMALIZE_WHITESPACE
    Pathway: Glycolysis / Gluconeogenesis
    KEGG ID: path:ko00010
    Image file: http://www.kegg.jp/kegg/pathway/ko/ko00010.png
    Organism: ko
    Entries: 99
    Entry types:
        ortholog: 61
        compound: 31
        map: 7

    """

    def __init__(self, elem):
        if False:
            while True:
                i = 10
        'Initialize the class.'
        self.entry = elem

    def parse(self):
        if False:
            return 10
        'Parse the input elements.'

        def _parse_pathway(attrib):
            if False:
                while True:
                    i = 10
            for (k, v) in attrib.items():
                self.pathway.__setattr__(k, v)

        def _parse_entry(element):
            if False:
                i = 10
                return i + 15
            new_entry = Entry()
            for (k, v) in element.attrib.items():
                new_entry.__setattr__(k, v)
            for subelement in element:
                if subelement.tag == 'graphics':
                    _parse_graphics(subelement, new_entry)
                elif subelement.tag == 'component':
                    _parse_component(subelement, new_entry)
            self.pathway.add_entry(new_entry)

        def _parse_graphics(element, entry):
            if False:
                while True:
                    i = 10
            new_graphics = Graphics(entry)
            for (k, v) in element.attrib.items():
                new_graphics.__setattr__(k, v)
            entry.add_graphics(new_graphics)

        def _parse_component(element, entry):
            if False:
                print('Hello World!')
            new_component = Component(entry)
            for (k, v) in element.attrib.items():
                new_component.__setattr__(k, v)
            entry.add_component(new_component)

        def _parse_reaction(element):
            if False:
                i = 10
                return i + 15
            new_reaction = Reaction()
            for (k, v) in element.attrib.items():
                new_reaction.__setattr__(k, v)
            for subelement in element:
                if subelement.tag == 'substrate':
                    new_reaction.add_substrate(int(subelement.attrib['id']))
                elif subelement.tag == 'product':
                    new_reaction.add_product(int(subelement.attrib['id']))
            self.pathway.add_reaction(new_reaction)

        def _parse_relation(element):
            if False:
                i = 10
                return i + 15
            new_relation = Relation()
            new_relation.entry1 = int(element.attrib['entry1'])
            new_relation.entry2 = int(element.attrib['entry2'])
            new_relation.type = element.attrib['type']
            for subtype in element:
                (name, value) = (subtype.attrib['name'], subtype.attrib['value'])
                if name in ('compound', 'hidden compound'):
                    new_relation.subtypes.append((name, int(value)))
                else:
                    new_relation.subtypes.append((name, value))
            self.pathway.add_relation(new_relation)
        self.pathway = Pathway()
        _parse_pathway(self.entry.attrib)
        for element in self.entry:
            if element.tag == 'entry':
                _parse_entry(element)
            elif element.tag == 'reaction':
                _parse_reaction(element)
            elif element.tag == 'relation':
                _parse_relation(element)
            else:
                import warnings
                from Bio import BiopythonParserWarning
                warnings.warn(f'Warning: tag {element.tag} not implemented in parser', BiopythonParserWarning)
        return self.pathway
if __name__ == '__main__':
    from Bio._utils import run_doctest
    run_doctest(verbose=0)