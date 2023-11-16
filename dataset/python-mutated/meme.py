"""Module for the support of MEME motif format."""
import xml.etree.ElementTree as ET
from Bio import Align
from Bio import Seq
from Bio import motifs

def read(handle):
    if False:
        while True:
            i = 10
    'Parse the text output of the MEME program into a meme.Record object.\n\n    Examples\n    --------\n    >>> from Bio.motifs import meme\n    >>> with open("motifs/meme.INO_up800.classic.oops.xml") as f:\n    ...     record = meme.read(f)\n    >>> for motif in record:\n    ...     for sequence in motif.alignment.sequences:\n    ...         print(sequence.motif_name, sequence.sequence_name, sequence.sequence_id, sequence.strand, sequence.pvalue)\n    GSKGCATGTGAAA INO1 sequence_5 + 1.21e-08\n    GSKGCATGTGAAA FAS1 sequence_2 - 1.87e-08\n    GSKGCATGTGAAA ACC1 sequence_4 - 6.62e-08\n    GSKGCATGTGAAA CHO2 sequence_1 - 1.05e-07\n    GSKGCATGTGAAA CHO1 sequence_0 - 1.69e-07\n    GSKGCATGTGAAA FAS2 sequence_3 - 5.62e-07\n    GSKGCATGTGAAA OPI3 sequence_6 + 1.08e-06\n    TTGACWCYTGCYCWG CHO2 sequence_1 + 7.2e-10\n    TTGACWCYTGCYCWG OPI3 sequence_6 - 2.56e-08\n    TTGACWCYTGCYCWG ACC1 sequence_4 - 1.59e-07\n    TTGACWCYTGCYCWG CHO1 sequence_0 + 2.05e-07\n    TTGACWCYTGCYCWG FAS1 sequence_2 + 3.85e-07\n    TTGACWCYTGCYCWG FAS2 sequence_3 - 5.11e-07\n    TTGACWCYTGCYCWG INO1 sequence_5 + 8.01e-07\n\n    '
    record = Record()
    try:
        xml_tree = ET.parse(handle)
    except ET.ParseError:
        raise ValueError('Improper MEME XML input file. XML root tag should start with <MEME version= ...')
    __read_metadata(record, xml_tree)
    __read_alphabet(record, xml_tree)
    sequence_id_name_map = __get_sequence_id_name_map(xml_tree)
    record.sequences = list(sequence_id_name_map.keys())
    __read_motifs(record, xml_tree, sequence_id_name_map)
    return record

class Motif(motifs.Motif):
    """A subclass of Motif used in parsing MEME (and MAST) output.

    This subclass defines functions and data specific to MEME motifs.
    This includes the motif name, the evalue for a motif, and its number
    of occurrences.
    """

    def __init__(self, alphabet=None, alignment=None):
        if False:
            print('Hello World!')
        'Initialize the class.'
        motifs.Motif.__init__(self, alphabet, alignment)
        self.evalue = 0.0
        self.num_occurrences = 0
        self.name = None
        self.id = None
        self.alt_id = None

class Instance(Seq.Seq):
    """A class describing the instances of a MEME motif, and the data thereof."""

    def __init__(self, *args, **kwds):
        if False:
            i = 10
            return i + 15
        'Initialize the class.'
        Seq.Seq.__init__(self, *args, **kwds)
        self.sequence_name = ''
        self.sequence_id = ''
        self.start = 0
        self.pvalue = 1.0
        self.strand = 0
        self.length = 0
        self.motif_name = ''

class Record(list):
    """A class for holding the results of a MEME run.

    A meme.Record is an object that holds the results from running
    MEME. It implements no methods of its own.

    The meme.Record class inherits from list, so you can access individual
    motifs in the record by their index. Alternatively, you can find a motif
    by its name:

    >>> from Bio import motifs
    >>> with open("motifs/meme.INO_up800.classic.oops.xml") as f:
    ...     record = motifs.parse(f, 'MEME')
    >>> motif = record[0]
    >>> print(motif.name)
    GSKGCATGTGAAA
    >>> motif = record['GSKGCATGTGAAA']
    >>> print(motif.name)
    GSKGCATGTGAAA
    """

    def __init__(self):
        if False:
            return 10
        'Initialize the class.'
        self.version = ''
        self.datafile = ''
        self.command = ''
        self.alphabet = ''
        self.sequences = []

    def __getitem__(self, key):
        if False:
            return 10
        'Return the motif of index key.'
        if isinstance(key, str):
            for motif in self:
                if motif.name == key:
                    return motif
        else:
            return list.__getitem__(self, key)

def __read_metadata(record, xml_tree):
    if False:
        while True:
            i = 10
    record.version = xml_tree.getroot().get('version')
    record.datafile = xml_tree.find('training_set').get('primary_sequences')
    record.command = xml_tree.find('model').find('command_line').text

def __read_alphabet(record, xml_tree):
    if False:
        print('Hello World!')
    alphabet_tree = xml_tree.find('training_set').find('letter_frequencies').find('alphabet_array')
    for value in alphabet_tree.findall('value'):
        record.alphabet += value.get('letter_id')

def __get_sequence_id_name_map(xml_tree):
    if False:
        while True:
            i = 10
    return {sequence_tree.get('id'): sequence_tree.get('name') for sequence_tree in xml_tree.find('training_set').findall('sequence')}

def __read_motifs(record, xml_tree, sequence_id_name_map):
    if False:
        for i in range(10):
            print('nop')
    for motif_tree in xml_tree.find('motifs').findall('motif'):
        instances = []
        for site_tree in motif_tree.find('contributing_sites').findall('contributing_site'):
            letters = [letter_ref.get('letter_id') for letter_ref in site_tree.find('site').findall('letter_ref')]
            sequence = ''.join(letters)
            instance = Instance(sequence)
            instance.motif_name = motif_tree.get('name')
            instance.sequence_id = site_tree.get('sequence_id')
            instance.sequence_name = sequence_id_name_map[instance.sequence_id]
            instance.start = int(site_tree.get('position')) + 1
            instance.pvalue = float(site_tree.get('pvalue'))
            instance.strand = __convert_strand(site_tree.get('strand'))
            instance.length = len(sequence)
            instances.append(instance)
        alignment = Align.Alignment(instances)
        motif = Motif(record.alphabet, alignment)
        motif.id = motif_tree.get('id')
        motif.name = motif_tree.get('name')
        motif.alt_id = motif_tree.get('alt')
        motif.length = int(motif_tree.get('width'))
        motif.num_occurrences = int(motif_tree.get('sites'))
        motif.evalue = float(motif_tree.get('e_value'))
        record.append(motif)

def __convert_strand(strand):
    if False:
        while True:
            i = 10
    'Convert strand (+/-) from XML if present.\n\n    Default: +\n    '
    if strand == 'minus':
        return '-'
    if strand == 'plus' or strand == 'none':
        return '+'
if __name__ == '__main__':
    from Bio._utils import run_doctest
    run_doctest()