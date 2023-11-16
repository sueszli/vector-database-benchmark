"""Internal code for parsing GenBank and EMBL files (PRIVATE).

This code is NOT intended for direct use.  It provides a basic scanner
(for use with a event consumer such as Bio.GenBank._FeatureConsumer)
to parse a GenBank or EMBL file (with their shared INSDC feature table).

It is used by Bio.GenBank to parse GenBank files
It is also used by Bio.SeqIO to parse GenBank and EMBL files

Feature Table Documentation:

- http://www.insdc.org/files/feature_table.html
- http://www.ncbi.nlm.nih.gov/projects/collab/FT/index.html
- ftp://ftp.ncbi.nih.gov/genbank/docs/
"""
import warnings
import re
import sys
from collections import defaultdict
from Bio.File import as_handle
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonParserWarning
from typing import List

class InsdcScanner:
    """Basic functions for breaking up a GenBank/EMBL file into sub sections.

    The International Nucleotide Sequence Database Collaboration (INSDC)
    between the DDBJ, EMBL, and GenBank.  These organisations all use the
    same "Feature Table" layout in their plain text flat file formats.

    However, the header and sequence sections of an EMBL file are very
    different in layout to those produced by GenBank/DDBJ.
    """
    RECORD_START = 'XXX'
    HEADER_WIDTH = 3
    FEATURE_START_MARKERS = ['XXX***FEATURES***XXX']
    FEATURE_END_MARKERS = ['XXX***END FEATURES***XXX']
    FEATURE_QUALIFIER_INDENT = 0
    FEATURE_QUALIFIER_SPACER = ''
    SEQUENCE_HEADERS = ['XXX']

    def __init__(self, debug=0):
        if False:
            print('Hello World!')
        'Initialize the class.'
        assert len(self.RECORD_START) == self.HEADER_WIDTH
        for marker in self.SEQUENCE_HEADERS:
            assert marker == marker.rstrip()
        assert len(self.FEATURE_QUALIFIER_SPACER) == self.FEATURE_QUALIFIER_INDENT
        self.debug = debug
        self.handle = None
        self.line = None

    def set_handle(self, handle):
        if False:
            print('Hello World!')
        'Set the handle attribute.'
        self.handle = handle
        self.line = ''

    def find_start(self):
        if False:
            for i in range(10):
                print('nop')
        'Read in lines until find the ID/LOCUS line, which is returned.\n\n        Any preamble (such as the header used by the NCBI on ``*.seq.gz`` archives)\n        will we ignored.\n        '
        while True:
            if self.line:
                line = self.line
                self.line = ''
            else:
                line = self.handle.readline()
            if not line:
                if self.debug:
                    print('End of file')
                return None
            if isinstance(line[0], int):
                raise ValueError('Is this handle in binary mode not text mode?')
            if line[:self.HEADER_WIDTH] == self.RECORD_START:
                if self.debug > 1:
                    print('Found the start of a record:\n' + line)
                break
            line = line.rstrip()
            if line == '//':
                if self.debug > 1:
                    print('Skipping // marking end of last record')
            elif line == '':
                if self.debug > 1:
                    print('Skipping blank line before record')
            elif self.debug > 1:
                print('Skipping header line before record:\n' + line)
        self.line = line
        return line

    def parse_header(self):
        if False:
            print('Hello World!')
        'Return list of strings making up the header.\n\n        New line characters are removed.\n\n        Assumes you have just read in the ID/LOCUS line.\n        '
        if self.line[:self.HEADER_WIDTH] != self.RECORD_START:
            raise ValueError('Not at start of record')
        header_lines = []
        while True:
            line = self.handle.readline()
            if not line:
                raise ValueError('Premature end of line during sequence data')
            line = line.rstrip()
            if line in self.FEATURE_START_MARKERS:
                if self.debug:
                    print('Found feature table')
                break
            if line[:self.HEADER_WIDTH].rstrip() in self.SEQUENCE_HEADERS:
                if self.debug:
                    print('Found start of sequence')
                break
            if line == '//':
                raise ValueError("Premature end of sequence data marker '//' found")
            header_lines.append(line)
        self.line = line
        return header_lines

    def parse_features(self, skip=False):
        if False:
            return 10
        'Return list of tuples for the features (if present).\n\n        Each feature is returned as a tuple (key, location, qualifiers)\n        where key and location are strings (e.g. "CDS" and\n        "complement(join(490883..490885,1..879))") while qualifiers\n        is a list of two string tuples (feature qualifier keys and values).\n\n        Assumes you have already read to the start of the features table.\n        '
        if self.line.rstrip() not in self.FEATURE_START_MARKERS:
            if self.debug:
                print("Didn't find any feature table")
            return []
        while self.line.rstrip() in self.FEATURE_START_MARKERS:
            self.line = self.handle.readline()
        features = []
        line = self.line
        while True:
            if not line:
                raise ValueError('Premature end of line during features table')
            if line[:self.HEADER_WIDTH].rstrip() in self.SEQUENCE_HEADERS:
                if self.debug:
                    print('Found start of sequence')
                break
            line = line.rstrip()
            if line == '//':
                raise ValueError("Premature end of features table, marker '//' found")
            if line in self.FEATURE_END_MARKERS:
                if self.debug:
                    print('Found end of features')
                line = self.handle.readline()
                break
            if line[2:self.FEATURE_QUALIFIER_INDENT].strip() == '':
                line = self.handle.readline()
                continue
            if len(line) < self.FEATURE_QUALIFIER_INDENT:
                warnings.warn(f'line too short to contain a feature: {line!r}', BiopythonParserWarning)
                line = self.handle.readline()
                continue
            if skip:
                line = self.handle.readline()
                while line[:self.FEATURE_QUALIFIER_INDENT] == self.FEATURE_QUALIFIER_SPACER:
                    line = self.handle.readline()
            else:
                if line[self.FEATURE_QUALIFIER_INDENT] != ' ' and ' ' in line[self.FEATURE_QUALIFIER_INDENT:]:
                    (feature_key, line) = line[2:].strip().split(None, 1)
                    feature_lines = [line]
                    warnings.warn(f'Over indented {feature_key} feature?', BiopythonParserWarning)
                else:
                    feature_key = line[2:self.FEATURE_QUALIFIER_INDENT].strip()
                    feature_lines = [line[self.FEATURE_QUALIFIER_INDENT:]]
                line = self.handle.readline()
                while line[:self.FEATURE_QUALIFIER_INDENT] == self.FEATURE_QUALIFIER_SPACER or (line != '' and line.rstrip() == ''):
                    feature_lines.append(line[self.FEATURE_QUALIFIER_INDENT:].strip())
                    line = self.handle.readline()
                features.append(self.parse_feature(feature_key, feature_lines))
        self.line = line
        return features

    def parse_feature(self, feature_key, lines):
        if False:
            for i in range(10):
                print('nop')
        'Parse a feature given as a list of strings into a tuple.\n\n        Expects a feature as a list of strings, returns a tuple (key, location,\n        qualifiers)\n\n        For example given this GenBank feature::\n\n             CDS             complement(join(490883..490885,1..879))\n                             /locus_tag="NEQ001"\n                             /note="conserved hypothetical [Methanococcus jannaschii];\n                             COG1583:Uncharacterized ACR; IPR001472:Bipartite nuclear\n                             localization signal; IPR002743: Protein of unknown\n                             function DUF57"\n                             /codon_start=1\n                             /transl_table=11\n                             /product="hypothetical protein"\n                             /protein_id="NP_963295.1"\n                             /db_xref="GI:41614797"\n                             /db_xref="GeneID:2732620"\n                             /translation="MRLLLELKALNSIDKKQLSNYLIQGFIYNILKNTEYSWLHNWKK\n                             EKYFNFTLIPKKDIIENKRYYLIISSPDKRFIEVLHNKIKDLDIITIGLAQFQLRKTK\n                             KFDPKLRFPWVTITPIVLREGKIVILKGDKYYKVFVKRLEELKKYNLIKKKEPILEEP\n                             IEISLNQIKDGWKIIDVKDRYYDFRNKSFSAFSNWLRDLKEQSLRKYNNFCGKNFYFE\n                             EAIFEGFTFYKTVSIRIRINRGEAVYIGTLWKELNVYRKLDKEEREFYKFLYDCGLGS\n                             LNSMGFGFVNTKKNSAR"\n\n        Then should give input key="CDS" and the rest of the data as a list of strings\n        lines=["complement(join(490883..490885,1..879))", ..., "LNSMGFGFVNTKKNSAR"]\n        where the leading spaces and trailing newlines have been removed.\n\n        Returns tuple containing: (key as string, location string, qualifiers as list)\n        as follows for this example:\n\n        key = "CDS", string\n        location = "complement(join(490883..490885,1..879))", string\n        qualifiers = list of string tuples:\n\n        [(\'locus_tag\', \'"NEQ001"\'),\n         (\'note\', \'"conserved hypothetical [Methanococcus jannaschii];\\nCOG1583:..."\'),\n         (\'codon_start\', \'1\'),\n         (\'transl_table\', \'11\'),\n         (\'product\', \'"hypothetical protein"\'),\n         (\'protein_id\', \'"NP_963295.1"\'),\n         (\'db_xref\', \'"GI:41614797"\'),\n         (\'db_xref\', \'"GeneID:2732620"\'),\n         (\'translation\', \'"MRLLLELKALNSIDKKQLSNYLIQGFIYNILKNTEYSWLHNWKK\\nEKYFNFT..."\')]\n\n        In the above example, the "note" and "translation" were edited for compactness,\n        and they would contain multiple new line characters (displayed above as \\n)\n\n        If a qualifier is quoted (in this case, everything except codon_start and\n        transl_table) then the quotes are NOT removed.\n\n        Note that no whitespace is removed.\n        '
        iterator = (x for x in lines if x)
        try:
            line = next(iterator)
            feature_location = line.strip()
            while feature_location[-1:] == ',':
                line = next(iterator)
                feature_location += line.strip()
            if feature_location.count('(') > feature_location.count(')'):
                warnings.warn("Non-standard feature line wrapping (didn't break on comma)?", BiopythonParserWarning)
                while feature_location[-1:] == ',' or feature_location.count('(') > feature_location.count(')'):
                    line = next(iterator)
                    feature_location += line.strip()
            qualifiers = []
            for (line_number, line) in enumerate(iterator):
                if line_number == 0 and line.startswith(')'):
                    feature_location += line.strip()
                elif line[0] == '/':
                    i = line.find('=')
                    key = line[1:i]
                    value = line[i + 1:]
                    if i and value.startswith(' ') and value.lstrip().startswith('"'):
                        warnings.warn('White space after equals in qualifier', BiopythonParserWarning)
                        value = value.lstrip()
                    if i == -1:
                        key = line[1:]
                        qualifiers.append((key, None))
                    elif not value:
                        qualifiers.append((key, ''))
                    elif value == '"':
                        if self.debug:
                            print(f'Single quote {key}:{value}')
                        qualifiers.append((key, value))
                    elif value[0] == '"':
                        value_list = [value]
                        while value_list[-1][-1] != '"':
                            value_list.append(next(iterator))
                        value = '\n'.join(value_list)
                        qualifiers.append((key, value))
                    else:
                        qualifiers.append((key, value))
                else:
                    assert len(qualifiers) > 0
                    assert key == qualifiers[-1][0]
                    if qualifiers[-1][1] is None:
                        raise StopIteration
                    qualifiers[-1] = (key, qualifiers[-1][1] + '\n' + line)
            return (feature_key, feature_location, qualifiers)
        except StopIteration:
            raise ValueError("Problem with '%s' feature:\n%s" % (feature_key, '\n'.join(lines))) from None

    def parse_footer(self):
        if False:
            i = 10
            return i + 15
        'Return a tuple containing a list of any misc strings, and the sequence.'
        if self.line in self.FEATURE_END_MARKERS:
            while self.line[:self.HEADER_WIDTH].rstrip() not in self.SEQUENCE_HEADERS:
                self.line = self.handle.readline()
                if not self.line:
                    raise ValueError('Premature end of file')
                self.line = self.line.rstrip()
        if self.line[:self.HEADER_WIDTH].rstrip() not in self.SEQUENCE_HEADERS:
            raise ValueError('Not at start of sequence')
        while True:
            line = self.handle.readline()
            if not line:
                raise ValueError('Premature end of line during sequence data')
            line = line.rstrip()
            if line == '//':
                break
        self.line = line
        return ([], '')

    def _feed_first_line(self, consumer, line):
        if False:
            i = 10
            return i + 15
        'Handle the LOCUS/ID line, passing data to the consumer (PRIVATE).\n\n        This should be implemented by the EMBL / GenBank specific subclass\n\n        Used by the parse_records() and parse() methods.\n        '

    def _feed_header_lines(self, consumer, lines):
        if False:
            return 10
        'Handle the header lines (list of strings), passing data to the consumer (PRIVATE).\n\n        This should be implemented by the EMBL / GenBank specific subclass\n\n        Used by the parse_records() and parse() methods.\n        '

    @staticmethod
    def _feed_feature_table(consumer, feature_tuples):
        if False:
            i = 10
            return i + 15
        'Handle the feature table (list of tuples), passing data to the consumer (PRIVATE).\n\n        Used by the parse_records() and parse() methods.\n        '
        consumer.start_feature_table()
        for (feature_key, location_string, qualifiers) in feature_tuples:
            consumer.feature_key(feature_key)
            consumer.location(location_string)
            for (q_key, q_value) in qualifiers:
                if q_value is None:
                    consumer.feature_qualifier(q_key, q_value)
                else:
                    consumer.feature_qualifier(q_key, q_value.replace('\n', ' '))

    def _feed_misc_lines(self, consumer, lines):
        if False:
            while True:
                i = 10
        'Handle any lines between features and sequence (list of strings), passing data to the consumer (PRIVATE).\n\n        This should be implemented by the EMBL / GenBank specific subclass\n\n        Used by the parse_records() and parse() methods.\n        '

    def feed(self, handle, consumer, do_features=True):
        if False:
            print('Hello World!')
        'Feed a set of data into the consumer.\n\n        This method is intended for use with the "old" code in Bio.GenBank\n\n        Arguments:\n         - handle - A handle with the information to parse.\n         - consumer - The consumer that should be informed of events.\n         - do_features - Boolean, should the features be parsed?\n           Skipping the features can be much faster.\n\n        Return values:\n         - true  - Passed a record\n         - false - Did not find a record\n\n        '
        self.set_handle(handle)
        if not self.find_start():
            consumer.data = None
            return False
        self._feed_first_line(consumer, self.line)
        self._feed_header_lines(consumer, self.parse_header())
        if do_features:
            self._feed_feature_table(consumer, self.parse_features(skip=False))
        else:
            self.parse_features(skip=True)
        (misc_lines, sequence_string) = self.parse_footer()
        self._feed_misc_lines(consumer, misc_lines)
        consumer.sequence(sequence_string)
        consumer.record_end('//')
        assert self.line == '//'
        return True

    def parse(self, handle, do_features=True):
        if False:
            print('Hello World!')
        'Return a SeqRecord (with SeqFeatures if do_features=True).\n\n        See also the method parse_records() for use on multi-record files.\n        '
        from Bio.GenBank import _FeatureConsumer
        from Bio.GenBank.utils import FeatureValueCleaner
        consumer = _FeatureConsumer(use_fuzziness=1, feature_cleaner=FeatureValueCleaner())
        if self.feed(handle, consumer, do_features):
            return consumer.data
        else:
            return None

    def parse_records(self, handle, do_features=True):
        if False:
            print('Hello World!')
        'Parse records, return a SeqRecord object iterator.\n\n        Each record (from the ID/LOCUS line to the // line) becomes a SeqRecord\n\n        The SeqRecord objects include SeqFeatures if do_features=True\n\n        This method is intended for use in Bio.SeqIO\n        '
        with as_handle(handle) as handle:
            while True:
                record = self.parse(handle, do_features)
                if record is None:
                    break
                if record.id is None:
                    raise ValueError("Failed to parse the record's ID. Invalid ID line?")
                if record.name == '<unknown name>':
                    raise ValueError("Failed to parse the record's name. Invalid ID line?")
                if record.description == '<unknown description>':
                    raise ValueError("Failed to parse the record's description")
                yield record

    def parse_cds_features(self, handle, alphabet=None, tags2id=('protein_id', 'locus_tag', 'product')):
        if False:
            print('Hello World!')
        'Parse CDS features, return SeqRecord object iterator.\n\n        Each CDS feature becomes a SeqRecord.\n\n        Arguments:\n         - alphabet - Obsolete, should be left as None.\n         - tags2id  - Tuple of three strings, the feature keys to use\n           for the record id, name and description,\n\n        This method is intended for use in Bio.SeqIO\n\n        '
        if alphabet is not None:
            raise ValueError('The alphabet argument is no longer supported')
        with as_handle(handle) as handle:
            self.set_handle(handle)
            while self.find_start():
                self.parse_header()
                feature_tuples = self.parse_features()
                while True:
                    line = self.handle.readline()
                    if not line:
                        break
                    if line[:2] == '//':
                        break
                self.line = line.rstrip()
                for (key, location_string, qualifiers) in feature_tuples:
                    if key == 'CDS':
                        record = SeqRecord(seq=None)
                        annotations = record.annotations
                        annotations['molecule_type'] = 'protein'
                        annotations['raw_location'] = location_string.replace(' ', '')
                        for (qualifier_name, qualifier_data) in qualifiers:
                            if qualifier_data is not None and qualifier_data[0] == '"' and (qualifier_data[-1] == '"'):
                                qualifier_data = qualifier_data[1:-1]
                            if qualifier_name == 'translation':
                                assert record.seq is None, 'Multiple translations!'
                                record.seq = Seq(qualifier_data.replace('\n', ''))
                            elif qualifier_name == 'db_xref':
                                record.dbxrefs.append(qualifier_data)
                            else:
                                if qualifier_data is not None:
                                    qualifier_data = qualifier_data.replace('\n', ' ').replace('  ', ' ')
                                try:
                                    annotations[qualifier_name] += ' ' + qualifier_data
                                except KeyError:
                                    annotations[qualifier_name] = qualifier_data
                        try:
                            record.id = annotations[tags2id[0]]
                        except KeyError:
                            pass
                        try:
                            record.name = annotations[tags2id[1]]
                        except KeyError:
                            pass
                        try:
                            record.description = annotations[tags2id[2]]
                        except KeyError:
                            pass
                        yield record

class EmblScanner(InsdcScanner):
    """For extracting chunks of information in EMBL files."""
    RECORD_START = 'ID   '
    HEADER_WIDTH = 5
    FEATURE_START_MARKERS = ['FH   Key             Location/Qualifiers', 'FH']
    FEATURE_END_MARKERS = ['XX']
    FEATURE_QUALIFIER_INDENT = 21
    FEATURE_QUALIFIER_SPACER = 'FT' + ' ' * (FEATURE_QUALIFIER_INDENT - 2)
    SEQUENCE_HEADERS = ['SQ', 'CO']
    EMBL_INDENT = HEADER_WIDTH
    EMBL_SPACER = ' ' * EMBL_INDENT

    def parse_footer(self):
        if False:
            i = 10
            return i + 15
        'Return a tuple containing a list of any misc strings, and the sequence.'
        if self.line[:self.HEADER_WIDTH].rstrip() not in self.SEQUENCE_HEADERS:
            raise ValueError(f"Footer format unexpected: '{self.line}'")
        misc_lines = []
        while self.line[:self.HEADER_WIDTH].rstrip() in self.SEQUENCE_HEADERS:
            misc_lines.append(self.line)
            self.line = self.handle.readline()
            if not self.line:
                raise ValueError('Premature end of file')
            self.line = self.line.rstrip()
        if not (self.line[:self.HEADER_WIDTH] == ' ' * self.HEADER_WIDTH or self.line.strip() == '//'):
            raise ValueError(f'Unexpected content after SQ or CO line: {self.line!r}')
        seq_lines = []
        line = self.line
        while True:
            if not line:
                raise ValueError('Premature end of file in sequence data')
            line = line.strip()
            if not line:
                raise ValueError('Blank line in sequence data')
            if line == '//':
                break
            if self.line[:self.HEADER_WIDTH] != ' ' * self.HEADER_WIDTH:
                raise ValueError('Problem with characters in header line,  or incorrect header width: ' + self.line)
            linersplit = line.rsplit(None, 1)
            if len(linersplit) == 2 and linersplit[1].isdigit():
                seq_lines.append(linersplit[0])
            elif line.isdigit():
                pass
            else:
                warnings.warn('EMBL sequence line missing coordinates', BiopythonParserWarning)
                seq_lines.append(line)
            line = self.handle.readline()
        self.line = line
        return (misc_lines, ''.join(seq_lines).replace(' ', ''))

    def _feed_first_line(self, consumer, line):
        if False:
            print('Hello World!')
        assert line[:self.HEADER_WIDTH].rstrip() == 'ID'
        if line[self.HEADER_WIDTH:].count(';') == 6:
            self._feed_first_line_new(consumer, line)
        elif line[self.HEADER_WIDTH:].count(';') == 3:
            if line.rstrip().endswith(' SQ'):
                self._feed_first_line_patents(consumer, line)
            else:
                self._feed_first_line_old(consumer, line)
        elif line[self.HEADER_WIDTH:].count(';') == 2:
            self._feed_first_line_patents_kipo(consumer, line)
        else:
            raise ValueError('Did not recognise the ID line layout:\n' + line)

    def _feed_first_line_patents(self, consumer, line):
        if False:
            i = 10
            return i + 15
        fields = [data.strip() for data in line[self.HEADER_WIDTH:].strip()[:-3].split(';')]
        assert len(fields) == 4
        consumer.locus(fields[0])
        consumer.residue_type(fields[1])
        consumer.data_file_division(fields[2])

    def _feed_first_line_patents_kipo(self, consumer, line):
        if False:
            while True:
                i = 10
        assert line[:self.HEADER_WIDTH].rstrip() == 'ID'
        fields = [line[self.HEADER_WIDTH:].split(None, 1)[0]]
        fields.extend(line[self.HEADER_WIDTH:].split(None, 1)[1].split(';'))
        fields = [entry.strip() for entry in fields]
        "\n        The tokens represent:\n\n           0. Primary accession number\n           (space sep)\n           1. ??? (e.g. standard)\n           (semi-colon)\n           2. Molecule type (protein)? Division? Always 'PRT'\n           3. Sequence length (e.g. '111 AA.')\n        "
        consumer.locus(fields[0])
        self._feed_seq_length(consumer, fields[3])

    def _feed_first_line_old(self, consumer, line):
        if False:
            return 10
        assert line[:self.HEADER_WIDTH].rstrip() == 'ID'
        fields = [line[self.HEADER_WIDTH:].split(None, 1)[0]]
        fields.extend(line[self.HEADER_WIDTH:].split(None, 1)[1].split(';'))
        fields = [entry.strip() for entry in fields]
        "\n        The tokens represent:\n\n           0. Primary accession number\n           (space sep)\n           1. ??? (e.g. standard)\n           (semi-colon)\n           2. Topology and/or Molecule type (e.g. 'circular DNA' or 'DNA')\n           3. Taxonomic division (e.g. 'PRO')\n           4. Sequence length (e.g. '4639675 BP.')\n\n        "
        consumer.locus(fields[0])
        consumer.residue_type(fields[2])
        if 'circular' in fields[2]:
            consumer.topology('circular')
            consumer.molecule_type(fields[2].replace('circular', '').strip())
        elif 'linear' in fields[2]:
            consumer.topology('linear')
            consumer.molecule_type(fields[2].replace('linear', '').strip())
        else:
            consumer.molecule_type(fields[2].strip())
        consumer.data_file_division(fields[3])
        self._feed_seq_length(consumer, fields[4])

    def _feed_first_line_new(self, consumer, line):
        if False:
            return 10
        assert line[:self.HEADER_WIDTH].rstrip() == 'ID'
        fields = [data.strip() for data in line[self.HEADER_WIDTH:].strip().split(';')]
        assert len(fields) == 7
        "\n        The tokens represent:\n\n           0. Primary accession number\n           1. Sequence version number\n           2. Topology: 'circular' or 'linear'\n           3. Molecule type (e.g. 'genomic DNA')\n           4. Data class (e.g. 'STD')\n           5. Taxonomic division (e.g. 'PRO')\n           6. Sequence length (e.g. '4639675 BP.')\n\n        "
        consumer.locus(fields[0])
        consumer.accession(fields[0])
        version_parts = fields[1].split()
        if len(version_parts) == 2 and version_parts[0] == 'SV' and version_parts[1].isdigit():
            consumer.version_suffix(version_parts[1])
        consumer.residue_type(' '.join(fields[2:4]))
        consumer.topology(fields[2])
        consumer.molecule_type(fields[3])
        consumer.data_file_division(fields[5])
        self._feed_seq_length(consumer, fields[6])

    @staticmethod
    def _feed_seq_length(consumer, text):
        if False:
            for i in range(10):
                print('nop')
        length_parts = text.split()
        assert len(length_parts) == 2, f'Invalid sequence length string {text!r}'
        assert length_parts[1].upper() in ['BP', 'BP.', 'AA', 'AA.']
        consumer.size(length_parts[0])

    def _feed_header_lines(self, consumer, lines):
        if False:
            while True:
                i = 10
        consumer_dict = {'AC': 'accession', 'SV': 'version', 'DE': 'definition', 'RG': 'consrtm', 'RL': 'journal', 'OS': 'organism', 'OC': 'taxonomy', 'CC': 'comment'}
        for line in lines:
            line_type = line[:self.EMBL_INDENT].strip()
            data = line[self.EMBL_INDENT:].strip()
            if line_type == 'XX':
                pass
            elif line_type == 'RN':
                if data[0] == '[' and data[-1] == ']':
                    data = data[1:-1]
                consumer.reference_num(data)
            elif line_type == 'RP':
                if data.strip() == '[-]':
                    pass
                else:
                    parts = [bases.replace('-', ' to ').strip() for bases in data.split(',') if bases.strip()]
                    consumer.reference_bases(f"(bases {'; '.join(parts)})")
            elif line_type == 'RT':
                if data.startswith('"'):
                    data = data[1:]
                if data.endswith('";'):
                    data = data[:-2]
                consumer.title(data)
            elif line_type == 'RX':
                (key, value) = data.split(';', 1)
                if value.endswith('.'):
                    value = value[:-1]
                value = value.strip()
                if key == 'PUBMED':
                    consumer.pubmed_id(value)
            elif line_type == 'CC':
                consumer.comment([data])
            elif line_type == 'DR':
                parts = data.rstrip('.').split(';')
                if len(parts) == 1:
                    warnings.warn('Malformed DR line in EMBL file.', BiopythonParserWarning)
                else:
                    consumer.dblink(f'{parts[0].strip()}:{parts[1].strip()}')
            elif line_type == 'RA':
                consumer.authors(data.rstrip(';'))
            elif line_type == 'PR':
                if data.startswith('Project:'):
                    consumer.project(data.rstrip(';'))
            elif line_type == 'KW':
                consumer.keywords(data.rstrip(';'))
            elif line_type in consumer_dict:
                getattr(consumer, consumer_dict[line_type])(data)
            elif self.debug:
                print(f'Ignoring EMBL header line:\n{line}')

    def _feed_misc_lines(self, consumer, lines):
        if False:
            while True:
                i = 10
        lines.append('')
        line_iter = iter(lines)
        try:
            for line in line_iter:
                if line.startswith('CO   '):
                    line = line[5:].strip()
                    contig_location = line
                    while True:
                        line = next(line_iter)
                        if not line:
                            break
                        elif line.startswith('CO   '):
                            contig_location += line[5:].strip()
                        else:
                            raise ValueError('Expected CO (contig) continuation line, got:\n' + line)
                    consumer.contig_location(contig_location)
                if line.startswith('SQ   Sequence '):
                    self._feed_seq_length(consumer, line[14:].rstrip().rstrip(';').split(';', 1)[0])
            return
        except StopIteration:
            raise ValueError('Problem in misc lines before sequence') from None

class _ImgtScanner(EmblScanner):
    """For extracting chunks of information in IMGT (EMBL like) files (PRIVATE).

    IMGT files are like EMBL files but in order to allow longer feature types
    the features should be indented by 25 characters not 21 characters. In
    practice the IMGT flat files tend to use either 21 or 25 characters, so we
    must cope with both.

    This is private to encourage use of Bio.SeqIO rather than Bio.GenBank.
    """
    FEATURE_START_MARKERS = ['FH   Key             Location/Qualifiers', 'FH   Key             Location/Qualifiers (from EMBL)', 'FH   Key                 Location/Qualifiers', 'FH']

    def _feed_first_line(self, consumer, line):
        if False:
            i = 10
            return i + 15
        assert line[:self.HEADER_WIDTH].rstrip() == 'ID'
        if line[self.HEADER_WIDTH:].count(';') != 5:
            return EmblScanner._feed_first_line(self, consumer, line)
        fields = [data.strip() for data in line[self.HEADER_WIDTH:].strip().split(';')]
        assert len(fields) == 6
        "\n        The tokens represent:\n\n           0. Primary accession number (eg 'HLA00001')\n           1. Sequence version number (eg 'SV 1')\n           2. ??? eg 'standard'\n           3. Molecule type (e.g. 'DNA')\n           4. Taxonomic division (e.g. 'HUM')\n           5. Sequence length (e.g. '3503 BP.')\n        "
        consumer.locus(fields[0])
        version_parts = fields[1].split()
        if len(version_parts) == 2 and version_parts[0] == 'SV' and version_parts[1].isdigit():
            consumer.version_suffix(version_parts[1])
        consumer.residue_type(fields[3])
        if 'circular' in fields[3]:
            consumer.topology('circular')
            consumer.molecule_type(fields[3].replace('circular', '').strip())
        elif 'linear' in fields[3]:
            consumer.topology('linear')
            consumer.molecule_type(fields[3].replace('linear', '').strip())
        else:
            consumer.molecule_type(fields[3].strip())
        consumer.data_file_division(fields[4])
        self._feed_seq_length(consumer, fields[5])

    def parse_features(self, skip=False):
        if False:
            return 10
        'Return list of tuples for the features (if present).\n\n        Each feature is returned as a tuple (key, location, qualifiers)\n        where key and location are strings (e.g. "CDS" and\n        "complement(join(490883..490885,1..879))") while qualifiers\n        is a list of two string tuples (feature qualifier keys and values).\n\n        Assumes you have already read to the start of the features table.\n        '
        if self.line.rstrip() not in self.FEATURE_START_MARKERS:
            if self.debug:
                print("Didn't find any feature table")
            return []
        while self.line.rstrip() in self.FEATURE_START_MARKERS:
            self.line = self.handle.readline()
        bad_position_re = re.compile('([0-9]+)>')
        features = []
        line = self.line
        while True:
            if not line:
                raise ValueError('Premature end of line during features table')
            if line[:self.HEADER_WIDTH].rstrip() in self.SEQUENCE_HEADERS:
                if self.debug:
                    print('Found start of sequence')
                break
            line = line.rstrip()
            if line == '//':
                raise ValueError("Premature end of features table, marker '//' found")
            if line in self.FEATURE_END_MARKERS:
                if self.debug:
                    print('Found end of features')
                line = self.handle.readline()
                break
            if line[2:self.FEATURE_QUALIFIER_INDENT].strip() == '':
                line = self.handle.readline()
                continue
            if skip:
                line = self.handle.readline()
                while line[:self.FEATURE_QUALIFIER_INDENT] == self.FEATURE_QUALIFIER_SPACER:
                    line = self.handle.readline()
            else:
                assert line[:2] == 'FT'
                try:
                    (feature_key, location_start) = line[2:].strip().split()
                except ValueError:
                    feature_key = line[2:25].strip()
                    location_start = line[25:].strip()
                feature_lines = [location_start]
                line = self.handle.readline()
                while line[:self.FEATURE_QUALIFIER_INDENT] == self.FEATURE_QUALIFIER_SPACER or line.rstrip() == '':
                    assert line[:2] == 'FT'
                    feature_lines.append(line[self.FEATURE_QUALIFIER_INDENT:].strip())
                    line = self.handle.readline()
                (feature_key, location, qualifiers) = self.parse_feature(feature_key, feature_lines)
                if '>' in location:
                    location = bad_position_re.sub('>\\1', location)
                features.append((feature_key, location, qualifiers))
        self.line = line
        return features

class GenBankScanner(InsdcScanner):
    """For extracting chunks of information in GenBank files."""
    RECORD_START = 'LOCUS       '
    HEADER_WIDTH = 12
    FEATURE_START_MARKERS = ['FEATURES             Location/Qualifiers', 'FEATURES']
    FEATURE_END_MARKERS: List[str] = []
    FEATURE_QUALIFIER_INDENT = 21
    FEATURE_QUALIFIER_SPACER = ' ' * FEATURE_QUALIFIER_INDENT
    SEQUENCE_HEADERS = ['CONTIG', 'ORIGIN', 'BASE COUNT', 'WGS', 'TSA', 'TLS']
    GENBANK_INDENT = HEADER_WIDTH
    GENBANK_SPACER = ' ' * GENBANK_INDENT
    STRUCTURED_COMMENT_START = '-START##'
    STRUCTURED_COMMENT_END = '-END##'
    STRUCTURED_COMMENT_DELIM = ' :: '

    def parse_footer(self):
        if False:
            while True:
                i = 10
        'Return a tuple containing a list of any misc strings, and the sequence.'
        if self.line[:self.HEADER_WIDTH].rstrip() not in self.SEQUENCE_HEADERS:
            raise ValueError(f"Footer format unexpected:  '{self.line}'")
        misc_lines = []
        while self.line[:self.HEADER_WIDTH].rstrip() in self.SEQUENCE_HEADERS or self.line[:self.HEADER_WIDTH] == ' ' * self.HEADER_WIDTH or 'WGS' == self.line[:3]:
            misc_lines.append(self.line.rstrip())
            self.line = self.handle.readline()
            if not self.line:
                raise ValueError('Premature end of file')
        if self.line[:self.HEADER_WIDTH].rstrip() in self.SEQUENCE_HEADERS:
            raise ValueError(f"Eh? '{self.line}'")
        seq_lines = []
        line = self.line
        while True:
            if not line:
                warnings.warn('Premature end of file in sequence data', BiopythonParserWarning)
                line = '//'
                break
            line = line.rstrip()
            if not line:
                warnings.warn('Blank line in sequence data', BiopythonParserWarning)
                line = self.handle.readline()
                continue
            if line == '//':
                break
            if line.startswith('CONTIG'):
                break
            if len(line) > 9 and line[9:10] != ' ':
                warnings.warn('Invalid indentation for sequence line', BiopythonParserWarning)
                line = line[1:]
                if len(line) > 9 and line[9:10] != ' ':
                    raise ValueError(f"Sequence line mal-formed, '{line}'")
            seq_lines.append(line[10:])
            line = self.handle.readline()
        self.line = line
        return (misc_lines, ''.join(seq_lines).replace(' ', ''))

    def _feed_first_line(self, consumer, line):
        if False:
            print('Hello World!')
        'Scan over and parse GenBank LOCUS line (PRIVATE).\n\n        This must cope with several variants, primarily the old and new column\n        based standards from GenBank. Additionally EnsEMBL produces GenBank\n        files where the LOCUS line is space separated rather that following\n        the column based layout.\n\n        We also try to cope with GenBank like files with partial LOCUS lines.\n\n        As of release 229.0, the columns are no longer strictly in a given\n        position. See GenBank format release notes:\n\n            "Historically, the LOCUS line has had a fixed length and its\n            elements have been presented at specific column positions...\n            But with the anticipated increases in the lengths of accession\n            numbers, and the advent of sequences that are gigabases long,\n            maintaining the column positions will not always be possible and\n            the overall length of the LOCUS line could exceed 79 characters."\n\n        '
        if line[0:self.GENBANK_INDENT] != 'LOCUS       ':
            raise ValueError('LOCUS line does not start correctly:\n' + line)
        if line[29:33] in [' bp ', ' aa ', ' rc '] and line[55:62] == '       ':
            if line[41:42] != ' ':
                raise ValueError('LOCUS line does not contain space at position 42:\n' + line)
            if line[42:51].strip() not in ['', 'linear', 'circular']:
                raise ValueError('LOCUS line does not contain valid entry (linear, circular, ...):\n' + line)
            if line[51:52] != ' ':
                raise ValueError('LOCUS line does not contain space at position 52:\n' + line)
            if line[62:73].strip():
                if line[64:65] != '-':
                    raise ValueError('LOCUS line does not contain - at position 65 in date:\n' + line)
                if line[68:69] != '-':
                    raise ValueError('LOCUS line does not contain - at position 69 in date:\n' + line)
            name_and_length_str = line[self.GENBANK_INDENT:29]
            while '  ' in name_and_length_str:
                name_and_length_str = name_and_length_str.replace('  ', ' ')
            name_and_length = name_and_length_str.split(' ')
            if len(name_and_length) > 2:
                raise ValueError('Cannot parse the name and length in the LOCUS line:\n' + line)
            if len(name_and_length) == 1:
                raise ValueError('Name and length collide in the LOCUS line:\n' + line)
            (name, length) = name_and_length
            if len(name) > 16:
                warnings.warn('GenBank LOCUS line identifier over 16 characters', BiopythonParserWarning)
            consumer.locus(name)
            consumer.size(length)
            if line[33:51].strip() == '' and line[29:33] == ' aa ':
                consumer.residue_type('PROTEIN')
            else:
                consumer.residue_type(line[33:51].strip())
            consumer.molecule_type(line[33:41].strip())
            consumer.topology(line[42:51].strip())
            consumer.data_file_division(line[52:55])
            if line[62:73].strip():
                consumer.date(line[62:73])
        elif line[40:44] in [' bp ', ' aa ', ' rc '] and line[54:64].strip() in ['', 'linear', 'circular']:
            if len(line) < 79:
                warnings.warn(f'Truncated LOCUS line found - is this correct?\n:{line!r}', BiopythonParserWarning)
                padding_len = 79 - len(line)
                padding = ' ' * padding_len
                line += padding
            if line[40:44] not in [' bp ', ' aa ', ' rc ']:
                raise ValueError('LOCUS line does not contain size units at expected position:\n' + line)
            if line[44:47] not in ['   ', 'ss-', 'ds-', 'ms-']:
                raise ValueError('LOCUS line does not have valid strand type (Single stranded, ...):\n' + line)
            if not (line[47:54].strip() == '' or 'DNA' in line[47:54].strip().upper() or 'RNA' in line[47:54].strip().upper()):
                raise ValueError('LOCUS line does not contain valid sequence type (DNA, RNA, ...):\n' + line)
            if line[54:55] != ' ':
                raise ValueError('LOCUS line does not contain space at position 55:\n' + line)
            if line[55:63].strip() not in ['', 'linear', 'circular']:
                raise ValueError('LOCUS line does not contain valid entry (linear, circular, ...):\n' + line)
            if line[63:64] != ' ':
                raise ValueError('LOCUS line does not contain space at position 64:\n' + line)
            if line[67:68] != ' ':
                raise ValueError('LOCUS line does not contain space at position 68:\n' + line)
            if line[68:79].strip():
                if line[70:71] != '-':
                    raise ValueError('LOCUS line does not contain - at position 71 in date:\n' + line)
                if line[74:75] != '-':
                    raise ValueError('LOCUS line does not contain - at position 75 in date:\n' + line)
            name_and_length_str = line[self.GENBANK_INDENT:40]
            while '  ' in name_and_length_str:
                name_and_length_str = name_and_length_str.replace('  ', ' ')
            name_and_length = name_and_length_str.split(' ')
            if len(name_and_length) > 2:
                raise ValueError('Cannot parse the name and length in the LOCUS line:\n' + line)
            if len(name_and_length) == 1:
                raise ValueError('Name and length collide in the LOCUS line:\n' + line)
            consumer.locus(name_and_length[0])
            consumer.size(name_and_length[1])
            if line[44:54].strip() == '' and line[40:44] == ' aa ':
                consumer.residue_type(('PROTEIN ' + line[54:63]).strip())
            else:
                consumer.residue_type(line[44:63].strip())
            consumer.molecule_type(line[44:54].strip())
            consumer.topology(line[55:63].strip())
            if line[64:76].strip():
                consumer.data_file_division(line[64:67])
            if line[68:79].strip():
                consumer.date(line[68:79])
        elif line[self.GENBANK_INDENT:].strip().count(' ') == 0:
            if line[self.GENBANK_INDENT:].strip() != '':
                consumer.locus(line[self.GENBANK_INDENT:].strip())
            else:
                warnings.warn(f'Minimal LOCUS line found - is this correct?\n:{line!r}', BiopythonParserWarning)
        elif len(line.split()) == 8 and line.split()[3] in ('aa', 'bp') and (line.split()[5] in ('linear', 'circular')):
            splitline = line.split()
            consumer.locus(splitline[1])
            if int(splitline[2]) > sys.maxsize:
                raise ValueError('Tried to load a sequence with a length %s, your installation of python can only load sesquences of length %s' % (splitline[2], sys.maxsize))
            else:
                consumer.size(splitline[2])
            consumer.residue_type(splitline[4])
            consumer.topology(splitline[5])
            consumer.data_file_division(splitline[6])
            consumer.date(splitline[7])
            if len(line) < 80:
                warnings.warn('Attempting to parse malformed locus line:\n%r\nFound locus %r size %r residue_type %r\nSome fields may be wrong.' % (line, splitline[1], splitline[2], splitline[4]), BiopythonParserWarning)
        elif len(line.split()) == 7 and line.split()[3] in ['aa', 'bp']:
            splitline = line.split()
            consumer.locus(splitline[1])
            consumer.size(splitline[2])
            consumer.residue_type(splitline[4])
            consumer.data_file_division(splitline[5])
            consumer.date(splitline[6])
        elif len(line.split()) >= 4 and line.split()[3] in ['aa', 'bp']:
            warnings.warn(f'Malformed LOCUS line found - is this correct?\n:{line!r}', BiopythonParserWarning)
            consumer.locus(line.split()[1])
            consumer.size(line.split()[2])
        elif len(line.split()) >= 4 and line.split()[-1] in ['aa', 'bp']:
            warnings.warn(f'Malformed LOCUS line found - is this correct?\n:{line!r}', BiopythonParserWarning)
            consumer.locus(line[5:].rsplit(None, 2)[0].strip())
            consumer.size(line.split()[-2])
        else:
            raise ValueError('Did not recognise the LOCUS line layout:\n' + line)

    def _feed_header_lines(self, consumer, lines):
        if False:
            print('Hello World!')
        consumer_dict = {'DEFINITION': 'definition', 'ACCESSION': 'accession', 'NID': 'nid', 'PID': 'pid', 'DBSOURCE': 'db_source', 'KEYWORDS': 'keywords', 'SEGMENT': 'segment', 'SOURCE': 'source', 'AUTHORS': 'authors', 'CONSRTM': 'consrtm', 'PROJECT': 'project', 'TITLE': 'title', 'JOURNAL': 'journal', 'MEDLINE': 'medline_id', 'PUBMED': 'pubmed_id', 'REMARK': 'remark'}
        lines = [_f for _f in lines if _f]
        lines.append('')
        line_iter = iter(lines)
        try:
            line = next(line_iter)
            while True:
                if not line:
                    break
                line_type = line[:self.GENBANK_INDENT].strip()
                data = line[self.GENBANK_INDENT:].strip()
                if line_type == 'VERSION':
                    while '  ' in data:
                        data = data.replace('  ', ' ')
                    if ' GI:' not in data:
                        consumer.version(data)
                    else:
                        if self.debug:
                            print('Version [' + data.split(' GI:')[0] + '], gi [' + data.split(' GI:')[1] + ']')
                        consumer.version(data.split(' GI:')[0])
                        consumer.gi(data.split(' GI:')[1])
                    line = next(line_iter)
                elif line_type == 'DBLINK':
                    consumer.dblink(data.strip())
                    while True:
                        line = next(line_iter)
                        if line[:self.GENBANK_INDENT] == self.GENBANK_SPACER:
                            consumer.dblink(line[self.GENBANK_INDENT:].strip())
                        else:
                            break
                elif line_type == 'REFERENCE':
                    if self.debug > 1:
                        print('Found reference [' + data + ']')
                    data = data.strip()
                    while True:
                        line = next(line_iter)
                        if line[:self.GENBANK_INDENT] == self.GENBANK_SPACER:
                            data += ' ' + line[self.GENBANK_INDENT:]
                            if self.debug > 1:
                                print('Extended reference text [' + data + ']')
                        else:
                            break
                    while '  ' in data:
                        data = data.replace('  ', ' ')
                    if ' ' not in data:
                        if self.debug > 2:
                            print('Reference number "' + data + '"')
                        consumer.reference_num(data)
                    else:
                        if self.debug > 2:
                            print('Reference number "' + data[:data.find(' ')] + '", "' + data[data.find(' ') + 1:] + '"')
                        consumer.reference_num(data[:data.find(' ')])
                        consumer.reference_bases(data[data.find(' ') + 1:])
                elif line_type == 'ORGANISM':
                    organism_data = data
                    lineage_data = ''
                    while True:
                        line = next(line_iter)
                        if line[0:self.GENBANK_INDENT] == self.GENBANK_SPACER:
                            if lineage_data or ';' in line or line[self.GENBANK_INDENT:].strip() in ('Bacteria.', 'Archaea.', 'Eukaryota.', 'Unclassified.', 'Viruses.', 'cellular organisms.', 'other sequences.', 'unclassified sequences.'):
                                lineage_data += ' ' + line[self.GENBANK_INDENT:]
                            elif line[self.GENBANK_INDENT:].strip() == '.':
                                pass
                            else:
                                organism_data += ' ' + line[self.GENBANK_INDENT:].strip()
                        else:
                            break
                    consumer.organism(organism_data)
                    if lineage_data.strip() == '' and self.debug > 1:
                        print('Taxonomy line(s) missing or blank')
                    consumer.taxonomy(lineage_data.strip())
                    del organism_data, lineage_data
                elif line_type == 'COMMENT':
                    data = line[self.GENBANK_INDENT:]
                    if self.debug > 1:
                        print('Found comment')
                    comment_list = []
                    structured_comment_dict = defaultdict(dict)
                    regex = f'([^#]+){self.STRUCTURED_COMMENT_START}$'
                    structured_comment_key = re.search(regex, data)
                    if structured_comment_key is not None:
                        structured_comment_key = structured_comment_key.group(1)
                        if self.debug > 1:
                            print('Found Structured Comment')
                    else:
                        comment_list.append(data)
                    while True:
                        line = next(line_iter)
                        data = line[self.GENBANK_INDENT:]
                        if line[0:self.GENBANK_INDENT] == self.GENBANK_SPACER:
                            if self.STRUCTURED_COMMENT_START in data:
                                regex = f'([^#]+){self.STRUCTURED_COMMENT_START}$'
                                structured_comment_key = re.search(regex, data)
                                if structured_comment_key is not None:
                                    structured_comment_key = structured_comment_key.group(1)
                                else:
                                    comment_list.append(data)
                            elif structured_comment_key is not None and self.STRUCTURED_COMMENT_DELIM in data:
                                match = re.search(f'(.+?)\\s*{self.STRUCTURED_COMMENT_DELIM}\\s*(.+)', data)
                                structured_comment_dict[structured_comment_key][match.group(1)] = match.group(2)
                                if self.debug > 2:
                                    print('Structured Comment continuation [' + data + ']')
                            elif structured_comment_key is not None and self.STRUCTURED_COMMENT_END not in data:
                                if structured_comment_key not in structured_comment_dict:
                                    warnings.warn('Structured comment not parsed for %s. Is it malformed?' % consumer.data.name, BiopythonParserWarning)
                                    continue
                                previous_value_line = structured_comment_dict[structured_comment_key][match.group(1)]
                                structured_comment_dict[structured_comment_key][match.group(1)] = previous_value_line + ' ' + line.strip()
                            elif self.STRUCTURED_COMMENT_END in data:
                                structured_comment_key = None
                            else:
                                comment_list.append(data)
                                if self.debug > 2:
                                    print('Comment continuation [' + data + ']')
                        else:
                            break
                    if comment_list:
                        consumer.comment(comment_list)
                    if structured_comment_dict:
                        consumer.structured_comment(structured_comment_dict)
                    del comment_list, structured_comment_key, structured_comment_dict
                elif line_type in consumer_dict:
                    while True:
                        line = next(line_iter)
                        if line[0:self.GENBANK_INDENT] == self.GENBANK_SPACER:
                            data += ' ' + line[self.GENBANK_INDENT:]
                        else:
                            if line_type == 'DEFINITION' and data.endswith('.'):
                                data = data[:-1]
                            getattr(consumer, consumer_dict[line_type])(data)
                            break
                else:
                    if self.debug:
                        print('Ignoring GenBank header line:\n' % line)
                    line = next(line_iter)
        except StopIteration:
            raise ValueError('Problem in header') from None

    def _feed_misc_lines(self, consumer, lines):
        if False:
            return 10
        lines.append('')
        line_iter = iter(lines)
        try:
            for line in line_iter:
                if line.startswith('BASE COUNT'):
                    line = line[10:].strip()
                    if line:
                        if self.debug:
                            print('base_count = ' + line)
                        consumer.base_count(line)
                if line.startswith('ORIGIN'):
                    line = line[6:].strip()
                    if line:
                        if self.debug:
                            print('origin_name = ' + line)
                        consumer.origin_name(line)
                if line.startswith('TLS '):
                    line = line[3:].strip()
                    consumer.tls(line)
                if line.startswith('TSA '):
                    line = line[3:].strip()
                    consumer.tsa(line)
                if line.startswith('WGS '):
                    line = line[3:].strip()
                    consumer.wgs(line)
                if line.startswith('WGS_SCAFLD'):
                    line = line[10:].strip()
                    consumer.add_wgs_scafld(line)
                if line.startswith('CONTIG'):
                    line = line[6:].strip()
                    contig_location = line
                    while True:
                        line = next(line_iter)
                        if not line:
                            break
                        elif line[:self.GENBANK_INDENT] == self.GENBANK_SPACER:
                            contig_location += line[self.GENBANK_INDENT:].rstrip()
                        elif line.startswith('ORIGIN'):
                            line = line[6:].strip()
                            if line:
                                consumer.origin_name(line)
                            break
                        else:
                            raise ValueError('Expected CONTIG continuation line, got:\n' + line)
                    consumer.contig_location(contig_location)
            return
        except StopIteration:
            raise ValueError('Problem in misc lines before sequence') from None