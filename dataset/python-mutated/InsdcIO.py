"""Bio.SeqIO support for the "genbank" and "embl" file formats.

You are expected to use this module via the Bio.SeqIO functions.
Note that internally this module calls Bio.GenBank to do the actual
parsing of GenBank, EMBL and IMGT files.

See Also:
International Nucleotide Sequence Database Collaboration
http://www.insdc.org/

GenBank
http://www.ncbi.nlm.nih.gov/Genbank/

EMBL Nucleotide Sequence Database
http://www.ebi.ac.uk/embl/

DDBJ (DNA Data Bank of Japan)
http://www.ddbj.nig.ac.jp/

IMGT (use a variant of EMBL format with longer feature indents)
http://imgt.cines.fr/download/LIGM-DB/userman_doc.html
http://imgt.cines.fr/download/LIGM-DB/ftable_doc.html
http://www.ebi.ac.uk/imgt/hla/docs/manual.html

"""
import warnings
from datetime import datetime
from Bio import BiopythonWarning
from Bio import SeqFeature
from Bio import SeqIO
from Bio.GenBank.Scanner import _ImgtScanner
from Bio.GenBank.Scanner import EmblScanner
from Bio.GenBank.Scanner import GenBankScanner
from Bio.Seq import UndefinedSequenceError
from .Interfaces import _get_seq_string
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter

class GenBankIterator(SequenceIterator):
    """Parser for GenBank files."""

    def __init__(self, source):
        if False:
            for i in range(10):
                print('nop')
        'Break up a Genbank file into SeqRecord objects.\n\n        Argument source is a file-like object opened in text mode or a path to a file.\n        Every section from the LOCUS line to the terminating // becomes\n        a single SeqRecord with associated annotation and features.\n\n        Note that for genomes or chromosomes, there is typically only\n        one record.\n\n        This gets called internally by Bio.SeqIO for the GenBank file format:\n\n        >>> from Bio import SeqIO\n        >>> for record in SeqIO.parse("GenBank/cor6_6.gb", "gb"):\n        ...     print(record.id)\n        ...\n        X55053.1\n        X62281.1\n        M81224.1\n        AJ237582.1\n        L31939.1\n        AF297471.1\n\n        Equivalently,\n\n        >>> with open("GenBank/cor6_6.gb") as handle:\n        ...     for record in GenBankIterator(handle):\n        ...         print(record.id)\n        ...\n        X55053.1\n        X62281.1\n        M81224.1\n        AJ237582.1\n        L31939.1\n        AF297471.1\n\n        '
        super().__init__(source, mode='t', fmt='GenBank')

    def parse(self, handle):
        if False:
            print('Hello World!')
        'Start parsing the file, and return a SeqRecord generator.'
        records = GenBankScanner(debug=0).parse_records(handle)
        return records

class EmblIterator(SequenceIterator):
    """Parser for EMBL files."""

    def __init__(self, source):
        if False:
            for i in range(10):
                print('nop')
        'Break up an EMBL file into SeqRecord objects.\n\n        Argument source is a file-like object opened in text mode or a path to a file.\n        Every section from the LOCUS line to the terminating // becomes\n        a single SeqRecord with associated annotation and features.\n\n        Note that for genomes or chromosomes, there is typically only\n        one record.\n\n        This gets called internally by Bio.SeqIO for the EMBL file format:\n\n        >>> from Bio import SeqIO\n        >>> for record in SeqIO.parse("EMBL/epo_prt_selection.embl", "embl"):\n        ...     print(record.id)\n        ...\n        A00022.1\n        A00028.1\n        A00031.1\n        A00034.1\n        A00060.1\n        A00071.1\n        A00072.1\n        A00078.1\n        CQ797900.1\n\n        Equivalently,\n\n        >>> with open("EMBL/epo_prt_selection.embl") as handle:\n        ...     for record in EmblIterator(handle):\n        ...         print(record.id)\n        ...\n        A00022.1\n        A00028.1\n        A00031.1\n        A00034.1\n        A00060.1\n        A00071.1\n        A00072.1\n        A00078.1\n        CQ797900.1\n\n        '
        super().__init__(source, mode='t', fmt='EMBL')

    def parse(self, handle):
        if False:
            for i in range(10):
                print('nop')
        'Start parsing the file, and return a SeqRecord generator.'
        records = EmblScanner(debug=0).parse_records(handle)
        return records

class ImgtIterator(SequenceIterator):
    """Parser for IMGT files."""

    def __init__(self, source):
        if False:
            return 10
        'Break up an IMGT file into SeqRecord objects.\n\n        Argument source is a file-like object opened in text mode or a path to a file.\n        Every section from the LOCUS line to the terminating // becomes\n        a single SeqRecord with associated annotation and features.\n\n        Note that for genomes or chromosomes, there is typically only\n        one record.\n        '
        super().__init__(source, mode='t', fmt='IMGT')

    def parse(self, handle):
        if False:
            while True:
                i = 10
        'Start parsing the file, and return a SeqRecord generator.'
        records = _ImgtScanner(debug=0).parse_records(handle)
        return records

class GenBankCdsFeatureIterator(SequenceIterator):
    """Parser for GenBank files, creating a SeqRecord for each CDS feature."""

    def __init__(self, source):
        if False:
            print('Hello World!')
        'Break up a Genbank file into SeqRecord objects for each CDS feature.\n\n        Argument source is a file-like object opened in text mode or a path to a file.\n\n        Every section from the LOCUS line to the terminating // can contain\n        many CDS features.  These are returned as with the stated amino acid\n        translation sequence (if given).\n        '
        super().__init__(source, mode='t', fmt='GenBank')

    def parse(self, handle):
        if False:
            i = 10
            return i + 15
        'Start parsing the file, and return a SeqRecord generator.'
        return GenBankScanner(debug=0).parse_cds_features(handle)

class EmblCdsFeatureIterator(SequenceIterator):
    """Parser for EMBL files, creating a SeqRecord for each CDS feature."""

    def __init__(self, source):
        if False:
            for i in range(10):
                print('nop')
        'Break up a EMBL file into SeqRecord objects for each CDS feature.\n\n        Argument source is a file-like object opened in text mode or a path to a file.\n\n        Every section from the LOCUS line to the terminating // can contain\n        many CDS features.  These are returned as with the stated amino acid\n        translation sequence (if given).\n        '
        super().__init__(source, mode='t', fmt='EMBL')

    def parse(self, handle):
        if False:
            while True:
                i = 10
        'Start parsing the file, and return a SeqRecord generator.'
        return EmblScanner(debug=0).parse_cds_features(handle)

def _insdc_feature_position_string(pos, offset=0):
    if False:
        return 10
    'Build a GenBank/EMBL position string (PRIVATE).\n\n    Use offset=1 to add one to convert a start position from python counting.\n    '
    if isinstance(pos, SeqFeature.ExactPosition):
        return '%i' % (pos + offset)
    elif isinstance(pos, SeqFeature.WithinPosition):
        return '(%i.%i)' % (pos._left + offset, pos._right + offset)
    elif isinstance(pos, SeqFeature.BetweenPosition):
        return '(%i^%i)' % (pos._left + offset, pos._right + offset)
    elif isinstance(pos, SeqFeature.BeforePosition):
        return '<%i' % (pos + offset)
    elif isinstance(pos, SeqFeature.AfterPosition):
        return '>%i' % (pos + offset)
    elif isinstance(pos, SeqFeature.OneOfPosition):
        return 'one-of(%s)' % ','.join((_insdc_feature_position_string(p, offset) for p in pos.position_choices))
    elif isinstance(pos, SeqFeature.Position):
        raise NotImplementedError('Please report this as a bug in Biopython.')
    else:
        raise ValueError('Expected a SeqFeature position object.')

def _insdc_location_string_ignoring_strand_and_subfeatures(location, rec_length):
    if False:
        return 10
    if location.ref:
        ref = f'{location.ref}:'
    else:
        ref = ''
    assert not location.ref_db
    if isinstance(location.start, SeqFeature.ExactPosition) and isinstance(location.end, SeqFeature.ExactPosition) and (location.start == location.end):
        if location.end == rec_length:
            return '%s%i^1' % (ref, rec_length)
        else:
            return '%s%i^%i' % (ref, location.end, location.end + 1)
    if isinstance(location.start, SeqFeature.ExactPosition) and isinstance(location.end, SeqFeature.ExactPosition) and (location.start + 1 == location.end):
        return '%s%i' % (ref, location.end)
    elif isinstance(location.start, SeqFeature.UnknownPosition) or isinstance(location.end, SeqFeature.UnknownPosition):
        if isinstance(location.start, SeqFeature.UnknownPosition) and isinstance(location.end, SeqFeature.UnknownPosition):
            raise ValueError('Feature with unknown location')
        elif isinstance(location.start, SeqFeature.UnknownPosition):
            return '%s<%i..%s' % (ref, location.end, _insdc_feature_position_string(location.end))
        else:
            return '%s%s..>%i' % (ref, _insdc_feature_position_string(location.start, +1), location.start + 1)
    else:
        return ref + _insdc_feature_position_string(location.start, +1) + '..' + _insdc_feature_position_string(location.end)

def _insdc_location_string(location, rec_length):
    if False:
        for i in range(10):
            print('nop')
    'Build a GenBank/EMBL location from a (Compound) SimpleLocation (PRIVATE).\n\n    There is a choice of how to show joins on the reverse complement strand,\n    GenBank used "complement(join(1,10),(20,100))" while EMBL used to use\n    "join(complement(20,100),complement(1,10))" instead (but appears to have\n    now adopted the GenBank convention). Notice that the order of the entries\n    is reversed! This function therefore uses the first form. In this situation\n    we expect the CompoundLocation and its parts to all be marked as\n    strand == -1, and to be in the order 19:100 then 0:10.\n    '
    try:
        parts = location.parts
        if location.strand == -1:
            return 'complement(%s(%s))' % (location.operator, ','.join((_insdc_location_string_ignoring_strand_and_subfeatures(p, rec_length) for p in parts[::-1])))
        else:
            return '%s(%s)' % (location.operator, ','.join((_insdc_location_string(p, rec_length) for p in parts)))
    except AttributeError:
        loc = _insdc_location_string_ignoring_strand_and_subfeatures(location, rec_length)
        if location.strand == -1:
            return f'complement({loc})'
        else:
            return loc

class _InsdcWriter(SequenceWriter):
    """Base class for GenBank and EMBL writers (PRIVATE)."""
    MAX_WIDTH = 80
    QUALIFIER_INDENT = 21
    QUALIFIER_INDENT_STR = ' ' * QUALIFIER_INDENT
    QUALIFIER_INDENT_TMP = '     %s                '
    FTQUAL_NO_QUOTE = ('anticodon', 'citation', 'codon_start', 'compare', 'direction', 'estimated_length', 'mod_base', 'number', 'rpt_type', 'rpt_unit_range', 'tag_peptide', 'transl_except', 'transl_table')

    def _write_feature_qualifier(self, key, value=None, quote=None):
        if False:
            for i in range(10):
                print('nop')
        if value is None:
            self.handle.write(f'{self.QUALIFIER_INDENT_STR}/{key}\n')
            return
        if isinstance(value, str):
            value = value.replace('"', '""')
        if quote is None:
            if isinstance(value, int) or key in self.FTQUAL_NO_QUOTE:
                quote = False
            else:
                quote = True
        if quote:
            line = f'{self.QUALIFIER_INDENT_STR}/{key}="{value}"'
        else:
            line = f'{self.QUALIFIER_INDENT_STR}/{key}={value}'
        if len(line) <= self.MAX_WIDTH:
            self.handle.write(line + '\n')
            return
        while line.lstrip():
            if len(line) <= self.MAX_WIDTH:
                self.handle.write(line + '\n')
                return
            for index in range(min(len(line) - 1, self.MAX_WIDTH), self.QUALIFIER_INDENT + 1, -1):
                if line[index] == ' ':
                    break
            if line[index] != ' ':
                index = self.MAX_WIDTH
            assert index <= self.MAX_WIDTH
            self.handle.write(line[:index] + '\n')
            line = self.QUALIFIER_INDENT_STR + line[index:].lstrip()

    def _wrap_location(self, location):
        if False:
            return 10
        'Split a feature location into lines (break at commas) (PRIVATE).'
        length = self.MAX_WIDTH - self.QUALIFIER_INDENT
        if len(location) <= length:
            return location
        index = location[:length].rfind(',')
        if index == -1:
            warnings.warn(f"Couldn't split location:\n{location}", BiopythonWarning)
            return location
        return location[:index + 1] + '\n' + self.QUALIFIER_INDENT_STR + self._wrap_location(location[index + 1:])

    def _write_feature(self, feature, record_length):
        if False:
            for i in range(10):
                print('nop')
        'Write a single SeqFeature object to features table (PRIVATE).'
        assert feature.type, feature
        location = _insdc_location_string(feature.location, record_length)
        f_type = feature.type.replace(' ', '_')
        line = (self.QUALIFIER_INDENT_TMP % f_type)[:self.QUALIFIER_INDENT] + self._wrap_location(location) + '\n'
        self.handle.write(line)
        for (key, values) in feature.qualifiers.items():
            if isinstance(values, (list, tuple)):
                for value in values:
                    self._write_feature_qualifier(key, value)
            else:
                self._write_feature_qualifier(key, values)

    @staticmethod
    def _get_annotation_str(record, key, default='.', just_first=False):
        if False:
            for i in range(10):
                print('nop')
        'Get an annotation dictionary entry (as a string) (PRIVATE).\n\n        Some entries are lists, in which case if just_first=True the first entry\n        is returned.  If just_first=False (default) this verifies there is only\n        one entry before returning it.\n        '
        try:
            answer = record.annotations[key]
        except KeyError:
            return default
        if isinstance(answer, list):
            if not just_first:
                assert len(answer) == 1
            return str(answer[0])
        else:
            return str(answer)

    @staticmethod
    def _split_multi_line(text, max_len):
        if False:
            i = 10
            return i + 15
        'Return a list of strings (PRIVATE).\n\n        Any single words which are too long get returned as a whole line\n        (e.g. URLs) without an exception or warning.\n        '
        text = text.strip()
        if len(text) <= max_len:
            return [text]
        words = text.split()
        text = ''
        while words and len(text) + 1 + len(words[0]) <= max_len:
            text += ' ' + words.pop(0)
            text = text.strip()
        answer = [text]
        while words:
            text = words.pop(0)
            while words and len(text) + 1 + len(words[0]) <= max_len:
                text += ' ' + words.pop(0)
                text = text.strip()
            answer.append(text)
        assert not words
        return answer

    def _split_contig(self, record, max_len):
        if False:
            i = 10
            return i + 15
        'Return a list of strings, splits on commas (PRIVATE).'
        contig = record.annotations.get('contig', '')
        if isinstance(contig, (list, tuple)):
            contig = ''.join(contig)
        contig = self.clean(contig)
        answer = []
        while contig:
            if len(contig) > max_len:
                pos = contig[:max_len - 1].rfind(',')
                if pos == -1:
                    raise ValueError('Could not break up CONTIG')
                (text, contig) = (contig[:pos + 1], contig[pos + 1:])
            else:
                (text, contig) = (contig, '')
            answer.append(text)
        return answer

class GenBankWriter(_InsdcWriter):
    """GenBank writer."""
    HEADER_WIDTH = 12
    QUALIFIER_INDENT = 21
    STRUCTURED_COMMENT_START = '-START##'
    STRUCTURED_COMMENT_END = '-END##'
    STRUCTURED_COMMENT_DELIM = ' :: '
    LETTERS_PER_LINE = 60
    SEQUENCE_INDENT = 9

    def _write_single_line(self, tag, text):
        if False:
            while True:
                i = 10
        "Write single line in each GenBank record (PRIVATE).\n\n        Used in the 'header' of each GenBank record.\n        "
        assert len(tag) < self.HEADER_WIDTH
        if len(text) > self.MAX_WIDTH - self.HEADER_WIDTH:
            if tag:
                warnings.warn(f'Annotation {text!r} too long for {tag!r} line', BiopythonWarning)
            else:
                warnings.warn(f'Annotation {text!r} too long', BiopythonWarning)
        self.handle.write('%s%s\n' % (tag.ljust(self.HEADER_WIDTH), text.replace('\n', ' ')))

    def _write_multi_line(self, tag, text):
        if False:
            for i in range(10):
                print('nop')
        "Write multiple lines in each GenBank record (PRIVATE).\n\n        Used in the 'header' of each GenBank record.\n        "
        max_len = self.MAX_WIDTH - self.HEADER_WIDTH
        lines = self._split_multi_line(text, max_len)
        self._write_single_line(tag, lines[0])
        for line in lines[1:]:
            self._write_single_line('', line)

    def _write_multi_entries(self, tag, text_list):
        if False:
            print('Hello World!')
        for (i, text) in enumerate(text_list):
            if i == 0:
                self._write_single_line(tag, text)
            else:
                self._write_single_line('', text)

    @staticmethod
    def _get_date(record):
        if False:
            return 10
        default = '01-JAN-1980'
        try:
            date = record.annotations['date']
        except KeyError:
            return default
        if isinstance(date, list) and len(date) == 1:
            date = date[0]
        if isinstance(date, datetime):
            date = date.strftime('%d-%b-%Y').upper()
        months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        if not isinstance(date, str) or len(date) != 11:
            return default
        try:
            datetime(int(date[-4:]), months.index(date[3:6]) + 1, int(date[0:2]))
        except ValueError:
            date = default
        return date

    @staticmethod
    def _get_data_division(record):
        if False:
            while True:
                i = 10
        try:
            division = record.annotations['data_file_division']
        except KeyError:
            division = 'UNK'
        if division in ['PRI', 'ROD', 'MAM', 'VRT', 'INV', 'PLN', 'BCT', 'VRL', 'PHG', 'SYN', 'UNA', 'EST', 'PAT', 'STS', 'GSS', 'HTG', 'HTC', 'ENV', 'CON', 'TSA']:
            pass
        else:
            embl_to_gbk = {'FUN': 'PLN', 'HUM': 'PRI', 'MUS': 'ROD', 'PRO': 'BCT', 'UNC': 'UNK', 'XXX': 'UNK'}
            try:
                division = embl_to_gbk[division]
            except KeyError:
                division = 'UNK'
        assert len(division) == 3
        return division

    def _get_topology(self, record):
        if False:
            for i in range(10):
                print('nop')
        "Set the topology to 'circular', 'linear' if defined (PRIVATE)."
        max_topology_len = len('circular')
        topology = self._get_annotation_str(record, 'topology', default='')
        if topology and len(topology) <= max_topology_len:
            return topology.ljust(max_topology_len)
        else:
            return ' ' * max_topology_len

    def _write_the_first_line(self, record):
        if False:
            return 10
        'Write the LOCUS line (PRIVATE).'
        locus = record.name
        if not locus or locus == '<unknown name>':
            locus = record.id
        if not locus or locus == '<unknown id>':
            locus = self._get_annotation_str(record, 'accession', just_first=True)
        if len(locus) > 16:
            if len(locus) + 1 + len(str(len(record))) > 28:
                warnings.warn('Increasing length of locus line to allow long name. This will result in fields that are not in usual positions.', BiopythonWarning)
        if len(locus.split()) > 1:
            raise ValueError(f'Invalid whitespace in {locus!r} for LOCUS line')
        if len(record) > 99999999999:
            warnings.warn('The sequence length is very long. The LOCUS line will be increased in length to compensate. This may cause unexpected behavior.', BiopythonWarning)
        mol_type = self._get_annotation_str(record, 'molecule_type', None)
        if mol_type is None:
            raise ValueError('missing molecule_type in annotations')
        if mol_type and len(mol_type) > 7:
            mol_type = mol_type.replace('unassigned ', '').replace('genomic ', '')
            if len(mol_type) > 7:
                warnings.warn(f'Molecule type {mol_type!r} too long', BiopythonWarning)
                mol_type = 'DNA'
        if mol_type in ['protein', 'PROTEIN']:
            mol_type = ''
        if mol_type == '':
            units = 'aa'
        else:
            units = 'bp'
        topology = self._get_topology(record)
        division = self._get_data_division(record)
        if len(locus) > 16 and len(str(len(record))) > 11 - (len(locus) - 16):
            name_length = locus + ' ' + str(len(record))
        else:
            name_length = str(len(record)).rjust(28)
            name_length = locus + name_length[len(locus):]
            assert len(name_length) == 28, name_length
            assert ' ' in name_length, name_length
        assert len(units) == 2
        assert len(division) == 3
        line = 'LOCUS       %s %s    %s %s %s %s\n' % (name_length, units, mol_type.ljust(7), topology, division, self._get_date(record))
        if len(line) > 80:
            splitline = line.split()
            if splitline[3] not in ['bp', 'aa']:
                raise ValueError('LOCUS line does not contain size units at expected position:\n' + line)
            if not (splitline[3].strip() == 'aa' or 'DNA' in splitline[4].strip().upper() or 'RNA' in splitline[4].strip().upper()):
                raise ValueError('LOCUS line does not contain valid sequence type (DNA, RNA, ...):\n' + line)
            self.handle.write(line)
        else:
            assert len(line) == 79 + 1, repr(line)
            assert line[12:40].split() == [locus, str(len(record))], line
            if line[40:44] not in [' bp ', ' aa ']:
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
            if line[70:71] != '-':
                raise ValueError('LOCUS line does not contain - at position 71 in date:\n' + line)
            if line[74:75] != '-':
                raise ValueError('LOCUS line does not contain - at position 75 in date:\n' + line)
            self.handle.write(line)

    def _write_references(self, record):
        if False:
            while True:
                i = 10
        number = 0
        for ref in record.annotations['references']:
            if not isinstance(ref, SeqFeature.Reference):
                continue
            number += 1
            data = str(number)
            if ref.location and len(ref.location) == 1:
                molecule_type = record.annotations.get('molecule_type')
                if molecule_type and 'protein' in molecule_type:
                    units = 'residues'
                else:
                    units = 'bases'
                data += '  (%s %i to %i)' % (units, ref.location[0].start + 1, ref.location[0].end)
            self._write_single_line('REFERENCE', data)
            if ref.authors:
                self._write_multi_line('  AUTHORS', ref.authors)
            if ref.consrtm:
                self._write_multi_line('  CONSRTM', ref.consrtm)
            if ref.title:
                self._write_multi_line('  TITLE', ref.title)
            if ref.journal:
                self._write_multi_line('  JOURNAL', ref.journal)
            if ref.medline_id:
                self._write_multi_line('  MEDLINE', ref.medline_id)
            if ref.pubmed_id:
                self._write_multi_line('   PUBMED', ref.pubmed_id)
            if ref.comment:
                self._write_multi_line('  REMARK', ref.comment)

    def _write_comment(self, record):
        if False:
            print('Hello World!')
        lines = []
        if 'structured_comment' in record.annotations:
            comment = record.annotations['structured_comment']
            padding = 0
            for (key, data) in comment.items():
                for (subkey, subdata) in data.items():
                    padding = len(subkey) if len(subkey) > padding else padding
            for (key, data) in comment.items():
                lines.append(f'##{key}{self.STRUCTURED_COMMENT_START}')
                for (subkey, subdata) in data.items():
                    spaces = ' ' * (padding - len(subkey))
                    lines.append(f'{subkey}{spaces}{self.STRUCTURED_COMMENT_DELIM}{subdata}')
                lines.append(f'##{key}{self.STRUCTURED_COMMENT_END}')
        if 'comment' in record.annotations:
            comment = record.annotations['comment']
            if isinstance(comment, str):
                lines += comment.split('\n')
            elif isinstance(comment, (list, tuple)):
                lines += list(comment)
            else:
                raise ValueError('Could not understand comment annotation')
        self._write_multi_line('COMMENT', lines[0])
        for line in lines[1:]:
            self._write_multi_line('', line)

    def _write_contig(self, record):
        if False:
            i = 10
            return i + 15
        max_len = self.MAX_WIDTH - self.HEADER_WIDTH
        lines = self._split_contig(record, max_len)
        self._write_single_line('CONTIG', lines[0])
        for text in lines[1:]:
            self._write_single_line('', text)

    def _write_sequence(self, record):
        if False:
            for i in range(10):
                print('nop')
        try:
            data = _get_seq_string(record)
        except UndefinedSequenceError:
            if 'contig' in record.annotations:
                self._write_contig(record)
            else:
                self.handle.write('ORIGIN\n')
            return
        data = data.lower()
        seq_len = len(data)
        self.handle.write('ORIGIN\n')
        for line_number in range(0, seq_len, self.LETTERS_PER_LINE):
            self.handle.write(str(line_number + 1).rjust(self.SEQUENCE_INDENT))
            for words in range(line_number, min(line_number + self.LETTERS_PER_LINE, seq_len), 10):
                self.handle.write(f' {data[words:words + 10]}')
            self.handle.write('\n')

    def write_record(self, record):
        if False:
            i = 10
            return i + 15
        'Write a single record to the output file.'
        handle = self.handle
        self._write_the_first_line(record)
        default = record.id
        if default.count('.') == 1 and default[default.index('.') + 1:].isdigit():
            default = record.id.split('.', 1)[0]
        accession = self._get_annotation_str(record, 'accession', default, just_first=True)
        acc_with_version = accession
        if record.id.startswith(accession + '.'):
            try:
                acc_with_version = '%s.%i' % (accession, int(record.id.split('.', 1)[1]))
            except ValueError:
                pass
        gi = self._get_annotation_str(record, 'gi', just_first=True)
        descr = record.description
        if descr == '<unknown description>':
            descr = ''
        descr += '.'
        self._write_multi_line('DEFINITION', descr)
        self._write_single_line('ACCESSION', accession)
        if gi != '.':
            self._write_single_line('VERSION', f'{acc_with_version}  GI:{gi}')
        else:
            self._write_single_line('VERSION', acc_with_version)
        dbxrefs_with_space = []
        for x in record.dbxrefs:
            if ': ' not in x:
                x = x.replace(':', ': ')
            dbxrefs_with_space.append(x)
        self._write_multi_entries('DBLINK', dbxrefs_with_space)
        del dbxrefs_with_space
        try:
            keywords = '; '.join(record.annotations['keywords'])
            if not keywords.endswith('.'):
                keywords += '.'
        except KeyError:
            keywords = '.'
        self._write_multi_line('KEYWORDS', keywords)
        if 'segment' in record.annotations:
            segment = record.annotations['segment']
            if isinstance(segment, list):
                assert len(segment) == 1, segment
                segment = segment[0]
            self._write_single_line('SEGMENT', segment)
        self._write_multi_line('SOURCE', self._get_annotation_str(record, 'source'))
        org = self._get_annotation_str(record, 'organism')
        if len(org) > self.MAX_WIDTH - self.HEADER_WIDTH:
            org = org[:self.MAX_WIDTH - self.HEADER_WIDTH - 4] + '...'
        self._write_single_line('  ORGANISM', org)
        try:
            taxonomy = '; '.join(record.annotations['taxonomy'])
            if not taxonomy.endswith('.'):
                taxonomy += '.'
        except KeyError:
            taxonomy = '.'
        self._write_multi_line('', taxonomy)
        if 'db_source' in record.annotations:
            db_source = record.annotations['db_source']
            if isinstance(db_source, list):
                db_source = db_source[0]
            self._write_single_line('DBSOURCE', db_source)
        if 'references' in record.annotations:
            self._write_references(record)
        if 'comment' in record.annotations or 'structured_comment' in record.annotations:
            self._write_comment(record)
        handle.write('FEATURES             Location/Qualifiers\n')
        rec_length = len(record)
        for feature in record.features:
            self._write_feature(feature, rec_length)
        self._write_sequence(record)
        handle.write('//\n')

class EmblWriter(_InsdcWriter):
    """EMBL writer."""
    HEADER_WIDTH = 5
    QUALIFIER_INDENT = 21
    QUALIFIER_INDENT_STR = 'FT' + ' ' * (QUALIFIER_INDENT - 2)
    QUALIFIER_INDENT_TMP = 'FT   %s                '
    FEATURE_HEADER = 'FH   Key             Location/Qualifiers\nFH\n'
    LETTERS_PER_BLOCK = 10
    BLOCKS_PER_LINE = 6
    LETTERS_PER_LINE = LETTERS_PER_BLOCK * BLOCKS_PER_LINE
    POSITION_PADDING = 10

    def _write_contig(self, record):
        if False:
            for i in range(10):
                print('nop')
        max_len = self.MAX_WIDTH - self.HEADER_WIDTH
        lines = self._split_contig(record, max_len)
        for text in lines:
            self._write_single_line('CO', text)

    def _write_sequence(self, record):
        if False:
            for i in range(10):
                print('nop')
        handle = self.handle
        try:
            data = _get_seq_string(record)
        except UndefinedSequenceError:
            if 'contig' in record.annotations:
                self._write_contig(record)
            else:
                handle.write('SQ   \n')
            return
        data = data.lower()
        seq_len = len(data)
        molecule_type = record.annotations.get('molecule_type')
        if molecule_type is not None and 'DNA' in molecule_type:
            a_count = data.count('A') + data.count('a')
            c_count = data.count('C') + data.count('c')
            g_count = data.count('G') + data.count('g')
            t_count = data.count('T') + data.count('t')
            other = seq_len - (a_count + c_count + g_count + t_count)
            handle.write('SQ   Sequence %i BP; %i A; %i C; %i G; %i T; %i other;\n' % (seq_len, a_count, c_count, g_count, t_count, other))
        else:
            handle.write('SQ   \n')
        for line_number in range(seq_len // self.LETTERS_PER_LINE):
            handle.write('    ')
            for block in range(self.BLOCKS_PER_LINE):
                index = self.LETTERS_PER_LINE * line_number + self.LETTERS_PER_BLOCK * block
                handle.write(f' {data[index:index + self.LETTERS_PER_BLOCK]}')
            handle.write(str((line_number + 1) * self.LETTERS_PER_LINE).rjust(self.POSITION_PADDING))
            handle.write('\n')
        if seq_len % self.LETTERS_PER_LINE:
            line_number = seq_len // self.LETTERS_PER_LINE
            handle.write('    ')
            for block in range(self.BLOCKS_PER_LINE):
                index = self.LETTERS_PER_LINE * line_number + self.LETTERS_PER_BLOCK * block
                handle.write(f' {data[index:index + self.LETTERS_PER_BLOCK]}'.ljust(11))
            handle.write(str(seq_len).rjust(self.POSITION_PADDING))
            handle.write('\n')

    def _write_single_line(self, tag, text):
        if False:
            return 10
        assert len(tag) == 2
        line = tag + '   ' + text
        if len(text) > self.MAX_WIDTH:
            warnings.warn(f'Line {line!r} too long', BiopythonWarning)
        self.handle.write(line + '\n')

    def _write_multi_line(self, tag, text):
        if False:
            return 10
        max_len = self.MAX_WIDTH - self.HEADER_WIDTH
        lines = self._split_multi_line(text, max_len)
        for line in lines:
            self._write_single_line(tag, line)

    def _write_the_first_lines(self, record):
        if False:
            i = 10
            return i + 15
        'Write the ID and AC lines (PRIVATE).'
        if '.' in record.id and record.id.rsplit('.', 1)[1].isdigit():
            version = 'SV ' + record.id.rsplit('.', 1)[1]
            accession = self._get_annotation_str(record, 'accession', record.id.rsplit('.', 1)[0], just_first=True)
        else:
            version = ''
            accession = self._get_annotation_str(record, 'accession', record.id, just_first=True)
        if ';' in accession:
            raise ValueError(f"Cannot have semi-colon in EMBL accession, '{accession}'")
        if ' ' in accession:
            raise ValueError(f"Cannot have spaces in EMBL accession, '{accession}'")
        topology = self._get_annotation_str(record, 'topology', default='')
        mol_type = record.annotations.get('molecule_type')
        if mol_type is None:
            raise ValueError('missing molecule_type in annotations')
        if mol_type not in ('DNA', 'genomic DNA', 'unassigned DNA', 'mRNA', 'RNA', 'protein'):
            warnings.warn(f'Non-standard molecule type: {mol_type}', BiopythonWarning)
        mol_type_upper = mol_type.upper()
        if 'DNA' in mol_type_upper:
            units = 'BP'
        elif 'RNA' in mol_type_upper:
            units = 'BP'
        elif 'PROTEIN' in mol_type_upper:
            mol_type = 'PROTEIN'
            units = 'AA'
        else:
            raise ValueError(f"failed to understand molecule_type '{mol_type}'")
        division = self._get_data_division(record)
        handle = self.handle
        self._write_single_line('ID', '%s; %s; %s; %s; ; %s; %i %s.' % (accession, version, topology, mol_type, division, len(record), units))
        handle.write('XX\n')
        self._write_single_line('AC', accession + ';')
        handle.write('XX\n')

    @staticmethod
    def _get_data_division(record):
        if False:
            print('Hello World!')
        try:
            division = record.annotations['data_file_division']
        except KeyError:
            division = 'UNC'
        if division in ['PHG', 'ENV', 'FUN', 'HUM', 'INV', 'MAM', 'VRT', 'MUS', 'PLN', 'PRO', 'ROD', 'SYN', 'TGN', 'UNC', 'VRL', 'XXX']:
            pass
        else:
            gbk_to_embl = {'BCT': 'PRO', 'UNK': 'UNC'}
            try:
                division = gbk_to_embl[division]
            except KeyError:
                division = 'UNC'
        assert len(division) == 3
        return division

    def _write_keywords(self, record):
        if False:
            print('Hello World!')
        for keyword in record.annotations['keywords']:
            self._write_single_line('KW', keyword)
        self.handle.write('XX\n')

    def _write_references(self, record):
        if False:
            i = 10
            return i + 15
        number = 0
        for ref in record.annotations['references']:
            if not isinstance(ref, SeqFeature.Reference):
                continue
            number += 1
            self._write_single_line('RN', '[%i]' % number)
            if ref.location and len(ref.location) == 1:
                self._write_single_line('RP', '%i-%i' % (ref.location[0].start + 1, ref.location[0].end))
            if ref.pubmed_id:
                self._write_single_line('RX', f'PUBMED; {ref.pubmed_id}.')
            if ref.consrtm:
                self._write_single_line('RG', f'{ref.consrtm}')
            if ref.authors:
                self._write_multi_line('RA', ref.authors + ';')
            if ref.title:
                self._write_multi_line('RT', f'"{ref.title}";')
            if ref.journal:
                self._write_multi_line('RL', ref.journal)
            self.handle.write('XX\n')

    def _write_comment(self, record):
        if False:
            return 10
        comment = record.annotations['comment']
        if isinstance(comment, str):
            lines = comment.split('\n')
        elif isinstance(comment, (list, tuple)):
            lines = comment
        else:
            raise ValueError('Could not understand comment annotation')
        if not lines:
            return
        for line in lines:
            self._write_multi_line('CC', line)
        self.handle.write('XX\n')

    def write_record(self, record):
        if False:
            i = 10
            return i + 15
        'Write a single record to the output file.'
        handle = self.handle
        self._write_the_first_lines(record)
        for xref in sorted(record.dbxrefs):
            if xref.startswith('BioProject:'):
                self._write_single_line('PR', xref[3:] + ';')
                handle.write('XX\n')
                break
            if xref.startswith('Project:'):
                self._write_single_line('PR', xref + ';')
                handle.write('XX\n')
                break
        descr = record.description
        if descr == '<unknown description>':
            descr = '.'
        self._write_multi_line('DE', descr)
        handle.write('XX\n')
        if 'keywords' in record.annotations:
            self._write_keywords(record)
        self._write_multi_line('OS', self._get_annotation_str(record, 'organism'))
        try:
            taxonomy = '; '.join(record.annotations['taxonomy']) + '.'
        except KeyError:
            taxonomy = '.'
        self._write_multi_line('OC', taxonomy)
        handle.write('XX\n')
        if 'references' in record.annotations:
            self._write_references(record)
        if 'comment' in record.annotations:
            self._write_comment(record)
        handle.write(self.FEATURE_HEADER)
        rec_length = len(record)
        for feature in record.features:
            self._write_feature(feature, rec_length)
        handle.write('XX\n')
        self._write_sequence(record)
        handle.write('//\n')

class ImgtWriter(EmblWriter):
    """IMGT writer (EMBL format variant)."""
    HEADER_WIDTH = 5
    QUALIFIER_INDENT = 25
    QUALIFIER_INDENT_STR = 'FT' + ' ' * (QUALIFIER_INDENT - 2)
    QUALIFIER_INDENT_TMP = 'FT   %s                    '
    FEATURE_HEADER = 'FH   Key                 Location/Qualifiers\nFH\n'

def _genbank_convert_fasta(in_file, out_file):
    if False:
        while True:
            i = 10
    'Fast GenBank to FASTA (PRIVATE).'
    records = GenBankScanner().parse_records(in_file, do_features=False)
    return SeqIO.write(records, out_file, 'fasta')

def _embl_convert_fasta(in_file, out_file):
    if False:
        while True:
            i = 10
    'Fast EMBL to FASTA (PRIVATE).'
    records = EmblScanner().parse_records(in_file, do_features=False)
    return SeqIO.write(records, out_file, 'fasta')
if __name__ == '__main__':
    from Bio._utils import run_doctest
    run_doctest(verbose=0)