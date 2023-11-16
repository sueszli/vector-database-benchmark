"""Parsing TRANSFAC files."""
from Bio import motifs

class Motif(motifs.Motif, dict):
    """Store the information for one TRANSFAC motif.

    This class inherits from the Bio.motifs.Motif base class, as well
    as from a Python dictionary. All motif information found by the parser
    is stored as attributes of the base class when possible; see the
    Bio.motifs.Motif base class for a description of these attributes. All
    other information associated with the motif is stored as (key, value)
    pairs in the dictionary, where the key is the two-letter fields as found
    in the TRANSFAC file. References are an exception: These are stored in
    the .references attribute.

    These fields are commonly found in TRANSFAC files::

        AC:    Accession number
        AS:    Accession numbers, secondary
        BA:    Statistical basis
        BF:    Binding factors
        BS:    Factor binding sites underlying the matrix
               [sequence; SITE accession number; start position for matrix
               sequence; length of sequence used; number of gaps inserted;
               strand orientation.]
        CC:    Comments
        CO:    Copyright notice
        DE:    Short factor description
        DR:    External databases
               [database name: database accession number]
        DT:    Date created/updated
        HC:    Subfamilies
        HP:    Superfamilies
        ID:    Identifier
        NA:    Name of the binding factor
        OC:    Taxonomic classification
        OS:    Species/Taxon
        OV:    Older version
        PV:    Preferred version
        TY:    Type
        XX:    Empty line; these are not stored in the Record.

    References are stored in an .references attribute, which is a list of
    dictionaries with the following keys::

        RN:    Reference number
        RA:    Reference authors
        RL:    Reference data
        RT:    Reference title
        RX:    PubMed ID

    For more information, see the TRANSFAC documentation.
    """
    multiple_value_keys = {'BF', 'OV', 'HP', 'BS', 'HC', 'DT', 'DR', 'CC'}
    reference_keys = {'RX', 'RA', 'RT', 'RL'}

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        try:
            value = super().__getitem__(key)
        except TypeError:
            value = super(motifs.Motif, self).__getitem__(key)
        return value

class Record(list):
    """Store the information in a TRANSFAC matrix table.

    The record inherits from a list containing the individual motifs.

    Attributes:
     - version - The version number, corresponding to the 'VV' field
       in the TRANSFAC file;

    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        'Initialize the class.'
        self.version = None

    def __str__(self):
        if False:
            return 10
        'Turn the TRANSFAC matrix into a string.'
        return write(self)

def read(handle, strict=True):
    if False:
        print('Hello World!')
    'Parse a transfac format handle into a Record object.'
    annotations = {}
    references = []
    counts = None
    record = Record()
    for line in handle:
        line = line.strip()
        if not line:
            continue
        key_value = line.split(None, 1)
        key = key_value[0].strip()
        if strict:
            if len(key) != 2:
                raise ValueError(f'The key value of a TRANSFAC motif line should have 2 characters:"{line}"')
        if len(key_value) == 2:
            value = key_value[1].strip()
            if strict:
                if not line.partition('  ')[1]:
                    raise ValueError(f'A TRANSFAC motif line should have 2 spaces between key and value columns: "{line}"')
        if key == 'VV':
            record.version = value
        elif key in ('P0', 'PO'):
            counts = {}
            if value.split()[:4] != ['A', 'C', 'G', 'T']:
                raise ValueError(f'A TRANSFAC matrix "{key}" line should be followed by "A C G T": {line}')
            length = 0
            for c in 'ACGT':
                counts[c] = []
            for line in handle:
                line = line.strip()
                key_value = line.split(None, 1)
                key = key_value[0].strip()
                if len(key_value) == 2:
                    value = key_value[1].strip()
                    if strict:
                        if not line.partition('  ')[1]:
                            raise ValueError(f'A TRANSFAC motif line should have 2 spaces between key and value columns: "{line}"')
                try:
                    i = int(key)
                except ValueError:
                    break
                if length == 0 and i == 0:
                    if strict:
                        raise ValueError(f'A TRANSFAC matrix should start with "01" as first row of the matrix, but this matrix uses "00": "{line}')
                else:
                    length += 1
                if i != length:
                    raise ValueError(f'The TRANSFAC matrix row number does not match the position in the matrix: "{line}"')
                if strict:
                    if len(key) == 1:
                        raise ValueError(f'A TRANSFAC matrix line should have a 2 digit key at the start of the line ("{i:02d}"), but this matrix uses "{i:d}": "{line:s}".')
                    if len(key_value) != 2:
                        raise ValueError(f'A TRANSFAC matrix line should have a key and a value: "{line}"')
                values = value.split()[:4]
                if len(values) != 4:
                    raise ValueError(f'A TRANSFAC matrix line should have a value for each nucleotide (A, C, G and T): "{line}"')
                for (c, v) in zip('ACGT', values):
                    counts[c].append(float(v))
        if line == 'XX':
            pass
        elif key == 'RN':
            (index, separator, accession) = value.partition(';')
            if index[0] != '[':
                raise ValueError(f'The index "{index}" in a TRANSFAC RN line should start with a "[": "{line}"')
            if index[-1] != ']':
                raise ValueError(f'The index "{index}" in a TRANSFAC RN line should end with a "]": "{line}"')
            index = int(index[1:-1])
            if len(references) != index - 1:
                raise ValueError(f'The index "{index:d}" of the TRANSFAC RN line does not match the current number of seen references "{len(references) + 1:d}": "{line:s}"')
            reference = {key: value}
            references.append(reference)
        elif key == '//':
            if counts is not None:
                motif = Motif(alphabet='ACGT', counts=counts)
                motif.update(annotations)
                motif.references = references
                record.append(motif)
            annotations = {}
            references = []
        elif key in Motif.reference_keys:
            reference[key] = value
        elif key in Motif.multiple_value_keys:
            if key not in annotations:
                annotations[key] = []
            annotations[key].append(value)
        else:
            annotations[key] = value
    return record

def write(motifs):
    if False:
        return 10
    'Write the representation of a motif in TRANSFAC format.'
    blocks = []
    try:
        version = motifs.version
    except AttributeError:
        pass
    else:
        if version is not None:
            block = 'VV  %s\nXX\n//\n' % version
            blocks.append(block)
    multiple_value_keys = Motif.multiple_value_keys
    sections = (('AC', 'AS'), ('ID',), ('DT', 'CO'), ('NA',), ('DE',), ('TY',), ('OS', 'OC'), ('HP', 'HC'), ('BF',), ('P0',), ('BA',), ('BS',), ('CC',), ('DR',), ('OV', 'PV'))
    for motif in motifs:
        lines = []
        for section in sections:
            blank = False
            for key in section:
                if key == 'P0':
                    length = motif.length
                    if length == 0:
                        continue
                    sequence = motif.degenerate_consensus
                    letters = sorted(motif.alphabet)
                    line = '      '.join(['P0'] + letters)
                    lines.append(line)
                    for i in range(length):
                        line = ' '.join(['%02.d'] + ['%6.20g' for _ in letters]) + '      %s'
                        line = line % tuple([i + 1] + [motif.counts[_][i] for _ in letters] + [sequence[i]])
                        lines.append(line)
                    blank = True
                else:
                    try:
                        value = motif.get(key)
                    except AttributeError:
                        value = None
                    if value is not None:
                        if key in multiple_value_keys:
                            for v in value:
                                line = f'{key}  {v}'
                                lines.append(line)
                        else:
                            line = f'{key}  {value}'
                            lines.append(line)
                        blank = True
                if key == 'PV':
                    try:
                        references = motif.references
                    except AttributeError:
                        pass
                    else:
                        keys = ('RN', 'RX', 'RA', 'RT', 'RL')
                        for reference in references:
                            for key in keys:
                                value = reference.get(key)
                                if value is None:
                                    continue
                                line = f'{key}  {value}'
                                lines.append(line)
                                blank = True
            if blank:
                line = 'XX'
                lines.append(line)
        line = '//'
        lines.append(line)
        block = '\n'.join(lines) + '\n'
        blocks.append(block)
    text = ''.join(blocks)
    return text