"""Nexus class. Parse the contents of a NEXUS file.

Based upon 'NEXUS: An extensible file format for systematic information'
Maddison, Swofford, Maddison. 1997. Syst. Biol. 46(4):590-621
"""
from functools import reduce
import copy
import math
import random
import sys
import warnings
from Bio import File
from Bio.Data import IUPACData
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning, BiopythonWarning
from Bio.Nexus.StandardData import StandardData
from Bio.Nexus.Trees import Tree
INTERLEAVE = 70
SPECIAL_COMMANDS = ['charstatelabels', 'charlabels', 'taxlabels', 'taxset', 'charset', 'charpartition', 'taxpartition', 'matrix', 'tree', 'utree', 'translate', 'codonposset', 'title']
KNOWN_NEXUS_BLOCKS = ['trees', 'data', 'characters', 'taxa', 'sets', 'codons']
PUNCTUATION = '()[]{}\\,;:=*\\\'"`+-<>'
MRBAYESSAFE = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890_'
WHITESPACE = ' \t\n'
SPECIALCOMMENTS = ['&']
CHARSET = 'chars'
TAXSET = 'taxa'
CODONPOSITIONS = 'codonpositions'
DEFAULTNEXUS = '#NEXUS\nbegin data; dimensions ntax=0 nchar=0; format datatype=dna; end; '

class NexusError(Exception):
    """Provision for the management of Nexus exceptions."""

class CharBuffer:
    """Helps reading NEXUS-words and characters from a buffer (semi-PRIVATE).

    This class is not intended for public use (any more).
    """

    def __init__(self, string):
        if False:
            print('Hello World!')
        'Initialize the class.'
        if string:
            self.buffer = list(string)
        else:
            self.buffer = []

    def peek(self):
        if False:
            return 10
        'Return the first character from the buffer.'
        if self.buffer:
            return self.buffer[0]
        else:
            return None

    def peek_nonwhitespace(self):
        if False:
            print('Hello World!')
        'Return the first character from the buffer, do not include spaces.'
        b = ''.join(self.buffer).strip()
        if b:
            return b[0]
        else:
            return None

    def __next__(self):
        if False:
            for i in range(10):
                print('nop')
        'Iterate over NEXUS characters in the file.'
        if self.buffer:
            return self.buffer.pop(0)
        else:
            return None

    def next_nonwhitespace(self):
        if False:
            return 10
        'Check for next non whitespace character in NEXUS file.'
        while True:
            p = next(self)
            if p is None:
                break
            if p not in WHITESPACE:
                return p
        return None

    def skip_whitespace(self):
        if False:
            while True:
                i = 10
        'Skip whitespace characters in NEXUS file.'
        while self.buffer[0] in WHITESPACE:
            self.buffer = self.buffer[1:]

    def next_until(self, target):
        if False:
            print('Hello World!')
        'Iterate over the NEXUS file until a target character is reached.'
        for t in target:
            try:
                pos = self.buffer.index(t)
            except ValueError:
                pass
            else:
                found = ''.join(self.buffer[:pos])
                self.buffer = self.buffer[pos:]
                return found
        else:
            return None

    def peek_word(self, word):
        if False:
            i = 10
            return i + 15
        'Return a word stored in the buffer.'
        return ''.join(self.buffer[:len(word)]) == word

    def next_word(self):
        if False:
            print('Hello World!')
        'Return the next NEXUS word from a string.\n\n        This deals with single and double quotes, whitespace and punctuation.\n        '
        word = []
        quoted = False
        first = self.next_nonwhitespace()
        if not first:
            return None
        word.append(first)
        if first == "'":
            quoted = "'"
        elif first == '"':
            quoted = '"'
        elif first in PUNCTUATION:
            return first
        while True:
            c = self.peek()
            if c == quoted:
                word.append(next(self))
                if self.peek() == quoted:
                    next(self)
                elif quoted:
                    break
            elif quoted:
                word.append(next(self))
            elif not c or c in PUNCTUATION or c in WHITESPACE:
                break
            else:
                word.append(next(self))
        return ''.join(word)

    def rest(self):
        if False:
            while True:
                i = 10
        'Return the rest of the string without parsing.'
        return ''.join(self.buffer)

class StepMatrix:
    """Calculate a stepmatrix for weighted parsimony.

    See :
    COMBINATORIAL WEIGHTS IN PHYLOGENETIC ANALYSIS - A STATISTICAL PARSIMONY PROCEDURE
    Wheeler (1990), Cladistics 6:269-275.
    """

    def __init__(self, symbols, gap):
        if False:
            while True:
                i = 10
        'Initialize the class.'
        self.data = {}
        self.symbols = sorted(symbols)
        if gap:
            self.symbols.append(gap)
        for x in self.symbols:
            for y in [s for s in self.symbols if s != x]:
                self.set(x, y, 0)

    def set(self, x, y, value):
        if False:
            while True:
                i = 10
        "Set a given value in the matrix's position."
        if x > y:
            (x, y) = (y, x)
        self.data[x + y] = value

    def add(self, x, y, value):
        if False:
            i = 10
            return i + 15
        "Add the given value to existing, in matrix's position."
        if x > y:
            (x, y) = (y, x)
        self.data[x + y] += value

    def sum(self):
        if False:
            return 10
        'Calculate the associations, makes matrix of associations.'
        return reduce(lambda x, y: x + y, self.data.values())

    def transformation(self):
        if False:
            for i in range(10):
                print('nop')
        'Calculate the transformation matrix.\n\n        Normalizes the columns of the matrix of associations.\n        '
        total = self.sum()
        if total != 0:
            for k in self.data:
                self.data[k] /= total
        return self

    def weighting(self):
        if False:
            for i in range(10):
                print('nop')
        'Calculate the Phylogenetic weight matrix.\n\n        Constructed from the logarithmic transformation of the\n        transformation matrix.\n        '
        for k in self.data:
            if self.data[k] != 0:
                self.data[k] = -math.log(self.data[k])
        return self

    def smprint(self, name='your_name_here'):
        if False:
            i = 10
            return i + 15
        'Print a stepmatrix.'
        matrix = 'usertype %s stepmatrix=%d\n' % (name, len(self.symbols))
        matrix += f"        {'        '.join(self.symbols)}\n"
        for x in self.symbols:
            matrix += '[%s]'.ljust(8) % x
            for y in self.symbols:
                if x == y:
                    matrix += ' .       '
                else:
                    if x > y:
                        (x1, y1) = (y, x)
                    else:
                        (x1, y1) = (x, y)
                    if self.data[x1 + y1] == 0:
                        matrix += 'inf.     '
                    else:
                        matrix += '%2.2f'.ljust(10) % self.data[x1 + y1]
            matrix += '\n'
        matrix += ';\n'
        return matrix

def safename(name, mrbayes=False):
    if False:
        for i in range(10):
            print('nop')
    'Return a taxon identifier according to NEXUS standard.\n\n    Wrap quotes around names with punctuation or whitespace, and double\n    single quotes.\n\n    mrbayes=True: write names without quotes, whitespace or punctuation\n    for the mrbayes software package.\n    '
    if mrbayes:
        safe = name.replace(' ', '_')
        safe = ''.join((c for c in safe if c in MRBAYESSAFE))
    else:
        safe = name.replace("'", "''")
        if set(safe).intersection(set(WHITESPACE + PUNCTUATION)):
            safe = "'" + safe + "'"
    return safe

def quotestrip(word):
    if False:
        while True:
            i = 10
    'Remove quotes and/or double quotes around identifiers.'
    if not word:
        return None
    while word.startswith("'") and word.endswith("'") or (word.startswith('"') and word.endswith('"')):
        word = word[1:-1]
    return word

def get_start_end(sequence, skiplist=('-', '?')):
    if False:
        return 10
    "Return position of first and last character which is not in skiplist.\n\n    Skiplist defaults to ['-','?'].\n    "
    length = len(sequence)
    if length == 0:
        return (None, None)
    end = length - 1
    while end >= 0 and sequence[end] in skiplist:
        end -= 1
    start = 0
    while start < length and sequence[start] in skiplist:
        start += 1
    if start == length and end == -1:
        return (-1, -1)
    else:
        return (start, end)

def _sort_keys_by_values(p):
    if False:
        return 10
    'Return a sorted list of keys of p sorted by values of p (PRIVATE).'
    return sorted((pn for pn in p if p[pn]), key=lambda pn: p[pn])

def _make_unique(values):
    if False:
        i = 10
        return i + 15
    'Check all values in list are unique and return a pruned and sorted list (PRIVATE).'
    return sorted(set(values))

def _unique_label(previous_labels, label):
    if False:
        i = 10
        return i + 15
    'Return a unique name if label is already in previous_labels (PRIVATE).'
    while label in previous_labels:
        label_split = label.split('.')
        if label_split[-1].startswith('copy'):
            copy_num = 1
            if label_split[-1] != 'copy':
                copy_num = int(label_split[-1][4:]) + 1
            new_label = f"{'.'.join(label_split[:-1])}.copy{copy_num}"
            label = new_label
        else:
            label += '.copy'
    return label

def _seqmatrix2strmatrix(matrix):
    if False:
        for i in range(10):
            print('nop')
    'Convert a Seq-object matrix to a plain sequence-string matrix (PRIVATE).'
    return {t: str(matrix[t]) for t in matrix}

def _compact4nexus(orig_list):
    if False:
        while True:
            i = 10
    "Compact lists for Nexus output (PRIVATE).\n\n    Example\n    -------\n    >>> _compact4nexus([1, 2, 3, 5, 6, 7, 8, 12, 15, 18, 20])\n    '2-4 6-9 13-19\\\\3 21'\n\n    Transform [1 2 3 5 6 7 8 12 15 18 20] (baseindex 0, used in the Nexus class)\n    into '2-4 6-9 13-19\\\\3 21' (baseindex 1, used in programs like Paup or MrBayes.).\n\n    "
    if not orig_list:
        return ''
    orig_list = sorted(set(orig_list))
    shortlist = []
    clist = orig_list[:]
    clist.append(clist[-1] + 0.5)
    while len(clist) > 1:
        step = 1
        for (i, x) in enumerate(clist):
            if x == clist[0] + i * step:
                continue
            elif i == 1 and len(clist) > 3 and (clist[i + 1] - x == x - clist[0]):
                step = x - clist[0]
            else:
                sub = clist[:i]
                if len(sub) == 1:
                    shortlist.append(str(sub[0] + 1))
                elif step == 1:
                    shortlist.append('%d-%d' % (sub[0] + 1, sub[-1] + 1))
                else:
                    shortlist.append('%d-%d\\%d' % (sub[0] + 1, sub[-1] + 1, step))
                clist = clist[i:]
                break
    return ' '.join(shortlist)

def combine(matrices):
    if False:
        for i in range(10):
            print('nop')
    'Combine matrices in [(name,nexus-instance),...] and return new nexus instance.\n\n    combined_matrix=combine([(name1,nexus_instance1),(name2,nexus_instance2),...]\n    Character sets, character partitions and taxon sets are prefixed, readjusted\n    and present in the combined matrix.\n    '
    if not matrices:
        return None
    name = matrices[0][0]
    combined = copy.deepcopy(matrices[0][1])
    mixed_datatypes = len({n[1].datatype for n in matrices}) > 1
    if mixed_datatypes:
        combined.datatype = 'None'
    combined.charlabels = None
    combined.statelabels = None
    combined.interleave = False
    combined.translate = None
    for (cn, cs) in combined.charsets.items():
        combined.charsets[f'{name}.{cn}'] = cs
        del combined.charsets[cn]
    for (tn, ts) in combined.taxsets.items():
        combined.taxsets[f'{name}.{tn}'] = ts
        del combined.taxsets[tn]
    combined.charpartitions = {'combined': {name: list(range(combined.nchar))}}
    for (n, m) in matrices[1:]:
        both = [t for t in combined.taxlabels if t in m.taxlabels]
        combined_only = [t for t in combined.taxlabels if t not in both]
        m_only = [t for t in m.taxlabels if t not in both]
        for t in both:
            combined.matrix[t] += Seq(str(m.matrix[t]).replace(m.gap, combined.gap).replace(m.missing, combined.missing))
        for t in combined_only:
            combined.matrix[t] += Seq(combined.missing * m.nchar)
        for t in m_only:
            combined.matrix[t] = Seq(combined.missing * combined.nchar) + Seq(str(m.matrix[t]).replace(m.gap, combined.gap).replace(m.missing, combined.missing))
        combined.taxlabels.extend(m_only)
        for (cn, cs) in m.charsets.items():
            combined.charsets[f'{n}.{cn}'] = [x + combined.nchar for x in cs]
        if m.taxsets:
            if not combined.taxsets:
                combined.taxsets = {}
            combined.taxsets.update({f'{n}.{tn}': ts for (tn, ts) in m.taxsets.items()})
        combined.charpartitions['combined'][n] = list(range(combined.nchar, combined.nchar + m.nchar))
        if m.charlabels:
            if not combined.charlabels:
                combined.charlabels = {}
            combined.charlabels.update({combined.nchar + i: label for (i, label) in m.charlabels.items()})
        combined.nchar += m.nchar
        combined.ntax += len(m_only)
    for c in combined.charpartitions['combined']:
        combined.charsets[c] = combined.charpartitions['combined'][c]
    return combined

def _kill_comments_and_break_lines(text):
    if False:
        i = 10
        return i + 15
    "Delete []-delimited comments out of a file and break into lines separated by ';' (PRIVATE).\n\n    stripped_text=_kill_comments_and_break_lines(text):\n    Nested and multiline comments are allowed. [ and ] symbols within single\n    or double quotes are ignored, newline ends a quote, all symbols with quotes are\n    treated the same (thus not quoting inside comments like [this character ']' ends a comment])\n    Special [&...] and [\\...] comments remain untouched, if not inside standard comment.\n    Quotes inside special [& and [\\ are treated as normal characters,\n    but no nesting inside these special comments allowed (like [&   [\\   ]]).\n    ';' is deleted from end of line.\n\n    NOTE: this function is very slow for large files, and obsolete when using C extension cnexus\n    "
    if not text:
        return ''
    contents = iter(text)
    newtext = []
    newline = []
    quotelevel = ''
    speciallevel = False
    commlevel = 0
    t2 = next(contents)
    while True:
        t = t2
        try:
            t2 = next(contents)
        except StopIteration:
            t2 = None
        if t is None:
            break
        if t == quotelevel and (not (commlevel or speciallevel)):
            quotelevel = ''
        elif not quotelevel and (not (commlevel or speciallevel)) and (t == '"' or t == "'"):
            quotelevel = t
        elif not quotelevel and t == '[':
            if t2 in SPECIALCOMMENTS and commlevel == 0 and (not speciallevel):
                speciallevel = True
            else:
                commlevel += 1
        elif not quotelevel and t == ']':
            if speciallevel:
                speciallevel = False
            else:
                commlevel -= 1
                if commlevel < 0:
                    raise NexusError('Nexus formatting error: unmatched ]')
                continue
        if commlevel == 0:
            if t == ';' and (not quotelevel):
                newtext.append(''.join(newline))
                newline = []
            else:
                newline.append(t)
    if newline:
        newtext.append('\n'.join(newline))
    if commlevel > 0:
        raise NexusError('Nexus formatting error: unmatched [')
    return newtext

def _adjust_lines(lines):
    if False:
        print('Hello World!')
    "Adjust linebreaks to match ';', strip leading/trailing whitespace (PRIVATE).\n\n    list_of_commandlines=_adjust_lines(input_text)\n    Lines are adjusted so that no linebreaks occur within a commandline\n    (except matrix command line)\n    "
    formatted_lines = []
    for line in lines:
        line = line.replace('\r\n', '\n').replace('\r', '\n').strip()
        if line.lower().startswith('matrix'):
            formatted_lines.append(line)
        else:
            line = line.replace('\n', ' ')
            if line:
                formatted_lines.append(line)
    return formatted_lines

def _replace_parenthesized_ambigs(seq, rev_ambig_values):
    if False:
        return 10
    'Replace ambigs in xxx(ACG)xxx format by IUPAC ambiguity code (PRIVATE).'
    opening = seq.find('(')
    while opening > -1:
        closing = seq.find(')')
        if closing < 0:
            raise NexusError('Missing closing parenthesis in: ' + seq)
        elif closing < opening:
            raise NexusError('Missing opening parenthesis in: ' + seq)
        ambig = ''.join(sorted(seq[opening + 1:closing]))
        ambig_code = rev_ambig_values[ambig.upper()]
        if ambig != ambig.upper():
            ambig_code = ambig_code.lower()
        seq = seq[:opening] + ambig_code + seq[closing + 1:]
        opening = seq.find('(')
    return seq

class Commandline:
    """Represent a commandline as command and options."""

    def __init__(self, line, title):
        if False:
            print('Hello World!')
        'Initialize the class.'
        self.options = {}
        options = []
        self.command = None
        try:
            (self.command, options) = line.strip().split('\n', 1)
        except ValueError:
            self.command = line.split()[0]
            options = ' '.join(line.split()[1:])
        self.command = self.command.strip().lower()
        if self.command in SPECIAL_COMMANDS:
            self.options = options.strip()
        elif len(options) > 0:
            try:
                options = options.replace('=', ' = ').split()
                valued_indices = [(n - 1, n, n + 1) for n in range(len(options)) if options[n] == '=' and n != 0 and (n != len(options))]
                indices = []
                for sl in valued_indices:
                    indices.extend(sl)
                token_indices = [n for n in range(len(options)) if n not in indices]
                for opt in valued_indices:
                    self.options[options[opt[0]].lower()] = options[opt[2]]
                for token in token_indices:
                    self.options[options[token].lower()] = None
            except ValueError:
                raise NexusError(f'Incorrect formatting in line: {line}') from None

class Block:
    """Represent a NEXUS block with block name and list of commandlines."""

    def __init__(self, title=None):
        if False:
            i = 10
            return i + 15
        'Initialize the class.'
        self.title = title
        self.commandlines = []

class Nexus:
    """Create the Nexus class, main class for the management of Nexus files."""

    def __init__(self, input=None):
        if False:
            i = 10
            return i + 15
        'Initialize the class.'
        self.ntax = 0
        self.nchar = 0
        self.unaltered_taxlabels = []
        self.taxlabels = []
        self.charlabels = None
        self.statelabels = None
        self.datatype = 'dna'
        self.respectcase = False
        self.missing = '?'
        self.gap = '-'
        self.symbols = None
        self.equate = None
        self.matchchar = None
        self.labels = None
        self.transpose = False
        self.interleave = False
        self.tokens = False
        self.eliminate = None
        self.matrix = None
        self.unknown_blocks = []
        self.taxsets = {}
        self.charsets = {}
        self.charpartitions = {}
        self.taxpartitions = {}
        self.trees = []
        self.translate = None
        self.structured = []
        self.set = {}
        self.options = {}
        self.codonposset = None
        self.options['gapmode'] = 'missing'
        if input:
            self.read(input)
        else:
            self.read(DEFAULTNEXUS)

    def get_original_taxon_order(self):
        if False:
            for i in range(10):
                print('nop')
        'Included for backwards compatibility (DEPRECATED).'
        warnings.warn('The get_original_taxon_order method has been deprecated and will likely be removed from Biopython in the near future. Please use the original_taxon_order attribute instead.', BiopythonDeprecationWarning)
        return self.taxlabels

    def set_original_taxon_order(self, value):
        if False:
            while True:
                i = 10
        'Included for backwards compatibility (DEPRECATED).'
        warnings.warn('The set_original_taxon_order method has been deprecated and will likely be removed from Biopython in the near future. Please use the original_taxon_order attribute instead.', BiopythonDeprecationWarning)
        self.taxlabels = value
    original_taxon_order = property(get_original_taxon_order, set_original_taxon_order)

    def read(self, input):
        if False:
            print('Hello World!')
        'Read and parse NEXUS input (a filename, file-handle, or string).'
        try:
            with File.as_handle(input) as fp:
                file_contents = fp.read()
                self.filename = getattr(fp, 'name', 'Unknown_nexus_file')
        except (TypeError, OSError, AttributeError):
            if isinstance(input, str):
                file_contents = input
                self.filename = 'input_string'
            else:
                raise NexusError(f'Unrecognized input: {input[:100]} ...') from None
        file_contents = file_contents.strip()
        if file_contents.startswith('#NEXUS'):
            file_contents = file_contents[6:]
        commandlines = _get_command_lines(file_contents)
        for (i, cl) in enumerate(commandlines):
            try:
                if cl[:6].upper() == '#NEXUS':
                    commandlines[i] = cl[6:].strip()
            except IndexError:
                pass
        nexus_block_gen = self._get_nexus_block(commandlines)
        while True:
            try:
                (title, contents) = next(nexus_block_gen)
            except StopIteration:
                break
            if title in KNOWN_NEXUS_BLOCKS:
                self._parse_nexus_block(title, contents)
            else:
                self._unknown_nexus_block(title, contents)

    def _get_nexus_block(self, file_contents):
        if False:
            print('Hello World!')
        'Return a generator for looping through Nexus blocks (PRIVATE).'
        inblock = False
        blocklines = []
        while file_contents:
            cl = file_contents.pop(0)
            if cl.lower().startswith('begin'):
                if not inblock:
                    inblock = True
                    title = cl.split()[1].lower()
                else:
                    raise NexusError(f'Illegal block nesting in block {title}')
            elif cl.lower().startswith('end'):
                if inblock:
                    inblock = False
                    yield (title, blocklines)
                    blocklines = []
                else:
                    raise NexusError("Unmatched 'end'.")
            elif inblock:
                blocklines.append(cl)

    def _unknown_nexus_block(self, title, contents):
        if False:
            while True:
                i = 10
        block = Block()
        block.commandlines.append(contents)
        block.title = title
        self.unknown_blocks.append(block)

    def _parse_nexus_block(self, title, contents):
        if False:
            print('Hello World!')
        'Parse a known Nexus Block (PRIVATE).'
        self._apply_block_structure(title, contents)
        block = self.structured[-1]
        for line in block.commandlines:
            try:
                getattr(self, '_' + line.command)(line.options)
            except AttributeError:
                raise NexusError(f'Unknown command: {line.command} ') from None

    def _title(self, options):
        if False:
            return 10
        pass

    def _link(self, options):
        if False:
            return 10
        pass

    def _dimensions(self, options):
        if False:
            i = 10
            return i + 15
        if 'ntax' in options:
            self.ntax = eval(options['ntax'])
        if 'nchar' in options:
            self.nchar = eval(options['nchar'])

    def _format(self, options):
        if False:
            while True:
                i = 10
        if 'respectcase' in options:
            self.respectcase = True
        if 'symbols' in options:
            self.symbols = ''.join(options['symbols'].split())
            if self.symbols.startswith('"') and self.symbols.endswith('"') or (self.symbols.startswith("'") and self.symbols.endswith("'")):
                self.symbols = self.symbols[1:-1]
            if not self.respectcase:
                self.symbols = list(self.symbols.upper())
        if 'datatype' in options:
            self.datatype = options['datatype'].lower()
            if self.datatype == 'dna' or self.datatype == 'nucleotide':
                self.ambiguous_values = IUPACData.ambiguous_dna_values.copy()
                self.unambiguous_letters = IUPACData.unambiguous_dna_letters
            elif self.datatype == 'rna':
                self.ambiguous_values = IUPACData.ambiguous_rna_values.copy()
                self.unambiguous_letters = IUPACData.unambiguous_rna_letters
            elif self.datatype == 'protein':
                self.ambiguous_values = {'B': 'DN', 'Z': 'EQ', 'X': IUPACData.protein_letters}
                self.unambiguous_letters = IUPACData.protein_letters + '*'
            elif self.datatype == 'standard':
                self.ambiguous_values = {}
                if not self.symbols:
                    raise NexusError('Symbols must be defined when using standard datatype. Please remove any whitespace (spaces, tabs, etc.) between values for symbols as this confuses the Nexus parser.')
                self.unambiguous_letters = ''.join(self.symbols)
                if not self.respectcase:
                    self.unambiguous_letters += self.unambiguous_letters.lower()
            else:
                raise NexusError('Unsupported datatype: ' + self.datatype)
            self.valid_characters = ''.join(self.ambiguous_values) + self.unambiguous_letters
            if not self.respectcase:
                self.valid_characters = self.valid_characters.lower() + self.valid_characters.upper()
            rev = {v: k for (k, v) in self.ambiguous_values.items() if k != 'X'}
            self.rev_ambiguous_values = {}
            for (k, v) in rev.items():
                key = sorted((c for c in k))
                self.rev_ambiguous_values[''.join(key)] = v
        if self.datatype in ['dna', 'nucleotide']:
            self.symbols = IUPACData.ambiguous_dna_letters
            if self.missing not in self.ambiguous_values:
                self.ambiguous_values[self.missing] = self.unambiguous_letters + self.gap
            self.ambiguous_values[self.gap] = self.gap
        elif self.datatype == 'rna':
            self.symbols = IUPACData.ambiguous_rna_letters
            if self.missing not in self.ambiguous_values:
                self.ambiguous_values[self.missing] = self.unambiguous_letters + self.gap
            self.ambiguous_values[self.gap] = self.gap
        if 'missing' in options:
            self.missing = options['missing'][0]
        if 'gap' in options:
            self.gap = options['gap'][0]
        if 'equate' in options:
            self.equate = options['equate']
        if 'matchchar' in options:
            self.matchchar = options['matchchar'][0]
        if 'labels' in options:
            self.labels = options['labels']
        if 'interleave' in options:
            if options['interleave'] is None or options['interleave'].lower() == 'yes':
                self.interleave = True
        if 'tokens' in options:
            self.tokens = True
        if 'notokens' in options:
            self.tokens = False

    def _set(self, options):
        if False:
            print('Hello World!')
        self.set = options

    def _options(self, options):
        if False:
            for i in range(10):
                print('nop')
        self.options = options

    def _eliminate(self, options):
        if False:
            print('Hello World!')
        self.eliminate = options

    def _taxlabels(self, options):
        if False:
            return 10
        'Get taxon labels (PRIVATE).\n\n        As the taxon names are already in the matrix, this is superfluous\n        except for transpose matrices, which are currently unsupported anyway.\n        Thus, we ignore the taxlabels command to make handling of duplicate\n        taxon names easier.\n        '

    def _check_taxlabels(self, taxon):
        if False:
            for i in range(10):
                print('nop')
        'Check for presence of taxon in self.taxlabels (PRIVATE).'
        nextaxa = {t.replace(' ', '_'): t for t in self.taxlabels}
        nexid = taxon.replace(' ', '_')
        return nextaxa.get(nexid)

    def _charlabels(self, options):
        if False:
            return 10
        'Get labels for characters (PRIVATE).'
        self.charlabels = {}
        opts = CharBuffer(options)
        while True:
            w = opts.next_word()
            if w is None:
                break
            identifier = self._resolve(w, set_type=CHARSET)
            state = quotestrip(opts.next_word())
            self.charlabels[identifier] = state
            c = opts.next_nonwhitespace()
            if c is None:
                break
            elif c != ',':
                raise NexusError(f"Missing ',' in line {options}.")

    def _charstatelabels(self, options):
        if False:
            while True:
                i = 10
        self.charlabels = {}
        self.statelabels = {}
        opts = CharBuffer(options)
        if not self.symbols:
            raise NexusError('Symbols must be defined when using character states')
        while True:
            w = opts.next_word()
            if w is None:
                break
            identifier = self._resolve(w, set_type=CHARSET)
            character = quotestrip(opts.next_word())
            self.charlabels[identifier] = character
            self.statelabels[identifier] = []
            c = opts.next_nonwhitespace()
            if c is None:
                break
            elif c != ',':
                if c != '/':
                    raise NexusError(f"Missing ',' in line {options}.")
                state = quotestrip(opts.next_word())
                if state is None:
                    raise NexusError(f'Missing character state in line {options}.')
                while True:
                    if len(self.statelabels[identifier]) > len(self.symbols):
                        raise NexusError('Character states exceed number of available symbols in line %s.' % options)
                    self.statelabels[identifier].append(state)
                    state = quotestrip(opts.next_word())
                    if state is None:
                        return
                    elif state == ',':
                        break

    def _statelabels(self, options):
        if False:
            i = 10
            return i + 15
        pass

    def _matrix(self, options):
        if False:
            i = 10
            return i + 15
        'Create a matrix for NEXUS object (PRIVATE).'
        if not self.ntax or not self.nchar:
            raise NexusError('Dimensions must be specified before matrix!')
        self.matrix = {}
        taxcount = 0
        first_matrix_block = True
        lines = [_.strip() for _ in options.split('\n') if _.strip() != '']
        lineiter = iter(lines)
        while True:
            try:
                line = next(lineiter)
            except StopIteration:
                if taxcount < self.ntax:
                    raise NexusError('Not enough taxa in matrix.') from None
                elif taxcount > self.ntax:
                    raise NexusError('Too many taxa in matrix.') from None
                else:
                    break
            taxcount += 1
            if taxcount > self.ntax:
                if not self.interleave:
                    raise NexusError('Too many taxa in matrix - should matrix be interleaved?')
                else:
                    taxcount = 1
                    first_matrix_block = False
            linechars = CharBuffer(line)
            id = quotestrip(linechars.next_word())
            line = linechars.rest().strip()
            chars = ''
            if self.interleave:
                if line:
                    chars = ''.join(line.split())
                else:
                    chars = ''.join(next(lineiter).split())
            else:
                chars = ''.join(line.split())
                while len(chars) < self.nchar:
                    line = next(lineiter)
                    chars += ''.join(line.split())
            if self.datatype != 'standard':
                iupac_seq = Seq(_replace_parenthesized_ambigs(chars, self.rev_ambiguous_values))
                if taxcount == 1:
                    refseq = iupac_seq
                elif self.matchchar:
                    while True:
                        p = iupac_seq.find(self.matchchar)
                        if p == -1:
                            break
                        iupac_seq = Seq(iupac_seq[:p] + refseq[p] + iupac_seq[p + 1:])
                for (i, c) in enumerate(iupac_seq):
                    if c not in self.valid_characters and c != self.gap and (c != self.missing):
                        raise NexusError('Taxon %s: Illegal character %s in sequence %s (check dimensions/interleaving)' % (id, c, iupac_seq))
            else:
                iupac_seq = StandardData(chars)
                for (i, c) in enumerate(iupac_seq):
                    for coding in c['d']:
                        if coding not in self.valid_characters:
                            if coding != self.gap and coding != self.missing:
                                raise NexusError('Taxon %s: Illegal character %s in sequence %s (check dimensions/interleaving)' % (id, coding, iupac_seq))
            if first_matrix_block:
                self.unaltered_taxlabels.append(id)
                id = _unique_label(list(self.matrix.keys()), id)
                self.matrix[id] = iupac_seq
                self.taxlabels.append(id)
            else:
                id = _unique_label(self.taxlabels[:taxcount - 1], id)
                taxon_present = self._check_taxlabels(id)
                if taxon_present:
                    self.matrix[taxon_present] += iupac_seq
                else:
                    raise NexusError('Taxon %s not in first block of interleaved matrix. Check matrix dimensions and interleave.' % id)
        for taxon in self.matrix:
            if len(self.matrix[taxon]) != self.nchar:
                raise NexusError('Matrix Nchar %d does not match data length (%d) for taxon %s' % (self.nchar, len(self.matrix[taxon]), taxon))
        matrixkeys = sorted(self.matrix)
        taxlabelssort = sorted(self.taxlabels[:])
        if matrixkeys != taxlabelssort:
            raise ValueError('ERROR: TAXLABELS must be identical with MATRIX. Please Report this as a bug, and send in data file.')

    def _translate(self, options):
        if False:
            i = 10
            return i + 15
        'Translate a Nexus file (PRIVATE).'
        self.translate = {}
        opts = CharBuffer(options)
        while True:
            try:
                identifier = int(opts.next_word())
                label = quotestrip(opts.next_word())
                self.translate[identifier] = label
                c = opts.next_nonwhitespace()
                if c is None:
                    break
                elif c != ',':
                    raise NexusError(f"Missing ',' in line {options}.")
            except NexusError:
                raise
            except Exception:
                raise NexusError(f'Format error in line {options}.') from None

    def _utree(self, options):
        if False:
            for i in range(10):
                print('nop')
        "Use 'utree' to denote an unrooted tree (ex: clustalx) (PRIVATE)."
        self._tree(options)

    def _tree(self, options):
        if False:
            print('Hello World!')
        opts = CharBuffer(options)
        if opts.peek_nonwhitespace() == '*':
            dummy = opts.next_nonwhitespace()
        name = opts.next_word()
        if opts.next_nonwhitespace() != '=':
            raise NexusError(f'Syntax error in tree description: {options[:50]}')
        rooted = False
        weight = 1.0
        while opts.peek_nonwhitespace() == '[':
            opts.next_nonwhitespace()
            symbol = next(opts)
            if symbol != '&':
                raise NexusError('Illegal special comment [%s...] in tree description: %s' % (symbol, options[:50]))
            special = next(opts)
            value = opts.next_until(']')
            next(opts)
            if special == 'R':
                rooted = True
            elif special == 'U':
                rooted = False
            elif special == 'W':
                weight = float(value)
        tree = Tree(name=name, weight=weight, rooted=rooted, tree=opts.rest().strip())
        if self.translate:
            for n in tree.get_terminals():
                try:
                    tree.node(n).data.taxon = safename(self.translate[int(tree.node(n).data.taxon)])
                except (ValueError, KeyError):
                    raise NexusError("Unable to substitute %s using 'translate' data." % tree.node(n).data.taxon) from None
        self.trees.append(tree)

    def _apply_block_structure(self, title, lines):
        if False:
            for i in range(10):
                print('nop')
        'Apply Block structure to the NEXUS file (PRIVATE).'
        block = Block('')
        block.title = title
        for line in lines:
            block.commandlines.append(Commandline(line, title))
        self.structured.append(block)

    def _taxset(self, options):
        if False:
            while True:
                i = 10
        'Create unique taxset (PRIVATE).'
        (name, taxa) = self._get_indices(options, set_type=TAXSET)
        self.taxsets[name] = _make_unique(taxa)

    def _charset(self, options):
        if False:
            print('Hello World!')
        'Create unique character set (PRIVATE).'
        (name, sites) = self._get_indices(options, set_type=CHARSET)
        self.charsets[name] = _make_unique(sites)

    def _taxpartition(self, options):
        if False:
            i = 10
            return i + 15
        'Collect taxpartition from a NEXUS file (PRIVATE).'
        taxpartition = {}
        quotelevel = False
        opts = CharBuffer(options)
        name = self._name_n_vector(opts)
        if not name:
            raise NexusError(f'Formatting error in taxpartition: {options} ')
        sub = ''
        while True:
            w = next(opts)
            if w is None or (w == ',' and (not quotelevel)):
                (subname, subindices) = self._get_indices(sub, set_type=TAXSET, separator=':')
                taxpartition[subname] = _make_unique(subindices)
                sub = ''
                if w is None:
                    break
            else:
                if w == "'":
                    quotelevel = not quotelevel
                sub += w
        self.taxpartitions[name] = taxpartition

    def _codonposset(self, options):
        if False:
            while True:
                i = 10
        'Read codon positions from a codons block as written from McClade (PRIVATE).\n\n        Here codonposset is just a fancy name for a character partition with\n        the name CodonPositions and the partitions N,1,2,3\n        '
        prev_partitions = list(self.charpartitions.keys())
        self._charpartition(options)
        codonname = [n for n in self.charpartitions if n not in prev_partitions]
        if codonname == [] or len(codonname) > 1:
            raise NexusError(f'Formatting Error in codonposset: {options} ')
        else:
            self.codonposset = codonname[0]

    def _codeset(self, options):
        if False:
            while True:
                i = 10
        pass

    def _charpartition(self, options):
        if False:
            while True:
                i = 10
        'Collect character partition from NEXUS file (PRIVATE).'
        charpartition = {}
        quotelevel = False
        opts = CharBuffer(options)
        name = self._name_n_vector(opts)
        if not name:
            raise NexusError(f'Formatting error in charpartition: {options} ')
        sub = ''
        while True:
            w = next(opts)
            if w is None or (w == ',' and (not quotelevel)):
                (subname, subindices) = self._get_indices(sub, set_type=CHARSET, separator=':')
                charpartition[subname] = _make_unique(subindices)
                sub = ''
                if w is None:
                    break
            else:
                if w == "'":
                    quotelevel = not quotelevel
                sub += w
        self.charpartitions[name] = charpartition

    def _get_indices(self, options, set_type=CHARSET, separator='='):
        if False:
            for i in range(10):
                print('nop')
        "Parse the taxset/charset specification (PRIVATE).\n\n        e.g. '1 2   3 - 5 dog cat   10 - 20 \\\\ 3'\n        --> [0,1,2,3,4,'dog','cat',9,12,15,18]\n        "
        opts = CharBuffer(options)
        name = self._name_n_vector(opts, separator=separator)
        indices = self._parse_list(opts, set_type=set_type)
        if indices is None:
            raise NexusError(f'Formatting error in line: {options} ')
        return (name, indices)

    def _name_n_vector(self, opts, separator='='):
        if False:
            while True:
                i = 10
        "Extract name and check that it's not in vector format (PRIVATE)."
        rest = opts.rest()
        name = opts.next_word()
        if name == '*':
            name = opts.next_word()
        if not name:
            raise NexusError(f'Formatting error in line: {rest} ')
        name = quotestrip(name)
        if opts.peek_nonwhitespace == '(':
            open = opts.next_nonwhitespace()
            qualifier = open.next_word()
            close = opts.next_nonwhitespace()
            if qualifier.lower() == 'vector':
                raise NexusError(f'Unsupported VECTOR format in line {opts}')
            elif qualifier.lower() != 'standard':
                raise NexusError(f'Unknown qualifier {qualifier} in line {opts}')
        if opts.next_nonwhitespace() != separator:
            raise NexusError(f'Formatting error in line: {rest} ')
        return name

    def _parse_list(self, options_buffer, set_type):
        if False:
            for i in range(10):
                print('nop')
        'Parse a NEXUS list (PRIVATE).\n\n        e.g. [1, 2, 4-8\\\\2, dog, cat] --> [1,2,4,6,8,17,21],\n        (assuming dog is taxon no. 17 and cat is taxon no. 21).\n        '
        plain_list = []
        if options_buffer.peek_nonwhitespace():
            try:
                while True:
                    identifier = options_buffer.next_word()
                    if not identifier:
                        break
                    start = self._resolve(identifier, set_type=set_type)
                    if options_buffer.peek_nonwhitespace() == '-':
                        end = start
                        step = 1
                        hyphen = options_buffer.next_nonwhitespace()
                        end = self._resolve(options_buffer.next_word(), set_type=set_type)
                        if set_type == CHARSET:
                            if options_buffer.peek_nonwhitespace() == '\\':
                                backslash = options_buffer.next_nonwhitespace()
                                step = int(options_buffer.next_word())
                            plain_list.extend(range(start, end + 1, step))
                        else:
                            if isinstance(start, list) or isinstance(end, list):
                                raise NexusError('Name if character sets not allowed in range definition: %s' % identifier)
                            start = self.taxlabels.index(start)
                            end = self.taxlabels.index(end)
                            taxrange = self.taxlabels[start:end + 1]
                            plain_list.extend(taxrange)
                    elif isinstance(start, list):
                        plain_list.extend(start)
                    else:
                        plain_list.append(start)
            except NexusError:
                raise
            except Exception:
                return None
        return plain_list

    def _resolve(self, identifier, set_type=None):
        if False:
            print('Hello World!')
        "Translate identifier in list into character/taxon index (PRIVATE).\n\n        Characters (which are referred to by their index in Nexus.py):\n            Plain numbers are returned minus 1 (Nexus indices to python indices)\n            Text identifiers are translated into their indices (if plain character identifiers),\n            the first hit in charlabels is returned (charlabels don't need to be unique)\n            or the range of indices is returned (if names of character sets).\n        Taxa (which are referred to by their unique name in Nexus.py):\n            Plain numbers are translated in their taxon name, underscores and spaces are considered equal.\n            Names are returned unchanged (if plain taxon identifiers), or the names in\n            the corresponding taxon set is returned.\n\n        "
        identifier = quotestrip(identifier)
        if not set_type:
            raise NexusError('INTERNAL ERROR: Need type to resolve identifier.')
        if set_type == CHARSET:
            try:
                n = int(identifier)
            except ValueError:
                if self.charlabels and identifier in self.charlabels.values():
                    for k in self.charlabels:
                        if self.charlabels[k] == identifier:
                            return k
                elif self.charsets and identifier in self.charsets:
                    return self.charsets[identifier]
                else:
                    raise NexusError(f'Unknown character identifier: {identifier}') from None
            else:
                if n <= self.nchar:
                    return n - 1
                else:
                    raise NexusError('Illegal character identifier: %d>nchar (=%d).' % (identifier, self.nchar))
        elif set_type == TAXSET:
            try:
                n = int(identifier)
            except ValueError:
                taxlabels_id = self._check_taxlabels(identifier)
                if taxlabels_id:
                    return taxlabels_id
                elif self.taxsets and identifier in self.taxsets:
                    return self.taxsets[identifier]
                else:
                    raise NexusError(f'Unknown taxon identifier: {identifier}') from None
            else:
                if n > 0 and n <= self.ntax:
                    return self.taxlabels[n - 1]
                else:
                    raise NexusError('Illegal taxon identifier: %d>ntax (=%d).' % (identifier, self.ntax))
        else:
            raise NexusError(f'Unknown set specification: {set_type}.')

    def _stateset(self, options):
        if False:
            print('Hello World!')
        pass

    def _changeset(self, options):
        if False:
            return 10
        pass

    def _treeset(self, options):
        if False:
            while True:
                i = 10
        pass

    def _treepartition(self, options):
        if False:
            for i in range(10):
                print('nop')
        pass

    def write_nexus_data_partitions(self, matrix=None, filename=None, blocksize=None, interleave=False, exclude=(), delete=(), charpartition=None, comment='', mrbayes=False):
        if False:
            while True:
                i = 10
        'Write a nexus file for each partition in charpartition.\n\n        Only non-excluded characters and non-deleted taxa are included,\n        just the data block is written.\n        '
        if not matrix:
            matrix = self.matrix
        if not matrix:
            return
        if not filename:
            filename = self.filename
        if charpartition:
            pfilenames = {}
            for p in charpartition:
                total_exclude = list(exclude)
                total_exclude.extend((c for c in range(self.nchar) if c not in charpartition[p]))
                total_exclude = _make_unique(total_exclude)
                pcomment = comment + '\nPartition: ' + p + '\n'
                dot = filename.rfind('.')
                if dot > 0:
                    pfilename = filename[:dot] + '_' + p + '.data'
                else:
                    pfilename = filename + '_' + p
                pfilenames[p] = pfilename
                self.write_nexus_data(filename=pfilename, matrix=matrix, blocksize=blocksize, interleave=interleave, exclude=total_exclude, delete=delete, comment=pcomment, append_sets=False, mrbayes=mrbayes)
            return pfilenames
        else:
            fn = self.filename + '.data'
            self.write_nexus_data(filename=fn, matrix=matrix, blocksize=blocksize, interleave=interleave, exclude=exclude, delete=delete, comment=comment, append_sets=False, mrbayes=mrbayes)
            return fn

    def write_nexus_data(self, filename=None, matrix=None, exclude=(), delete=(), blocksize=None, interleave=False, interleave_by_partition=False, comment=None, omit_NEXUS=False, append_sets=True, mrbayes=False, codons_block=True):
        if False:
            i = 10
            return i + 15
        "Write a nexus file with data and sets block to a file or handle.\n\n        Character sets and partitions are appended by default, and are\n        adjusted according to excluded characters (i.e. character sets\n        still point to the same sites (not necessarily same positions),\n        without including the deleted characters.\n\n        - filename - Either a filename as a string (which will be opened,\n          written to and closed), or a handle object (which will\n          be written to but NOT closed).\n        - interleave_by_partition - Optional name of partition (string)\n        - omit_NEXUS - Boolean.  If true, the '#NEXUS' line normally at the\n          start of the file is omitted.\n\n        Returns the filename/handle used to write the data.\n        "
        if not matrix:
            matrix = self.matrix
        if not matrix:
            return
        if not filename:
            filename = self.filename
        if [t for t in delete if not self._check_taxlabels(t)]:
            raise NexusError('Unknown taxa: %s' % ', '.join(set(delete).difference(set(self.taxlabels))))
        if interleave_by_partition:
            if interleave_by_partition not in self.charpartitions:
                raise NexusError(f'Unknown partition: {interleave_by_partition!r}')
            else:
                partition = self.charpartitions[interleave_by_partition]
                names = _sort_keys_by_values(partition)
                newpartition = {}
                for p in partition:
                    newpartition[p] = [c for c in partition[p] if c not in exclude]
        undelete = [taxon for taxon in self.taxlabels if taxon in matrix and taxon not in delete]
        cropped_matrix = _seqmatrix2strmatrix(self.crop_matrix(matrix, exclude=exclude, delete=delete))
        ntax_adjusted = len(undelete)
        nchar_adjusted = len(cropped_matrix[undelete[0]])
        if not undelete or (undelete and undelete[0] == ''):
            return
        with File.as_handle(filename, mode='w') as fh:
            if not omit_NEXUS:
                fh.write('#NEXUS\n')
            if comment:
                fh.write('[' + comment + ']\n')
            fh.write('begin data;\n')
            fh.write('dimensions ntax=%d nchar=%d;\n' % (ntax_adjusted, nchar_adjusted))
            fh.write('format datatype=' + self.datatype)
            if self.respectcase:
                fh.write(' respectcase')
            if self.missing:
                fh.write(' missing=' + self.missing)
            if self.gap:
                fh.write(' gap=' + self.gap)
            if self.matchchar:
                fh.write(' matchchar=' + self.matchchar)
            if self.labels:
                fh.write(' labels=' + self.labels)
            if self.equate:
                fh.write(' equate=' + self.equate)
            if interleave or interleave_by_partition:
                fh.write(' interleave')
            fh.write(';\n')
            if self.charlabels:
                newcharlabels = self._adjust_charlabels(exclude=exclude)
                clkeys = sorted(newcharlabels)
                fh.write('charlabels ' + ', '.join((f'{k + 1} {safename(newcharlabels[k])}' for k in clkeys)) + ';\n')
            fh.write('matrix\n')
            if not blocksize:
                if interleave:
                    blocksize = 70
                else:
                    blocksize = self.nchar
            namelength = max((len(safename(t, mrbayes=mrbayes)) for t in undelete))
            if interleave_by_partition:
                seek = 0
                for p in names:
                    fh.write(f'[{interleave_by_partition}: {p}]\n')
                    if len(newpartition[p]) > 0:
                        for taxon in undelete:
                            fh.write(safename(taxon, mrbayes=mrbayes).ljust(namelength + 1))
                            fh.write(cropped_matrix[taxon][seek:seek + len(newpartition[p])] + '\n')
                        fh.write('\n')
                    else:
                        fh.write('[empty]\n\n')
                    seek += len(newpartition[p])
            elif interleave:
                for seek in range(0, nchar_adjusted, blocksize):
                    for taxon in undelete:
                        fh.write(safename(taxon, mrbayes=mrbayes).ljust(namelength + 1))
                        fh.write(cropped_matrix[taxon][seek:seek + blocksize] + '\n')
                    fh.write('\n')
            else:
                for taxon in undelete:
                    if blocksize < nchar_adjusted:
                        fh.write(safename(taxon, mrbayes=mrbayes) + '\n')
                    else:
                        fh.write(safename(taxon, mrbayes=mrbayes).ljust(namelength + 1))
                    taxon_seq = cropped_matrix[taxon]
                    for seek in range(0, nchar_adjusted, blocksize):
                        fh.write(taxon_seq[seek:seek + blocksize] + '\n')
                    del taxon_seq
            fh.write(';\nend;\n')
            if append_sets:
                if codons_block:
                    fh.write(self.append_sets(exclude=exclude, delete=delete, mrbayes=mrbayes, include_codons=False))
                    fh.write(self.append_sets(exclude=exclude, delete=delete, mrbayes=mrbayes, codons_only=True))
                else:
                    fh.write(self.append_sets(exclude=exclude, delete=delete, mrbayes=mrbayes))
        return filename

    def append_sets(self, exclude=(), delete=(), mrbayes=False, include_codons=True, codons_only=False):
        if False:
            return 10
        'Return a sets block.'
        if not self.charsets and (not self.taxsets) and (not self.charpartitions):
            return ''
        if codons_only:
            setsb = ['\nbegin codons']
        else:
            setsb = ['\nbegin sets']
        offset = 0
        offlist = []
        for c in range(self.nchar):
            if c in exclude:
                offset += 1
                offlist.append(-1)
            else:
                offlist.append(c - offset)
        if not codons_only:
            for (n, ns) in self.charsets.items():
                cset = [offlist[c] for c in ns if c not in exclude]
                if cset:
                    setsb.append(f'charset {safename(n)} = {_compact4nexus(cset)}')
            for (n, s) in self.taxsets.items():
                tset = [safename(t, mrbayes=mrbayes) for t in s if t not in delete]
                if tset:
                    setsb.append(f"taxset {safename(n)} = {' '.join(tset)}")
        for (n, p) in self.charpartitions.items():
            if not include_codons and n == CODONPOSITIONS:
                continue
            elif codons_only and n != CODONPOSITIONS:
                continue
            names = _sort_keys_by_values(p)
            newpartition = {}
            for sn in names:
                nsp = [offlist[c] for c in p[sn] if c not in exclude]
                if nsp:
                    newpartition[sn] = nsp
            if newpartition:
                if include_codons and n == CODONPOSITIONS:
                    command = 'codonposset'
                else:
                    command = 'charpartition'
                setsb.append('%s %s = %s' % (command, safename(n), ', '.join((f'{sn}: {_compact4nexus(newpartition[sn])}' for sn in names if sn in newpartition))))
        for (n, p) in self.taxpartitions.items():
            names = _sort_keys_by_values(p)
            newpartition = {}
            for sn in names:
                nsp = [t for t in p[sn] if t not in delete]
                if nsp:
                    newpartition[sn] = nsp
            if newpartition:
                setsb.append('taxpartition %s = %s' % (safename(n), ', '.join(('%s: %s' % (safename(sn), ' '.join((safename(x) for x in newpartition[sn]))) for sn in names if sn in newpartition))))
        setsb.append('end;\n')
        if len(setsb) == 2:
            return ''
        else:
            return ';\n'.join(setsb)

    def export_fasta(self, filename=None, width=70):
        if False:
            for i in range(10):
                print('nop')
        'Write matrix into a fasta file.'
        if not filename:
            if '.' in self.filename and self.filename.split('.')[-1].lower() in ['paup', 'nexus', 'nex', 'dat']:
                filename = '.'.join(self.filename.split('.')[:-1]) + '.fas'
            else:
                filename = self.filename + '.fas'
        with open(filename, 'w') as fh:
            for taxon in self.taxlabels:
                fh.write('>' + safename(taxon) + '\n')
                for i in range(0, len(str(self.matrix[taxon])), width):
                    fh.write(str(self.matrix[taxon])[i:i + width] + '\n')
        return filename

    def export_phylip(self, filename=None):
        if False:
            for i in range(10):
                print('nop')
        'Write matrix into a PHYLIP file.\n\n        Note that this writes a relaxed PHYLIP format file, where the names\n        are not truncated, nor checked for invalid characters.\n        '
        if not filename:
            if '.' in self.filename and self.filename.split('.')[-1].lower() in ['paup', 'nexus', 'nex', 'dat']:
                filename = '.'.join(self.filename.split('.')[:-1]) + '.phy'
            else:
                filename = self.filename + '.phy'
        with open(filename, 'w') as fh:
            fh.write('%d %d\n' % (self.ntax, self.nchar))
            for taxon in self.taxlabels:
                fh.write(f'{safename(taxon)} {self.matrix[taxon]!s}\n')
        return filename

    def constant(self, matrix=None, delete=(), exclude=()):
        if False:
            i = 10
            return i + 15
        'Return a list with all constant characters.'
        if not matrix:
            matrix = self.matrix
        undelete = [t for t in self.taxlabels if t in matrix and t not in delete]
        if not undelete:
            return None
        elif len(undelete) == 1:
            return [x for x in range(len(matrix[undelete[0]])) if x not in exclude]
        constant = [(x, self.ambiguous_values.get(n.upper(), n.upper())) for (x, n) in enumerate(str(matrix[undelete[0]])) if x not in exclude]
        for taxon in undelete[1:]:
            newconstant = []
            for site in constant:
                seqsite = matrix[taxon][site[0]].upper()
                if seqsite == self.missing or (seqsite == self.gap and self.options['gapmode'].lower() == 'missing') or seqsite == site[1]:
                    newconstant.append(site)
                elif seqsite in site[1] or site[1] == self.missing or (self.options['gapmode'].lower() == 'missing' and site[1] == self.gap):
                    newconstant.append((site[0], self.ambiguous_values.get(seqsite, seqsite)))
                elif seqsite in self.ambiguous_values:
                    intersect = set(self.ambiguous_values[seqsite]).intersection(set(site[1]))
                    if intersect:
                        newconstant.append((site[0], ''.join(intersect)))
            constant = newconstant
        cpos = [s[0] for s in constant]
        return cpos

    def cstatus(self, site, delete=(), narrow=True):
        if False:
            print('Hello World!')
        'Summarize character.\n\n        narrow=True:  paup-mode (a c ? --> ac; ? ? ? --> ?)\n        narrow=false:           (a c ? --> a c g t -; ? ? ? --> a c g t -)\n        '
        undelete = [t for t in self.taxlabels if t not in delete]
        if not undelete:
            return None
        cstatus = []
        for t in undelete:
            c = self.matrix[t][site].upper()
            if self.options.get('gapmode') == 'missing' and c == self.gap:
                c = self.missing
            if narrow and c == self.missing:
                if c not in cstatus:
                    cstatus.append(c)
            else:
                cstatus.extend((b for b in self.ambiguous_values[c] if b not in cstatus))
        if self.missing in cstatus and narrow and (len(cstatus) > 1):
            cstatus = [_ for _ in cstatus if _ != self.missing]
        cstatus.sort()
        return cstatus

    def weighted_stepmatrix(self, name='your_name_here', exclude=(), delete=()):
        if False:
            while True:
                i = 10
        'Calculate a stepmatrix for weighted parsimony.\n\n        See Wheeler (1990), Cladistics 6:269-275 and\n        Felsenstein (1981), Biol. J. Linn. Soc. 16:183-196\n        '
        m = StepMatrix(self.unambiguous_letters, self.gap)
        for site in [s for s in range(self.nchar) if s not in exclude]:
            cstatus = self.cstatus(site, delete)
            for (i, b1) in enumerate(cstatus[:-1]):
                for b2 in cstatus[i + 1:]:
                    m.add(b1.upper(), b2.upper(), 1)
        return m.transformation().weighting().smprint(name=name)

    def crop_matrix(self, matrix=None, delete=(), exclude=()):
        if False:
            for i in range(10):
                print('nop')
        'Return a matrix without deleted taxa and excluded characters.'
        if not matrix:
            matrix = self.matrix
        if [t for t in delete if not self._check_taxlabels(t)]:
            raise NexusError(f"Unknown taxa: {', '.join(set(delete).difference(self.taxlabels))}")
        if exclude != []:
            undelete = [t for t in self.taxlabels if t in matrix and t not in delete]
            if not undelete:
                return {}
            m = [str(matrix[k]) for k in undelete]
            sitesm = [s for (i, s) in enumerate(zip(*m)) if i not in exclude]
            if sitesm == []:
                return {t: Seq('') for t in undelete}
            else:
                m = [Seq(s) for s in (''.join(x) for x in zip(*sitesm))]
                return dict(zip(undelete, m))
        else:
            return {t: matrix[t] for t in self.taxlabels if t in matrix and t not in delete}

    def bootstrap(self, matrix=None, delete=(), exclude=()):
        if False:
            return 10
        'Return a bootstrapped matrix.'
        if not matrix:
            matrix = self.matrix
        seqobjects = isinstance(matrix[list(matrix.keys())[0]], Seq)
        cm = self.crop_matrix(delete=delete, exclude=exclude)
        if not cm:
            return {}
        elif not cm[list(cm.keys())[0]]:
            return cm
        undelete = [t for t in self.taxlabels if t in cm]
        if seqobjects:
            sitesm = list(zip(*(str(cm[t]) for t in undelete)))
        else:
            sitesm = list(zip(*(cm[t] for t in undelete)))
        bootstrapsitesm = [sitesm[random.randint(0, len(sitesm) - 1)] for _ in range(len(sitesm))]
        bootstrapseqs = [''.join(x) for x in zip(*bootstrapsitesm)]
        if seqobjects:
            bootstrapseqs = [Seq(s) for s in bootstrapseqs]
        return dict(zip(undelete, bootstrapseqs))

    def add_sequence(self, name, sequence):
        if False:
            i = 10
            return i + 15
        'Add a sequence (string) to the matrix.'
        if not name:
            raise NexusError('New sequence must have a name')
        diff = self.nchar - len(sequence)
        if diff < 0:
            self.insert_gap(self.nchar, -diff)
        elif diff > 0:
            sequence += self.missing * diff
        if name in self.taxlabels:
            unique_name = _unique_label(self.taxlabels, name)
        else:
            unique_name = name
        if unique_name in self.matrix:
            raise ValueError('ERROR. There is a discrepancy between taxlabels and matrix keys. Report this as a bug.')
        self.matrix[unique_name] = Seq(sequence)
        self.ntax += 1
        self.taxlabels.append(unique_name)
        self.unaltered_taxlabels.append(name)

    def insert_gap(self, pos, n=1, leftgreedy=False):
        if False:
            while True:
                i = 10
        'Add a gap into the matrix and adjust charsets and partitions.\n\n        pos=0: first position\n        pos=nchar: last position\n        '

        def _adjust(set, x, d, leftgreedy=False):
            if False:
                i = 10
                return i + 15
            'Adjust character sets if gaps are inserted (PRIVATE).\n\n            Takes care of new gaps within a coherent character set.\n            '
            set.sort()
            addpos = 0
            for (i, c) in enumerate(set):
                if c >= x:
                    set[i] = c + d
                if c == x:
                    if leftgreedy or (i > 0 and set[i - 1] == c - 1):
                        addpos = i
            if addpos > 0:
                set[addpos:addpos] = list(range(x, x + d))
            return set
        if pos < 0 or pos > self.nchar:
            raise NexusError('Illegal gap position: %d' % pos)
        if n == 0:
            return
        sitesm = list(zip(*(str(self.matrix[t]) for t in self.taxlabels)))
        sitesm[pos:pos] = [['-'] * len(self.taxlabels)] * n
        mapped = [''.join(x) for x in zip(*sitesm)]
        listed = [(taxon, Seq(mapped[i])) for (i, taxon) in enumerate(self.taxlabels)]
        self.matrix = dict(listed)
        self.nchar += n
        for (i, s) in self.charsets.items():
            self.charsets[i] = _adjust(s, pos, n, leftgreedy=leftgreedy)
        for p in self.charpartitions:
            for (sp, s) in self.charpartitions[p].items():
                self.charpartitions[p][sp] = _adjust(s, pos, n, leftgreedy=leftgreedy)
        self.charlabels = self._adjust_charlabels(insert=[pos] * n)
        return self.charlabels

    def _adjust_charlabels(self, exclude=None, insert=None):
        if False:
            for i in range(10):
                print('nop')
        'Return adjusted indices of self.charlabels if characters are excluded or inserted (PRIVATE).'
        if exclude and insert:
            raise NexusError("Can't exclude and insert at the same time")
        if not self.charlabels:
            return None
        labels = sorted(self.charlabels)
        newcharlabels = {}
        if exclude:
            exclude.sort()
            exclude.append(sys.maxsize)
            excount = 0
            for c in labels:
                if c not in exclude:
                    while c > exclude[excount]:
                        excount += 1
                    newcharlabels[c - excount] = self.charlabels[c]
        elif insert:
            insert.sort()
            insert.append(sys.maxsize)
            icount = 0
            for c in labels:
                while c >= insert[icount]:
                    icount += 1
                newcharlabels[c + icount] = self.charlabels[c]
        else:
            return self.charlabels
        return newcharlabels

    def invert(self, charlist):
        if False:
            while True:
                i = 10
        'Return all character indices that are not in charlist.'
        return [c for c in range(self.nchar) if c not in charlist]

    def gaponly(self, include_missing=False):
        if False:
            while True:
                i = 10
        'Return gap-only sites.'
        gap = set(self.gap)
        if include_missing:
            gap.add(self.missing)
        sitesm = zip(*(str(self.matrix[t]) for t in self.taxlabels))
        return [i for (i, site) in enumerate(sitesm) if set(site).issubset(gap)]

    def terminal_gap_to_missing(self, missing=None, skip_n=True):
        if False:
            i = 10
            return i + 15
        'Replace all terminal gaps with missing character.\n\n        Mixtures like ???------??------- are properly resolved.\n        '
        if not missing:
            missing = self.missing
        replace = [self.missing, self.gap]
        if not skip_n:
            replace.extend(['n', 'N'])
        for taxon in self.taxlabels:
            sequence = str(self.matrix[taxon])
            length = len(sequence)
            (start, end) = get_start_end(sequence, skiplist=replace)
            if start == -1 and end == -1:
                sequence = missing * length
            else:
                sequence = sequence[:end + 1] + missing * (length - end - 1)
                sequence = start * missing + sequence[start:]
            if length != len(sequence):
                raise RuntimeError('Illegal sequence manipulation in Nexus.terminal_gap_to_missing in taxon %s' % taxon)
            self.matrix[taxon] = Seq(sequence)
try:
    from . import cnexus
except ImportError as ex:
    warnings.warn(f'Import of C module failed ({ex}). Falling back to slow Python implementation', BiopythonWarning)

    def _get_command_lines(file_contents):
        if False:
            print('Hello World!')
        lines = _kill_comments_and_break_lines(file_contents)
        commandlines = _adjust_lines(lines)
        return commandlines
else:

    def _get_command_lines(file_contents):
        if False:
            print('Hello World!')
        decommented = cnexus.scanfile(file_contents)
        if decommented == '[' or decommented == ']':
            raise NexusError(f'Unmatched {decommented}')
        commandlines = _adjust_lines(decommented.split(chr(7)))
        return commandlines
if __name__ == '__main__':
    from Bio._utils import run_doctest
    run_doctest()