"""I/O function wrappers for the Newick file format.

See: http://evolution.genetics.washington.edu/phylip/newick_doc.html
"""
import re
from io import StringIO
from Bio.Phylo import Newick

class NewickError(Exception):
    """Exception raised when Newick object construction cannot continue."""
tokens = [('\\(', 'open parens'), ('\\)', 'close parens'), ("[^\\s\\(\\)\\[\\]\\'\\:\\;\\,]+", 'unquoted node label'), ('\\:\\ ?[+-]?[0-9]*\\.?[0-9]+([eE][+-]?[0-9]+)?', 'edge length'), ('\\,', 'comma'), ('\\[(\\\\.|[^\\]])*\\]', 'comment'), ("\\'(\\\\.|[^\\'])*\\'", 'quoted node label'), ('\\;', 'semicolon'), ('\\n', 'newline')]
tokenizer = re.compile(f"({'|'.join((token[0] for token in tokens))})")
token_dict = {name: re.compile(token) for (token, name) in tokens}

def parse(handle, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Iterate over the trees in a Newick file handle.\n\n    :returns: generator of Bio.Phylo.Newick.Tree objects.\n\n    '
    return Parser(handle).parse(**kwargs)

def write(trees, handle, plain=False, **kwargs):
    if False:
        i = 10
        return i + 15
    'Write a trees in Newick format to the given file handle.\n\n    :returns: number of trees written.\n\n    '
    return Writer(trees).write(handle, plain=plain, **kwargs)

def _parse_confidence(text):
    if False:
        for i in range(10):
            print('nop')
    if text.isdigit():
        return int(text)
    try:
        return float(text)
    except ValueError:
        return None

def _format_comment(text):
    if False:
        print('Hello World!')
    return '[%s]' % text.replace('[', '\\[').replace(']', '\\]')

def _get_comment(clade):
    if False:
        return 10
    try:
        comment = clade.comment
    except AttributeError:
        pass
    else:
        if comment:
            return _format_comment(str(comment))
    return ''

class Parser:
    """Parse a Newick tree given a file handle.

    Based on the parser in ``Bio.Nexus.Trees``.
    """

    def __init__(self, handle):
        if False:
            return 10
        'Initialize file handle for the Newick Tree.'
        if handle.read(0) != '':
            raise ValueError('Newick files must be opened in text mode') from None
        self.handle = handle

    @classmethod
    def from_string(cls, treetext):
        if False:
            i = 10
            return i + 15
        'Instantiate the Newick Tree class from the given string.'
        handle = StringIO(treetext)
        return cls(handle)

    def parse(self, values_are_confidence=False, comments_are_confidence=False, rooted=False):
        if False:
            for i in range(10):
                print('nop')
        'Parse the text stream this object was initialized with.'
        self.values_are_confidence = values_are_confidence
        self.comments_are_confidence = comments_are_confidence
        self.rooted = rooted
        buf = ''
        for line in self.handle:
            buf += line.rstrip()
            if buf.endswith(';'):
                yield self._parse_tree(buf)
                buf = ''
        if buf:
            yield self._parse_tree(buf)

    def _parse_tree(self, text):
        if False:
            for i in range(10):
                print('nop')
        'Parse the text representation into an Tree object (PRIVATE).'
        tokens = re.finditer(tokenizer, text.strip())
        new_clade = self.new_clade
        root_clade = new_clade()
        current_clade = root_clade
        entering_branch_length = False
        lp_count = 0
        rp_count = 0
        for match in tokens:
            token = match.group()
            if token.startswith("'"):
                current_clade.name = token[1:-1]
            elif token.startswith('['):
                current_clade.comment = token[1:-1]
                if self.comments_are_confidence:
                    current_clade.confidence = _parse_confidence(current_clade.comment)
            elif token == '(':
                current_clade = new_clade(current_clade)
                entering_branch_length = False
                lp_count += 1
            elif token == ',':
                if current_clade is root_clade:
                    root_clade = new_clade()
                    current_clade.parent = root_clade
                parent = self.process_clade(current_clade)
                current_clade = new_clade(parent)
                entering_branch_length = False
            elif token == ')':
                parent = self.process_clade(current_clade)
                if not parent:
                    raise NewickError('Parenthesis mismatch.')
                current_clade = parent
                entering_branch_length = False
                rp_count += 1
            elif token == ';':
                break
            elif token.startswith(':'):
                value = float(token[1:])
                if self.values_are_confidence:
                    current_clade.confidence = value
                else:
                    current_clade.branch_length = value
            elif token == '\n':
                pass
            else:
                current_clade.name = token
        if lp_count != rp_count:
            raise NewickError(f'Mismatch, {lp_count} open vs {rp_count} close parentheses.')
        try:
            next_token = next(tokens)
            raise NewickError(f'Text after semicolon in Newick tree: {next_token.group()}')
        except StopIteration:
            pass
        self.process_clade(current_clade)
        self.process_clade(root_clade)
        return Newick.Tree(root=root_clade, rooted=self.rooted)

    def new_clade(self, parent=None):
        if False:
            print('Hello World!')
        'Return new Newick.Clade, optionally with temporary reference to parent.'
        clade = Newick.Clade()
        if parent:
            clade.parent = parent
        return clade

    def process_clade(self, clade):
        if False:
            i = 10
            return i + 15
        "Remove node's parent and return it. Final processing of parsed clade."
        if clade.name and (not (self.values_are_confidence or self.comments_are_confidence)) and (clade.confidence is None) and clade.clades:
            clade.confidence = _parse_confidence(clade.name)
            if clade.confidence is not None:
                clade.name = None
        try:
            parent = clade.parent
        except AttributeError:
            pass
        else:
            parent.clades.append(clade)
            del clade.parent
            return parent

class Writer:
    """Based on the writer in Bio.Nexus.Trees (str, to_string)."""

    def __init__(self, trees):
        if False:
            while True:
                i = 10
        'Initialize parameter for Tree Writer object.'
        self.trees = trees

    def write(self, handle, **kwargs):
        if False:
            print('Hello World!')
        "Write this instance's trees to a file handle."
        count = 0
        for treestr in self.to_strings(**kwargs):
            handle.write(treestr + '\n')
            count += 1
        return count

    def to_strings(self, confidence_as_branch_length=False, branch_length_only=False, plain=False, plain_newick=True, ladderize=None, max_confidence=1.0, format_confidence='%1.2f', format_branch_length='%1.5f'):
        if False:
            return 10
        'Return an iterable of PAUP-compatible tree lines.'
        if confidence_as_branch_length or branch_length_only:
            plain = False
        make_info_string = self._info_factory(plain, confidence_as_branch_length, branch_length_only, max_confidence, format_confidence, format_branch_length)

        def newickize(clade):
            if False:
                while True:
                    i = 10
            'Convert a node tree to a Newick tree string, recursively.'
            label = clade.name or ''
            if label:
                unquoted_label = re.match(token_dict['unquoted node label'], label)
                if not unquoted_label or unquoted_label.end() < len(label):
                    label = "'%s'" % label.replace('\\', '\\\\').replace("'", "\\'")
            if clade.is_terminal():
                return label + make_info_string(clade, terminal=True)
            else:
                subtrees = (newickize(sub) for sub in clade)
                return f"({','.join(subtrees)}){label + make_info_string(clade)}"
        for tree in self.trees:
            if ladderize in ('left', 'LEFT', 'right', 'RIGHT'):
                tree.ladderize(reverse=ladderize in ('right', 'RIGHT'))
            rawtree = newickize(tree.root) + ';'
            if plain_newick:
                yield rawtree
                continue
            treeline = ['tree', tree.name or 'a_tree', '=']
            if tree.weight != 1:
                treeline.append(f'[&W{round(float(tree.weight), 3)}]')
            if tree.rooted:
                treeline.append('[&R]')
            treeline.append(rawtree)
            yield ' '.join(treeline)

    def _info_factory(self, plain, confidence_as_branch_length, branch_length_only, max_confidence, format_confidence, format_branch_length):
        if False:
            for i in range(10):
                print('nop')
        'Return a function that creates a nicely formatted node tag (PRIVATE).'
        if plain:

            def make_info_string(clade, terminal=False):
                if False:
                    print('Hello World!')
                return _get_comment(clade)
        elif confidence_as_branch_length:

            def make_info_string(clade, terminal=False):
                if False:
                    while True:
                        i = 10
                if terminal:
                    return ':' + format_confidence % max_confidence + _get_comment(clade)
                else:
                    return ':' + format_confidence % clade.confidence + _get_comment(clade)
        elif branch_length_only:

            def make_info_string(clade, terminal=False):
                if False:
                    i = 10
                    return i + 15
                return ':' + format_branch_length % clade.branch_length + _get_comment(clade)
        else:

            def make_info_string(clade, terminal=False):
                if False:
                    for i in range(10):
                        print('nop')
                if terminal or not hasattr(clade, 'confidence') or clade.confidence is None:
                    return (':' + format_branch_length) % (clade.branch_length or 0.0) + _get_comment(clade)
                else:
                    return (format_confidence + ':' + format_branch_length) % (clade.confidence, clade.branch_length or 0.0) + _get_comment(clade)
        return make_info_string