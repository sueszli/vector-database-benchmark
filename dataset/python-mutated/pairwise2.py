"""Pairwise sequence alignment using a dynamic programming algorithm.

This provides functions to get global and local alignments between two
sequences. A global alignment finds the best concordance between all
characters in two sequences. A local alignment finds just the
subsequences that align the best. Local alignments must have a positive
score to be reported and they will not be extended for 'zero counting'
matches. This means a local alignment will always start and end with
a positive counting match.

When doing alignments, you can specify the match score and gap
penalties.  The match score indicates the compatibility between an
alignment of two characters in the sequences. Highly compatible
characters should be given positive scores, and incompatible ones
should be given negative scores or 0.  The gap penalties should be
negative.

The names of the alignment functions in this module follow the
convention
<alignment type>XX
where <alignment type> is either "global" or "local" and XX is a 2
character code indicating the parameters it takes.  The first
character indicates the parameters for matches (and mismatches), and
the second indicates the parameters for gap penalties.

The match parameters are::

    CODE  DESCRIPTION & OPTIONAL KEYWORDS
    x     No parameters. Identical characters have score of 1, otherwise 0.
    m     A match score is the score of identical chars, otherwise mismatch
          score. Keywords ``match``, ``mismatch``.
    d     A dictionary returns the score of any pair of characters.
          Keyword ``match_dict``.
    c     A callback function returns scores. Keyword ``match_fn``.

The gap penalty parameters are::

    CODE  DESCRIPTION & OPTIONAL KEYWORDS
    x     No gap penalties.
    s     Same open and extend gap penalties for both sequences.
          Keywords ``open``, ``extend``.
    d     The sequences have different open and extend gap penalties.
          Keywords ``openA``, ``extendA``, ``openB``, ``extendB``.
    c     A callback function returns the gap penalties.
          Keywords ``gap_A_fn``, ``gap_B_fn``.

All the different alignment functions are contained in an object
``align``. For example:

    >>> from Bio import pairwise2
    >>> alignments = pairwise2.align.globalxx("ACCGT", "ACG")

For better readability, the required arguments can be used with optional keywords:

    >>> alignments = pairwise2.align.globalxx(sequenceA="ACCGT", sequenceB="ACG")

The result is a list of the alignments between the two strings. Each alignment
is a named tuple consisting of the two aligned sequences, the score and the
start and end positions of the alignment:

   >>> print(alignments)
   [Alignment(seqA='ACCGT', seqB='A-CG-', score=3.0, start=0, end=5), ...

You can access each element of an alignment by index or name:

   >>> alignments[0][2]
   3.0
   >>> alignments[0].score
   3.0

For a nice printout of an alignment, use the ``format_alignment`` method of
the module:

    >>> from Bio.pairwise2 import format_alignment
    >>> print(format_alignment(*alignments[0]))
    ACCGT
    | || 
    A-CG-
      Score=3
    <BLANKLINE>

All alignment functions have the following arguments:

- Two sequences: strings, Biopython sequence objects or lists.
  Lists are useful for supplying sequences which contain residues that are
  encoded by more than one letter.

- ``penalize_extend_when_opening``: boolean (default: False).
  Whether to count an extension penalty when opening a gap. If false, a gap of
  1 is only penalized an "open" penalty, otherwise it is penalized
  "open+extend".

- ``penalize_end_gaps``: boolean.
  Whether to count the gaps at the ends of an alignment. By default, they are
  counted for global alignments but not for local ones. Setting
  ``penalize_end_gaps`` to (boolean, boolean) allows you to specify for the
  two sequences separately whether gaps at the end of the alignment should be
  counted.

- ``gap_char``: string (default: ``'-'``).
  Which character to use as a gap character in the alignment returned. If your
  input sequences are lists, you must change this to ``['-']``.

- ``force_generic``: boolean (default: False).
  Always use the generic, non-cached, dynamic programming function (slow!).
  For debugging.

- ``score_only``: boolean (default: False).
  Only get the best score, don't recover any alignments. The return value of
  the function is the score. Faster and uses less memory.

- ``one_alignment_only``: boolean (default: False).
  Only recover one alignment.

The other parameters of the alignment function depend on the function called.
Some examples:

- Find the best global alignment between the two sequences. Identical
  characters are given 1 point. No points are deducted for mismatches or gaps.

    >>> for a in pairwise2.align.globalxx("ACCGT", "ACG"):
    ...     print(format_alignment(*a))
    ACCGT
    | || 
    A-CG-
      Score=3
    <BLANKLINE>
    ACCGT
    || | 
    AC-G-
      Score=3
    <BLANKLINE>

- Same thing as before, but with a local alignment. Note that
  ``format_alignment`` will only show the aligned parts of the sequences,
  together with the starting positions.

    >>> for a in pairwise2.align.localxx("ACCGT", "ACG"):
    ...     print(format_alignment(*a))
    1 ACCG
      | ||
    1 A-CG
      Score=3
    <BLANKLINE>
    1 ACCG
      || |
    1 AC-G
      Score=3
    <BLANKLINE>

  To restore the 'historic' behaviour of ``format_alignemt``, i.e., showing
  also the un-aligned parts of both sequences, use the new keyword parameter
  ``full_sequences``:

    >>> for a in pairwise2.align.localxx("ACCGT", "ACG"):
    ...     print(format_alignment(*a, full_sequences=True))
    ACCGT
    | || 
    A-CG-
      Score=3
    <BLANKLINE>
    ACCGT
    || | 
    AC-G-
      Score=3
    <BLANKLINE>


- Do a global alignment. Identical characters are given 2 points, 1 point is
  deducted for each non-identical character. Don't penalize gaps.

    >>> for a in pairwise2.align.globalmx("ACCGT", "ACG", 2, -1):
    ...     print(format_alignment(*a))
    ACCGT
    | || 
    A-CG-
      Score=6
    <BLANKLINE>
    ACCGT
    || | 
    AC-G-
      Score=6
    <BLANKLINE>

- Same as above, except now 0.5 points are deducted when opening a gap, and
  0.1 points are deducted when extending it.

    >>> for a in pairwise2.align.globalms("ACCGT", "ACG", 2, -1, -.5, -.1):
    ...     print(format_alignment(*a))
    ACCGT
    | || 
    A-CG-
      Score=5
    <BLANKLINE>
    ACCGT
    || | 
    AC-G-
      Score=5
    <BLANKLINE>

- Note that you can use keywords to increase the readability, e.g.:

    >>> a = pairwise2.align.globalms("ACGT", "ACG", match=2, mismatch=-1, open=-.5,
    ...                              extend=-.1)

- Depending on the penalties, a gap in one sequence may be followed by a gap in
  the other sequence.If you don't like this behaviour, increase the gap-open
  penalty:

    >>> for a in pairwise2.align.globalms("A", "T", 5, -4, -1, -.1):
    ...     print(format_alignment(*a))
    A-
    <BLANKLINE>
    -T
      Score=-2
    <BLANKLINE>
    >>> for a in pairwise2.align.globalms("A", "T", 5, -4, -3, -.1):
    ...	    print(format_alignment(*a))
    A
    .
    T
      Score=-4
    <BLANKLINE>

- The alignment function can also use known matrices already included in
  Biopython (in ``Bio.Align.substitution_matrices``):

    >>> from Bio.Align import substitution_matrices
    >>> matrix = substitution_matrices.load("BLOSUM62")
    >>> for a in pairwise2.align.globaldx("KEVLA", "EVL", matrix):
    ...     print(format_alignment(*a))
    KEVLA
     ||| 
    -EVL-
      Score=13
    <BLANKLINE>

- With the parameter ``c`` you can define your own match- and gap functions.
  E.g. to define an affine logarithmic gap function and using it:

    >>> from math import log
    >>> def gap_function(x, y):  # x is gap position in seq, y is gap length
    ...     if y == 0:  # No gap
    ...         return 0
    ...     elif y == 1:  # Gap open penalty
    ...         return -2
    ...     return - (2 + y/4.0 + log(y)/2.0)
    ...
    >>> alignment = pairwise2.align.globalmc("ACCCCCGT", "ACG", 5, -4,
    ...                                      gap_function, gap_function)

  You can define different gap functions for each sequence.
  Self-defined match functions must take the two residues to be compared and
  return a score.

To see a description of the parameters for a function, please look at
the docstring for the function via the help function, e.g.
type ``help(pairwise2.align.localds)`` at the Python prompt.

"""
import warnings
from collections import namedtuple
from Bio import BiopythonWarning
from Bio import BiopythonDeprecationWarning
from Bio.Align import substitution_matrices
warnings.warn('Bio.pairwise2 has been deprecated, and we intend to remove it in a future release of Biopython. As an alternative, please consider using Bio.Align.PairwiseAligner as a replacement, and contact the Biopython developers if you still need the Bio.pairwise2 module.', BiopythonDeprecationWarning)
MAX_ALIGNMENTS = 1000
Alignment = namedtuple('Alignment', 'seqA, seqB, score, start, end')

class align:
    """Provide functions that do alignments.

    Alignment functions are called as:

      pairwise2.align.globalXX

    or

      pairwise2.align.localXX

    Where XX is a 2 character code indicating the match/mismatch parameters
    (first character, either x, m, d or c) and the gap penalty parameters
    (second character, either x, s, d, or c).

    For a detailed description read the main module's docstring (e.g.,
    type ``help(pairwise2)``).
    To see a description of the parameters for a function, please
    look at the docstring for the function, e.g. type
    ``help(pairwise2.align.localds)`` at the Python prompt.
    """

    class alignment_function:
        """Callable class which impersonates an alignment function.

        The constructor takes the name of the function.  This class
        will decode the name of the function to figure out how to
        interpret the parameters.
        """
        match2args = {'x': ([], ''), 'm': (['match', 'mismatch'], 'match is the score to given to identical characters.\nmismatch is the score given to non-identical ones.'), 'd': (['match_dict'], "match_dict is a dictionary where the keys are tuples\nof pairs of characters and the values are the scores,\ne.g. ('A', 'C') : 2.5."), 'c': (['match_fn'], 'match_fn is a callback function that takes two characters and returns the score between them.')}
        penalty2args = {'x': ([], ''), 's': (['open', 'extend'], 'open and extend are the gap penalties when a gap is\nopened and extended.  They should be negative.'), 'd': (['openA', 'extendA', 'openB', 'extendB'], 'openA and extendA are the gap penalties for sequenceA,\nand openB and extendB for sequenceB.  The penalties\nshould be negative.'), 'c': (['gap_A_fn', 'gap_B_fn'], 'gap_A_fn and gap_B_fn are callback functions that takes\n(1) the index where the gap is opened, and (2) the length\nof the gap.  They should return a gap penalty.')}

        def __init__(self, name):
            if False:
                return 10
            'Check to make sure the name of the function is reasonable.'
            if name.startswith('global'):
                if len(name) != 8:
                    raise AttributeError('function should be globalXX')
            elif name.startswith('local'):
                if len(name) != 7:
                    raise AttributeError('function should be localXX')
            else:
                raise AttributeError(name)
            (align_type, match_type, penalty_type) = (name[:-2], name[-2], name[-1])
            try:
                (match_args, match_doc) = self.match2args[match_type]
            except KeyError:
                raise AttributeError(f'unknown match type {match_type!r}')
            try:
                (penalty_args, penalty_doc) = self.penalty2args[penalty_type]
            except KeyError:
                raise AttributeError(f'unknown penalty type {penalty_type!r}')
            param_names = ['sequenceA', 'sequenceB']
            param_names.extend(match_args)
            param_names.extend(penalty_args)
            self.function_name = name
            self.align_type = align_type
            self.param_names = param_names
            self.__name__ = self.function_name
            doc = f"{self.__name__}({', '.join(self.param_names)}) -> alignments\n"
            doc += '\nThe following parameters can also be used with optional\nkeywords of the same name.\n\n\nsequenceA and sequenceB must be of the same type, either\nstrings, lists or Biopython sequence objects.\n\n'
            if match_doc:
                doc += f'\n{match_doc}\n'
            if penalty_doc:
                doc += f'\n{penalty_doc}\n'
            doc += '\nalignments is a list of named tuples (seqA, seqB, score,\nbegin, end). seqA and seqB are strings showing the alignment\nbetween the sequences.  score is the score of the alignment.\nbegin and end are indexes of seqA and seqB that indicate\nwhere the alignment occurs.\n'
            self.__doc__ = doc

        def decode(self, *args, **keywds):
            if False:
                while True:
                    i = 10
            'Decode the arguments for the _align function.\n\n            keywds will get passed to it, so translate the arguments\n            to this function into forms appropriate for _align.\n            '
            keywds = keywds.copy()
            args += (len(self.param_names) - len(args)) * (None,)
            for key in keywds.copy():
                if key in self.param_names:
                    _index = self.param_names.index(key)
                    args = args[:_index] + (keywds[key],) + args[_index:]
                    del keywds[key]
            args = tuple((arg for arg in args if arg is not None))
            if len(args) != len(self.param_names):
                raise TypeError('%s takes exactly %d argument (%d given)' % (self.function_name, len(self.param_names), len(args)))
            i = 0
            while i < len(self.param_names):
                if self.param_names[i] in ['sequenceA', 'sequenceB', 'gap_A_fn', 'gap_B_fn', 'match_fn']:
                    keywds[self.param_names[i]] = args[i]
                    i += 1
                elif self.param_names[i] == 'match':
                    assert self.param_names[i + 1] == 'mismatch'
                    (match, mismatch) = (args[i], args[i + 1])
                    keywds['match_fn'] = identity_match(match, mismatch)
                    i += 2
                elif self.param_names[i] == 'match_dict':
                    keywds['match_fn'] = dictionary_match(args[i])
                    i += 1
                elif self.param_names[i] == 'open':
                    assert self.param_names[i + 1] == 'extend'
                    (open, extend) = (args[i], args[i + 1])
                    pe = keywds.get('penalize_extend_when_opening', 0)
                    keywds['gap_A_fn'] = affine_penalty(open, extend, pe)
                    keywds['gap_B_fn'] = affine_penalty(open, extend, pe)
                    i += 2
                elif self.param_names[i] == 'openA':
                    assert self.param_names[i + 3] == 'extendB'
                    (openA, extendA, openB, extendB) = args[i:i + 4]
                    pe = keywds.get('penalize_extend_when_opening', 0)
                    keywds['gap_A_fn'] = affine_penalty(openA, extendA, pe)
                    keywds['gap_B_fn'] = affine_penalty(openB, extendB, pe)
                    i += 4
                else:
                    raise ValueError(f'unknown parameter {self.param_names[i]!r}')
            pe = keywds.get('penalize_extend_when_opening', 0)
            default_params = [('match_fn', identity_match(1, 0)), ('gap_A_fn', affine_penalty(0, 0, pe)), ('gap_B_fn', affine_penalty(0, 0, pe)), ('penalize_extend_when_opening', 0), ('penalize_end_gaps', self.align_type == 'global'), ('align_globally', self.align_type == 'global'), ('gap_char', '-'), ('force_generic', 0), ('score_only', 0), ('one_alignment_only', 0)]
            for (name, default) in default_params:
                keywds[name] = keywds.get(name, default)
            value = keywds['penalize_end_gaps']
            try:
                n = len(value)
            except TypeError:
                keywds['penalize_end_gaps'] = tuple([value] * 2)
            else:
                assert n == 2
            return keywds

        def __call__(self, *args, **keywds):
            if False:
                while True:
                    i = 10
            'Call the alignment instance already created.'
            keywds = self.decode(*args, **keywds)
            return _align(**keywds)

    def __getattr__(self, attr):
        if False:
            return 10
        'Call alignment_function() to check and decode the attributes.'
        wrapper = self.alignment_function(attr)
        wrapper_type = type(wrapper)
        wrapper_dict = wrapper_type.__dict__.copy()
        wrapper_dict['__doc__'] = wrapper.__doc__
        new_alignment_function = type('alignment_function', (object,), wrapper_dict)
        return new_alignment_function(attr)
align = align()

def _align(sequenceA, sequenceB, match_fn, gap_A_fn, gap_B_fn, penalize_extend_when_opening, penalize_end_gaps, align_globally, gap_char, force_generic, score_only, one_alignment_only):
    if False:
        return 10
    'Return optimal alignments between two sequences (PRIVATE).\n\n    This method either returns a list of optimal alignments (with the same\n    score) or just the optimal score.\n    '
    if not sequenceA or not sequenceB:
        return []
    try:
        sequenceA + gap_char
        sequenceB + gap_char
    except TypeError:
        raise TypeError('both sequences must be of the same type, either string/sequence object or list. Gap character must fit the sequence type (string or list)')
    if not isinstance(sequenceA, list):
        sequenceA = str(sequenceA)
    if not isinstance(sequenceB, list):
        sequenceB = str(sequenceB)
    if not align_globally and (penalize_end_gaps[0] or penalize_end_gaps[1]):
        warnings.warn('"penalize_end_gaps" should not be used in local alignments. The resulting score may be wrong.', BiopythonWarning)
    if not force_generic and isinstance(gap_A_fn, affine_penalty) and isinstance(gap_B_fn, affine_penalty):
        (open_A, extend_A) = (gap_A_fn.open, gap_A_fn.extend)
        (open_B, extend_B) = (gap_B_fn.open, gap_B_fn.extend)
        matrices = _make_score_matrix_fast(sequenceA, sequenceB, match_fn, open_A, extend_A, open_B, extend_B, penalize_extend_when_opening, penalize_end_gaps, align_globally, score_only)
    else:
        matrices = _make_score_matrix_generic(sequenceA, sequenceB, match_fn, gap_A_fn, gap_B_fn, penalize_end_gaps, align_globally, score_only)
    (score_matrix, trace_matrix, best_score) = matrices
    if score_only:
        return best_score
    starts = _find_start(score_matrix, best_score, align_globally)
    alignments = _recover_alignments(sequenceA, sequenceB, starts, best_score, score_matrix, trace_matrix, align_globally, gap_char, one_alignment_only, gap_A_fn, gap_B_fn)
    if not alignments:
        (score_matrix, trace_matrix) = _reverse_matrices(score_matrix, trace_matrix)
        starts = [(z, (y, x)) for (z, (x, y)) in starts]
        alignments = _recover_alignments(sequenceB, sequenceA, starts, best_score, score_matrix, trace_matrix, align_globally, gap_char, one_alignment_only, gap_B_fn, gap_A_fn, reverse=True)
    return alignments

def _make_score_matrix_generic(sequenceA, sequenceB, match_fn, gap_A_fn, gap_B_fn, penalize_end_gaps, align_globally, score_only):
    if False:
        while True:
            i = 10
    'Generate a score and traceback matrix (PRIVATE).\n\n    This implementation according to Needleman-Wunsch allows the usage of\n    general gap functions and is rather slow. It is automatically called if\n    you define your own gap functions. You can force the usage of this method\n    with ``force_generic=True``.\n    '
    local_max_score = 0
    (lenA, lenB) = (len(sequenceA), len(sequenceB))
    (score_matrix, trace_matrix) = ([], [])
    for i in range(lenA + 1):
        score_matrix.append([None] * (lenB + 1))
        if not score_only:
            trace_matrix.append([None] * (lenB + 1))
    for i in range(lenA + 1):
        if penalize_end_gaps[1]:
            score = gap_B_fn(0, i)
        else:
            score = 0.0
        score_matrix[i][0] = score
    for i in range(lenB + 1):
        if penalize_end_gaps[0]:
            score = gap_A_fn(0, i)
        else:
            score = 0.0
        score_matrix[0][i] = score
    for row in range(1, lenA + 1):
        for col in range(1, lenB + 1):
            nogap_score = score_matrix[row - 1][col - 1] + match_fn(sequenceA[row - 1], sequenceB[col - 1])
            if not penalize_end_gaps[0] and row == lenA:
                row_open = score_matrix[row][col - 1]
                row_extend = max((score_matrix[row][x] for x in range(col)))
            else:
                row_open = score_matrix[row][col - 1] + gap_A_fn(row, 1)
                row_extend = max((score_matrix[row][x] + gap_A_fn(row, col - x) for x in range(col)))
            if not penalize_end_gaps[1] and col == lenB:
                col_open = score_matrix[row - 1][col]
                col_extend = max((score_matrix[x][col] for x in range(row)))
            else:
                col_open = score_matrix[row - 1][col] + gap_B_fn(col, 1)
                col_extend = max((score_matrix[x][col] + gap_B_fn(col, row - x) for x in range(row)))
            best_score = max(nogap_score, row_open, row_extend, col_open, col_extend)
            local_max_score = max(local_max_score, best_score)
            if not align_globally and best_score < 0:
                score_matrix[row][col] = 0.0
            else:
                score_matrix[row][col] = best_score
            if not score_only:
                trace_score = 0
                if rint(nogap_score) == rint(best_score):
                    trace_score += 2
                if rint(row_open) == rint(best_score):
                    trace_score += 1
                if rint(row_extend) == rint(best_score):
                    trace_score += 8
                if rint(col_open) == rint(best_score):
                    trace_score += 4
                if rint(col_extend) == rint(best_score):
                    trace_score += 16
                trace_matrix[row][col] = trace_score
    if not align_globally:
        best_score = local_max_score
    return (score_matrix, trace_matrix, best_score)

def _make_score_matrix_fast(sequenceA, sequenceB, match_fn, open_A, extend_A, open_B, extend_B, penalize_extend_when_opening, penalize_end_gaps, align_globally, score_only):
    if False:
        i = 10
        return i + 15
    'Generate a score and traceback matrix according to Gotoh (PRIVATE).\n\n    This is an implementation of the Needleman-Wunsch dynamic programming\n    algorithm as modified by Gotoh, implementing affine gap penalties.\n    In short, we have three matrices, holding scores for alignments ending\n    in (1) a match/mismatch, (2) a gap in sequence A, and (3) a gap in\n    sequence B, respectively. However, we can combine them in one matrix,\n    which holds the best scores, and store only those values from the\n    other matrices that are actually used for the next step of calculation.\n    The traceback matrix holds the positions for backtracing the alignment.\n    '
    first_A_gap = calc_affine_penalty(1, open_A, extend_A, penalize_extend_when_opening)
    first_B_gap = calc_affine_penalty(1, open_B, extend_B, penalize_extend_when_opening)
    local_max_score = 0
    (lenA, lenB) = (len(sequenceA), len(sequenceB))
    (score_matrix, trace_matrix) = ([], [])
    for i in range(lenA + 1):
        score_matrix.append([None] * (lenB + 1))
        if not score_only:
            trace_matrix.append([None] * (lenB + 1))
    for i in range(lenA + 1):
        if penalize_end_gaps[1]:
            score = calc_affine_penalty(i, open_B, extend_B, penalize_extend_when_opening)
        else:
            score = 0
        score_matrix[i][0] = score
    for i in range(lenB + 1):
        if penalize_end_gaps[0]:
            score = calc_affine_penalty(i, open_A, extend_A, penalize_extend_when_opening)
        else:
            score = 0
        score_matrix[0][i] = score
    col_score = [0]
    for i in range(1, lenB + 1):
        col_score.append(calc_affine_penalty(i, 2 * open_B, extend_B, penalize_extend_when_opening))
    for row in range(1, lenA + 1):
        row_score = calc_affine_penalty(row, 2 * open_A, extend_A, penalize_extend_when_opening)
        for col in range(1, lenB + 1):
            nogap_score = score_matrix[row - 1][col - 1] + match_fn(sequenceA[row - 1], sequenceB[col - 1])
            if not penalize_end_gaps[0] and row == lenA:
                row_open = score_matrix[row][col - 1]
                row_extend = row_score
            else:
                row_open = score_matrix[row][col - 1] + first_A_gap
                row_extend = row_score + extend_A
            row_score = max(row_open, row_extend)
            if not penalize_end_gaps[1] and col == lenB:
                col_open = score_matrix[row - 1][col]
                col_extend = col_score[col]
            else:
                col_open = score_matrix[row - 1][col] + first_B_gap
                col_extend = col_score[col] + extend_B
            col_score[col] = max(col_open, col_extend)
            best_score = max(nogap_score, col_score[col], row_score)
            local_max_score = max(local_max_score, best_score)
            if not align_globally and best_score < 0:
                score_matrix[row][col] = 0
            else:
                score_matrix[row][col] = best_score
            if not score_only:
                row_score_rint = rint(row_score)
                col_score_rint = rint(col_score[col])
                row_trace_score = 0
                col_trace_score = 0
                if rint(row_open) == row_score_rint:
                    row_trace_score += 1
                if rint(row_extend) == row_score_rint:
                    row_trace_score += 8
                if rint(col_open) == col_score_rint:
                    col_trace_score += 4
                if rint(col_extend) == col_score_rint:
                    col_trace_score += 16
                trace_score = 0
                best_score_rint = rint(best_score)
                if rint(nogap_score) == best_score_rint:
                    trace_score += 2
                if row_score_rint == best_score_rint:
                    trace_score += row_trace_score
                if col_score_rint == best_score_rint:
                    trace_score += col_trace_score
                trace_matrix[row][col] = trace_score
    if not align_globally:
        best_score = local_max_score
    return (score_matrix, trace_matrix, best_score)

def _recover_alignments(sequenceA, sequenceB, starts, best_score, score_matrix, trace_matrix, align_globally, gap_char, one_alignment_only, gap_A_fn, gap_B_fn, reverse=False):
    if False:
        for i in range(10):
            print('nop')
    "Do the backtracing and return a list of alignments (PRIVATE).\n\n    Recover the alignments by following the traceback matrix.  This\n    is a recursive procedure, but it's implemented here iteratively\n    with a stack.\n\n    sequenceA and sequenceB may be sequences, including strings,\n    lists, or list-like objects.  In order to preserve the type of\n    the object, we need to use slices on the sequences instead of\n    indexes.  For example, sequenceA[row] may return a type that's\n    not compatible with sequenceA, e.g. if sequenceA is a list and\n    sequenceA[row] is a string.  Thus, avoid using indexes and use\n    slices, e.g. sequenceA[row:row+1].  Assume that client-defined\n    sequence classes preserve these semantics.\n    "
    (lenA, lenB) = (len(sequenceA), len(sequenceB))
    (ali_seqA, ali_seqB) = (sequenceA[0:0], sequenceB[0:0])
    tracebacks = []
    in_process = []
    for start in starts:
        (score, (row, col)) = start
        begin = 0
        if align_globally:
            end = None
        else:
            if (score, (row - 1, col - 1)) in starts:
                continue
            if score <= 0:
                continue
            trace = trace_matrix[row][col]
            if (trace - trace % 2) % 4 == 2:
                trace_matrix[row][col] = 2
            else:
                continue
            end = -max(lenA - row, lenB - col)
            if not end:
                end = None
            col_distance = lenB - col
            row_distance = lenA - row
            ali_seqA = (col_distance - row_distance) * gap_char + sequenceA[lenA - 1:row - 1:-1]
            ali_seqB = (row_distance - col_distance) * gap_char + sequenceB[lenB - 1:col - 1:-1]
        in_process += [(ali_seqA, ali_seqB, end, row, col, False, trace_matrix[row][col])]
    while in_process and len(tracebacks) < MAX_ALIGNMENTS:
        dead_end = False
        (ali_seqA, ali_seqB, end, row, col, col_gap, trace) = in_process.pop()
        while (row > 0 or col > 0) and (not dead_end):
            cache = (ali_seqA[:], ali_seqB[:], end, row, col, col_gap)
            if not trace:
                if col and col_gap:
                    dead_end = True
                else:
                    (ali_seqA, ali_seqB) = _finish_backtrace(sequenceA, sequenceB, ali_seqA, ali_seqB, row, col, gap_char)
                break
            elif trace % 2 == 1:
                trace -= 1
                if col_gap:
                    dead_end = True
                else:
                    col -= 1
                    ali_seqA += gap_char
                    ali_seqB += sequenceB[col:col + 1]
                    col_gap = False
            elif trace % 4 == 2:
                trace -= 2
                row -= 1
                col -= 1
                ali_seqA += sequenceA[row:row + 1]
                ali_seqB += sequenceB[col:col + 1]
                col_gap = False
            elif trace % 8 == 4:
                trace -= 4
                row -= 1
                ali_seqA += sequenceA[row:row + 1]
                ali_seqB += gap_char
                col_gap = True
            elif trace in (8, 24):
                trace -= 8
                if col_gap:
                    dead_end = True
                else:
                    col_gap = False
                    x = _find_gap_open(sequenceA, sequenceB, ali_seqA, ali_seqB, end, row, col, col_gap, gap_char, score_matrix, trace_matrix, in_process, gap_A_fn, col, row, 'col', best_score, align_globally)
                    (ali_seqA, ali_seqB, row, col, in_process, dead_end) = x
            elif trace == 16:
                trace -= 16
                col_gap = True
                x = _find_gap_open(sequenceA, sequenceB, ali_seqA, ali_seqB, end, row, col, col_gap, gap_char, score_matrix, trace_matrix, in_process, gap_B_fn, row, col, 'row', best_score, align_globally)
                (ali_seqA, ali_seqB, row, col, in_process, dead_end) = x
            if trace:
                cache += (trace,)
                in_process.append(cache)
            trace = trace_matrix[row][col]
            if not align_globally:
                if score_matrix[row][col] == best_score:
                    dead_end = True
                elif score_matrix[row][col] <= 0:
                    begin = max(row, col)
                    trace = 0
        if not dead_end:
            if not reverse:
                tracebacks.append((ali_seqA[::-1], ali_seqB[::-1], score, begin, end))
            else:
                tracebacks.append((ali_seqB[::-1], ali_seqA[::-1], score, begin, end))
            if one_alignment_only:
                break
    return _clean_alignments(tracebacks)

def _find_start(score_matrix, best_score, align_globally):
    if False:
        return 10
    'Return a list of starting points (score, (row, col)) (PRIVATE).\n\n    Indicating every possible place to start the tracebacks.\n    '
    (nrows, ncols) = (len(score_matrix), len(score_matrix[0]))
    if align_globally:
        starts = [(best_score, (nrows - 1, ncols - 1))]
    else:
        starts = []
        tolerance = 0
        for row in range(nrows):
            for col in range(ncols):
                score = score_matrix[row][col]
                if rint(abs(score - best_score)) <= rint(tolerance):
                    starts.append((score, (row, col)))
    return starts

def _reverse_matrices(score_matrix, trace_matrix):
    if False:
        i = 10
        return i + 15
    'Reverse score and trace matrices (PRIVATE).'
    reverse_score_matrix = []
    reverse_trace_matrix = []
    reverse_trace = {1: 4, 2: 2, 3: 6, 4: 1, 5: 5, 6: 3, 7: 7, 8: 16, 9: 20, 10: 18, 11: 22, 12: 17, 13: 21, 14: 19, 15: 23, 16: 8, 17: 12, 18: 10, 19: 14, 20: 9, 21: 13, 22: 11, 23: 15, 24: 24, 25: 28, 26: 26, 27: 30, 28: 25, 29: 29, 30: 27, 31: 31, None: None}
    for col in range(len(score_matrix[0])):
        new_score_row = []
        new_trace_row = []
        for row in range(len(score_matrix)):
            new_score_row.append(score_matrix[row][col])
            new_trace_row.append(reverse_trace[trace_matrix[row][col]])
        reverse_score_matrix.append(new_score_row)
        reverse_trace_matrix.append(new_trace_row)
    return (reverse_score_matrix, reverse_trace_matrix)

def _clean_alignments(alignments):
    if False:
        print('Hello World!')
    'Take a list of alignments and return a cleaned version (PRIVATE).\n\n    Remove duplicates, make sure begin and end are set correctly, remove\n    empty alignments.\n    '
    unique_alignments = []
    for align in alignments:
        if align not in unique_alignments:
            unique_alignments.append(align)
    i = 0
    while i < len(unique_alignments):
        (seqA, seqB, score, begin, end) = unique_alignments[i]
        if end is None:
            end = len(seqA)
        elif end < 0:
            end = end + len(seqA)
        if begin >= end:
            del unique_alignments[i]
            continue
        unique_alignments[i] = Alignment(seqA, seqB, score, begin, end)
        i += 1
    return unique_alignments

def _finish_backtrace(sequenceA, sequenceB, ali_seqA, ali_seqB, row, col, gap_char):
    if False:
        i = 10
        return i + 15
    'Add remaining sequences and fill with gaps if necessary (PRIVATE).'
    if row:
        ali_seqA += sequenceA[row - 1::-1]
    if col:
        ali_seqB += sequenceB[col - 1::-1]
    if row > col:
        ali_seqB += gap_char * (len(ali_seqA) - len(ali_seqB))
    elif col > row:
        ali_seqA += gap_char * (len(ali_seqB) - len(ali_seqA))
    return (ali_seqA, ali_seqB)

def _find_gap_open(sequenceA, sequenceB, ali_seqA, ali_seqB, end, row, col, col_gap, gap_char, score_matrix, trace_matrix, in_process, gap_fn, target, index, direction, best_score, align_globally):
    if False:
        print('Hello World!')
    'Find the starting point(s) of the extended gap (PRIVATE).'
    dead_end = False
    target_score = score_matrix[row][col]
    for n in range(target):
        if direction == 'col':
            col -= 1
            ali_seqA += gap_char
            ali_seqB += sequenceB[col:col + 1]
        else:
            row -= 1
            ali_seqA += sequenceA[row:row + 1]
            ali_seqB += gap_char
        actual_score = score_matrix[row][col] + gap_fn(index, n + 1)
        if not align_globally and score_matrix[row][col] == best_score:
            dead_end = True
            break
        if rint(actual_score) == rint(target_score) and n > 0:
            if not trace_matrix[row][col]:
                break
            else:
                in_process.append((ali_seqA[:], ali_seqB[:], end, row, col, col_gap, trace_matrix[row][col]))
        if not trace_matrix[row][col]:
            dead_end = True
    return (ali_seqA, ali_seqB, row, col, in_process, dead_end)
_PRECISION = 1000

def rint(x, precision=_PRECISION):
    if False:
        print('Hello World!')
    'Print number with declared precision.'
    return int(x * precision + 0.5)

class identity_match:
    """Create a match function for use in an alignment.

    match and mismatch are the scores to give when two residues are equal
    or unequal.  By default, match is 1 and mismatch is 0.
    """

    def __init__(self, match=1, mismatch=0):
        if False:
            print('Hello World!')
        'Initialize the class.'
        self.match = match
        self.mismatch = mismatch

    def __call__(self, charA, charB):
        if False:
            for i in range(10):
                print('nop')
        'Call a match function instance already created.'
        if charA == charB:
            return self.match
        return self.mismatch

class dictionary_match:
    """Create a match function for use in an alignment.

    Attributes:
     - score_dict     - A dictionary where the keys are tuples (residue 1,
       residue 2) and the values are the match scores between those residues.
     - symmetric      - A flag that indicates whether the scores are symmetric.

    """

    def __init__(self, score_dict, symmetric=1):
        if False:
            i = 10
            return i + 15
        'Initialize the class.'
        if isinstance(score_dict, substitution_matrices.Array):
            score_dict = dict(score_dict)
        self.score_dict = score_dict
        self.symmetric = symmetric

    def __call__(self, charA, charB):
        if False:
            while True:
                i = 10
        'Call a dictionary match instance already created.'
        if self.symmetric and (charA, charB) not in self.score_dict:
            (charB, charA) = (charA, charB)
        return self.score_dict[charA, charB]

class affine_penalty:
    """Create a gap function for use in an alignment."""

    def __init__(self, open, extend, penalize_extend_when_opening=0):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the class.'
        if open > 0 or extend > 0:
            raise ValueError('Gap penalties should be non-positive.')
        if not penalize_extend_when_opening and extend < open:
            raise ValueError('Gap opening penalty should be higher than gap extension penalty (or equal)')
        (self.open, self.extend) = (open, extend)
        self.penalize_extend_when_opening = penalize_extend_when_opening

    def __call__(self, index, length):
        if False:
            for i in range(10):
                print('nop')
        'Call a gap function instance already created.'
        return calc_affine_penalty(length, self.open, self.extend, self.penalize_extend_when_opening)

def calc_affine_penalty(length, open, extend, penalize_extend_when_opening):
    if False:
        for i in range(10):
            print('nop')
    'Calculate a penalty score for the gap function.'
    if length <= 0:
        return 0.0
    penalty = open + extend * length
    if not penalize_extend_when_opening:
        penalty -= extend
    return penalty

def print_matrix(matrix):
    if False:
        return 10
    'Print out a matrix for debugging purposes.'
    matrixT = [[] for x in range(len(matrix[0]))]
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrixT[j].append(len(str(matrix[i][j])))
    ndigits = [max(x) for x in matrixT]
    for i in range(len(matrix)):
        print(' '.join(('%*s ' % (ndigits[j], matrix[i][j]) for j in range(len(matrix[i])))))

def format_alignment(align1, align2, score, begin, end, full_sequences=False):
    if False:
        i = 10
        return i + 15
    'Format the alignment prettily into a string.\n\n    IMPORTANT: Gap symbol must be "-" (or [\'-\'] for lists)!\n\n    Since Biopython 1.71 identical matches are shown with a pipe\n    character, mismatches as a dot, and gaps as a space.\n\n    Prior releases just used the pipe character to indicate the\n    aligned region (matches, mismatches and gaps).\n\n    Also, in local alignments, if the alignment does not include\n    the whole sequences, now only the aligned part is shown,\n    together with the start positions of the aligned subsequences.\n    The start positions are 1-based; so start position n is the\n    n-th base/amino acid in the *un-aligned* sequence.\n\n    NOTE: This is different to the alignment\'s begin/end values,\n    which give the Python indices (0-based) of the bases/amino acids\n    in the *aligned* sequences.\n\n    If you want to restore the \'historic\' behaviour, that means\n    displaying the whole sequences (including the non-aligned parts),\n    use ``full_sequences=True``. In this case, the non-aligned leading\n    and trailing parts are also indicated by spaces in the match-line.\n    '
    align_begin = begin
    align_end = end
    start1 = start2 = ''
    start_m = begin
    if not full_sequences and (begin != 0 or end != len(align1)):
        start1 = str(len(align1[:begin]) - align1[:begin].count('-') + 1) + ' '
        start2 = str(len(align2[:begin]) - align2[:begin].count('-') + 1) + ' '
        start_m = max(len(start1), len(start2))
    elif full_sequences:
        start_m = 0
        begin = 0
        end = len(align1)
    if isinstance(align1, list):
        align1 = [a + ' ' for a in align1]
        align2 = [a + ' ' for a in align2]
    s1_line = ['{:>{width}}'.format(start1, width=start_m)]
    m_line = [' ' * start_m]
    s2_line = ['{:>{width}}'.format(start2, width=start_m)]
    for (n, (a, b)) in enumerate(zip(align1[begin:end], align2[begin:end])):
        m_len = max(len(a), len(b))
        s1_line.append('{:^{width}}'.format(a, width=m_len))
        s2_line.append('{:^{width}}'.format(b, width=m_len))
        if full_sequences and (n < align_begin or n >= align_end):
            m_line.append('{:^{width}}'.format(' ', width=m_len))
            continue
        if a == b:
            m_line.append('{:^{width}}'.format('|', width=m_len))
        elif a.strip() == '-' or b.strip() == '-':
            m_line.append('{:^{width}}'.format(' ', width=m_len))
        else:
            m_line.append('{:^{width}}'.format('.', width=m_len))
    s2_line.append(f'\n  Score={score:g}\n')
    return '\n'.join([''.join(s1_line), ''.join(m_line), ''.join(s2_line)])
_python_make_score_matrix_fast = _make_score_matrix_fast
_python_rint = rint
try:
    from .cpairwise2 import rint, _make_score_matrix_fast
except ImportError:
    warnings.warn('Import of C module failed. Falling back to pure Python implementation. This may be slooow...', BiopythonWarning)
if __name__ == '__main__':
    from Bio._utils import run_doctest
    run_doctest()