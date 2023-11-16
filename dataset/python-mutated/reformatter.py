"""Decide what the format for the code should be.

The `logical_line.LogicalLine`s are now ready to be formatted. LogicalLInes that
can be merged together are. The best formatting is returned as a string.

  Reformat(): the main function exported by this module.
"""
import collections
import heapq
import re
from yapf_third_party._ylib2to3 import pytree
from yapf_third_party._ylib2to3.pgen2 import token
from yapf.pytree import pytree_utils
from yapf.yapflib import format_decision_state
from yapf.yapflib import format_token
from yapf.yapflib import line_joiner
from yapf.yapflib import style

def Reformat(llines, lines=None):
    if False:
        i = 10
        return i + 15
    'Reformat the logical lines.\n\n  Arguments:\n    llines: (list of logical_line.LogicalLine) Lines we want to format.\n    lines: (set of int) The lines which can be modified or None if there is no\n      line range restriction.\n\n  Returns:\n    A string representing the reformatted code.\n  '
    final_lines = []
    prev_line = None
    indent_width = style.Get('INDENT_WIDTH')
    for lline in _SingleOrMergedLines(llines):
        first_token = lline.first
        _FormatFirstToken(first_token, lline.depth, prev_line, final_lines)
        indent_amt = indent_width * lline.depth
        state = format_decision_state.FormatDecisionState(lline, indent_amt)
        state.MoveStateToNextToken()
        if not lline.disable:
            if lline.first.is_comment:
                lline.first.value = lline.first.value.rstrip()
            elif lline.last.is_comment:
                lline.last.value = lline.last.value.rstrip()
            if prev_line and prev_line.disable:
                _RetainRequiredVerticalSpacingBetweenTokens(lline.first, prev_line.last, lines)
            if any((tok.is_comment for tok in lline.tokens)):
                _RetainVerticalSpacingBeforeComments(lline)
        if lline.disable or _LineHasContinuationMarkers(lline):
            _RetainHorizontalSpacing(lline)
            _RetainRequiredVerticalSpacing(lline, prev_line, lines)
            _EmitLineUnformatted(state)
        elif _LineContainsPylintDisableLineTooLong(lline) or _LineContainsI18n(lline):
            _RetainRequiredVerticalSpacing(lline, prev_line, lines)
            _EmitLineUnformatted(state)
        elif _CanPlaceOnSingleLine(lline) and (not any((tok.must_break_before for tok in lline.tokens))):
            while state.next_token:
                state.AddTokenToState(newline=False, dry_run=False)
        elif not _AnalyzeSolutionSpace(state):
            state = format_decision_state.FormatDecisionState(lline, indent_amt)
            state.MoveStateToNextToken()
            _RetainHorizontalSpacing(lline)
            _RetainRequiredVerticalSpacing(lline, prev_line, None)
            _EmitLineUnformatted(state)
        final_lines.append(lline)
        prev_line = lline
    _AlignTrailingComments(final_lines)
    return _FormatFinalLines(final_lines)

def _RetainHorizontalSpacing(line):
    if False:
        for i in range(10):
            print('nop')
    'Retain all horizontal spacing between tokens.'
    for tok in line.tokens:
        tok.RetainHorizontalSpacing(line.first.column, line.depth)

def _RetainRequiredVerticalSpacing(cur_line, prev_line, lines):
    if False:
        for i in range(10):
            print('nop')
    'Retain all vertical spacing between lines.'
    prev_tok = None
    if prev_line is not None:
        prev_tok = prev_line.last
    if cur_line.disable:
        lines = set()
    for cur_tok in cur_line.tokens:
        _RetainRequiredVerticalSpacingBetweenTokens(cur_tok, prev_tok, lines)
        prev_tok = cur_tok

def _RetainRequiredVerticalSpacingBetweenTokens(cur_tok, prev_tok, lines):
    if False:
        print('Hello World!')
    'Retain vertical spacing between two tokens if not in editable range.'
    if prev_tok is None:
        return
    if prev_tok.is_string:
        prev_lineno = prev_tok.lineno + prev_tok.value.count('\n')
    elif prev_tok.is_pseudo:
        if not prev_tok.previous_token.is_multiline_string:
            prev_lineno = prev_tok.previous_token.lineno
        else:
            prev_lineno = prev_tok.lineno
    else:
        prev_lineno = prev_tok.lineno
    if cur_tok.is_comment:
        cur_lineno = cur_tok.lineno - cur_tok.value.count('\n')
    else:
        cur_lineno = cur_tok.lineno
    if not prev_tok.is_comment and prev_tok.value.endswith('\\'):
        prev_lineno += prev_tok.value.count('\n')
    required_newlines = cur_lineno - prev_lineno
    if cur_tok.is_comment and (not prev_tok.is_comment):
        pass
    elif lines and lines.intersection(range(prev_lineno, cur_lineno + 1)):
        desired_newlines = cur_tok.whitespace_prefix.count('\n')
        whitespace_lines = range(prev_lineno + 1, cur_lineno)
        deletable_lines = len(lines.intersection(whitespace_lines))
        required_newlines = max(required_newlines - deletable_lines, desired_newlines)
    cur_tok.AdjustNewlinesBefore(required_newlines)

def _RetainVerticalSpacingBeforeComments(line):
    if False:
        return 10
    'Retain vertical spacing before comments.'
    prev_token = None
    for tok in line.tokens:
        if tok.is_comment and prev_token:
            if tok.lineno - tok.value.count('\n') - prev_token.lineno > 1:
                tok.AdjustNewlinesBefore(ONE_BLANK_LINE)
        prev_token = tok

def _EmitLineUnformatted(state):
    if False:
        for i in range(10):
            print('nop')
    'Emit the line without formatting.\n\n  The line contains code that if reformatted would break a non-syntactic\n  convention. E.g., i18n comments and function calls are tightly bound by\n  convention. Instead, we calculate when / if a newline should occur and honor\n  that. But otherwise the code emitted will be the same as the original code.\n\n  Arguments:\n    state: (format_decision_state.FormatDecisionState) The format decision\n      state.\n  '
    while state.next_token:
        previous_token = state.next_token.previous_token
        previous_lineno = previous_token.lineno
        if previous_token.is_multiline_string or previous_token.is_string:
            previous_lineno += previous_token.value.count('\n')
        if previous_token.is_continuation:
            newline = False
        else:
            newline = state.next_token.lineno > previous_lineno
        state.AddTokenToState(newline=newline, dry_run=False)

def _LineContainsI18n(line):
    if False:
        for i in range(10):
            print('nop')
    'Return true if there are i18n comments or function calls in the line.\n\n  I18n comments and pseudo-function calls are closely related. They cannot\n  be moved apart without breaking i18n.\n\n  Arguments:\n    line: (logical_line.LogicalLine) The line currently being formatted.\n\n  Returns:\n    True if the line contains i18n comments or function calls. False otherwise.\n  '
    if style.Get('I18N_COMMENT'):
        for tok in line.tokens:
            if tok.is_comment and re.match(style.Get('I18N_COMMENT'), tok.value):
                return True
    if style.Get('I18N_FUNCTION_CALL'):
        length = len(line.tokens)
        for index in range(length - 1):
            if line.tokens[index + 1].value == '(' and line.tokens[index].value in style.Get('I18N_FUNCTION_CALL'):
                return True
    return False

def _LineContainsPylintDisableLineTooLong(line):
    if False:
        i = 10
        return i + 15
    'Return true if there is a "pylint: disable=line-too-long" comment.'
    return re.search('\\bpylint:\\s+disable=line-too-long\\b', line.last.value)

def _LineHasContinuationMarkers(line):
    if False:
        for i in range(10):
            print('nop')
    'Return true if the line has continuation markers in it.'
    return any((tok.is_continuation for tok in line.tokens))

def _CanPlaceOnSingleLine(line):
    if False:
        while True:
            i = 10
    'Determine if the logical line can go on a single line.\n\n  Arguments:\n    line: (logical_line.LogicalLine) The line currently being formatted.\n\n  Returns:\n    True if the line can or should be added to a single line. False otherwise.\n  '
    token_types = [x.type for x in line.tokens]
    if style.Get('SPLIT_ARGUMENTS_WHEN_COMMA_TERMINATED') and any((token_types[token_index - 1] == token.COMMA for (token_index, token_type) in enumerate(token_types[1:], start=1) if token_type == token.RPAR)):
        return False
    if style.Get('FORCE_MULTILINE_DICT') and token.LBRACE in token_types:
        return False
    indent_amt = style.Get('INDENT_WIDTH') * line.depth
    last = line.last
    last_index = -1
    if last.is_pylint_comment or last.is_pytype_comment or last.is_copybara_comment:
        last = last.previous_token
        last_index = -2
    if last is None:
        return True
    return last.total_length + indent_amt <= style.Get('COLUMN_LIMIT') and (not any((tok.is_comment for tok in line.tokens[:last_index])))

def _AlignTrailingComments(final_lines):
    if False:
        return 10
    'Align trailing comments to the same column.'
    final_lines_index = 0
    while final_lines_index < len(final_lines):
        line = final_lines[final_lines_index]
        assert line.tokens
        processed_content = False
        for tok in line.tokens:
            if tok.is_comment and isinstance(tok.spaces_required_before, list) and tok.value.startswith('#'):
                all_pc_line_lengths = []
                max_line_length = 0
                while True:
                    if final_lines_index + len(all_pc_line_lengths) == len(final_lines):
                        break
                    this_line = final_lines[final_lines_index + len(all_pc_line_lengths)]
                    assert this_line.tokens
                    if all_pc_line_lengths and this_line.tokens[0].formatted_whitespace_prefix.startswith('\n\n'):
                        break
                    if this_line.disable:
                        all_pc_line_lengths.append([])
                        continue
                    line_content = ''
                    pc_line_lengths = []
                    for line_tok in this_line.tokens:
                        whitespace_prefix = line_tok.formatted_whitespace_prefix
                        newline_index = whitespace_prefix.rfind('\n')
                        if newline_index != -1:
                            max_line_length = max(max_line_length, len(line_content))
                            line_content = ''
                            whitespace_prefix = whitespace_prefix[newline_index + 1:]
                        if line_tok.is_comment:
                            pc_line_lengths.append(len(line_content))
                        else:
                            line_content += '{}{}'.format(whitespace_prefix, line_tok.value)
                    if pc_line_lengths:
                        max_line_length = max(max_line_length, max(pc_line_lengths))
                    all_pc_line_lengths.append(pc_line_lengths)
                max_line_length += 2
                aligned_col = None
                for potential_col in tok.spaces_required_before:
                    if potential_col > max_line_length:
                        aligned_col = potential_col
                        break
                if aligned_col is None:
                    aligned_col = max_line_length
                for (all_pc_line_lengths_index, pc_line_lengths) in enumerate(all_pc_line_lengths):
                    if not pc_line_lengths:
                        continue
                    this_line = final_lines[final_lines_index + all_pc_line_lengths_index]
                    pc_line_length_index = 0
                    for line_tok in this_line.tokens:
                        if line_tok.is_comment:
                            assert pc_line_length_index < len(pc_line_lengths)
                            assert pc_line_lengths[pc_line_length_index] < aligned_col
                            whitespace = ' ' * (aligned_col - pc_line_lengths[pc_line_length_index] - 1)
                            pc_line_length_index += 1
                            line_content = []
                            for (comment_line_index, comment_line) in enumerate(line_tok.value.split('\n')):
                                line_content.append('{}{}'.format(whitespace, comment_line.strip()))
                                if comment_line_index == 0:
                                    whitespace = ' ' * (aligned_col - 1)
                            line_content = '\n'.join(line_content)
                            existing_whitespace_prefix = line_tok.formatted_whitespace_prefix.lstrip('\n')
                            if line_content.startswith(existing_whitespace_prefix):
                                line_content = line_content[len(existing_whitespace_prefix):]
                            line_tok.value = line_content
                    assert pc_line_length_index == len(pc_line_lengths)
                final_lines_index += len(all_pc_line_lengths)
                processed_content = True
                break
        if not processed_content:
            final_lines_index += 1

def _FormatFinalLines(final_lines):
    if False:
        for i in range(10):
            print('nop')
    'Compose the final output from the finalized lines.'
    formatted_code = []
    for line in final_lines:
        formatted_line = []
        for tok in line.tokens:
            if not tok.is_pseudo:
                formatted_line.append(tok.formatted_whitespace_prefix)
                formatted_line.append(tok.value)
            elif not tok.next_token.whitespace_prefix.startswith('\n') and (not tok.next_token.whitespace_prefix.startswith(' ')):
                if tok.previous_token.value == ':' or tok.next_token.value not in ',}])':
                    formatted_line.append(' ')
        formatted_code.append(''.join(formatted_line))
    return ''.join(formatted_code) + '\n'

class _StateNode(object):
    """An edge in the solution space from 'previous.state' to 'state'.

  Attributes:
    state: (format_decision_state.FormatDecisionState) The format decision state
      for this node.
    newline: If True, then on the edge from 'previous.state' to 'state' a
      newline is inserted.
    previous: (_StateNode) The previous state node in the graph.
  """

    def __init__(self, state, newline, previous):
        if False:
            for i in range(10):
                print('nop')
        self.state = state.Clone()
        self.newline = newline
        self.previous = previous

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'StateNode(state=[\n{0}\n], newline={1})'.format(self.state, self.newline)
_OrderedPenalty = collections.namedtuple('OrderedPenalty', ['penalty', 'count'])
_QueueItem = collections.namedtuple('QueueItem', ['ordered_penalty', 'state_node'])

def _AnalyzeSolutionSpace(initial_state):
    if False:
        i = 10
        return i + 15
    "Analyze the entire solution space starting from initial_state.\n\n  This implements a variant of Dijkstra's algorithm on the graph that spans\n  the solution space (LineStates are the nodes). The algorithm tries to find\n  the shortest path (the one with the lowest penalty) from 'initial_state' to\n  the state where all tokens are placed.\n\n  Arguments:\n    initial_state: (format_decision_state.FormatDecisionState) The initial state\n      to start the search from.\n\n  Returns:\n    True if a formatting solution was found. False otherwise.\n  "
    count = 0
    seen = set()
    p_queue = []
    node = _StateNode(initial_state, False, None)
    heapq.heappush(p_queue, _QueueItem(_OrderedPenalty(0, count), node))
    count += 1
    while p_queue:
        item = p_queue[0]
        penalty = item.ordered_penalty.penalty
        node = item.state_node
        if not node.state.next_token:
            break
        heapq.heappop(p_queue)
        if count > 10000:
            node.state.ignore_stack_for_comparison = True
        before_seen_count = len(seen)
        seen.add(node.state)
        if before_seen_count == len(seen):
            continue
        count = _AddNextStateToQueue(penalty, node, False, count, p_queue)
        count = _AddNextStateToQueue(penalty, node, True, count, p_queue)
    if not p_queue:
        return False
    _ReconstructPath(initial_state, heapq.heappop(p_queue).state_node)
    return True

def _AddNextStateToQueue(penalty, previous_node, newline, count, p_queue):
    if False:
        i = 10
        return i + 15
    "Add the following state to the analysis queue.\n\n  Assume the current state is 'previous_node' and has been reached with a\n  penalty of 'penalty'. Insert a line break if 'newline' is True.\n\n  Arguments:\n    penalty: (int) The penalty associated with the path up to this point.\n    previous_node: (_StateNode) The last _StateNode inserted into the priority\n      queue.\n    newline: (bool) Add a newline if True.\n    count: (int) The number of elements in the queue.\n    p_queue: (heapq) The priority queue representing the solution space.\n\n  Returns:\n    The updated number of elements in the queue.\n  "
    must_split = previous_node.state.MustSplit()
    if newline and (not previous_node.state.CanSplit(must_split)):
        return count
    if not newline and must_split:
        return count
    node = _StateNode(previous_node.state, newline, previous_node)
    penalty += node.state.AddTokenToState(newline=newline, dry_run=True, must_split=must_split)
    heapq.heappush(p_queue, _QueueItem(_OrderedPenalty(penalty, count), node))
    return count + 1

def _ReconstructPath(initial_state, current):
    if False:
        return 10
    'Reconstruct the path through the queue with lowest penalty.\n\n  Arguments:\n    initial_state: (format_decision_state.FormatDecisionState) The initial state\n      to start the search from.\n    current: (_StateNode) The node in the decision graph that is the end point\n      of the path with the least penalty.\n  '
    path = collections.deque()
    while current.previous:
        path.appendleft(current)
        current = current.previous
    for node in path:
        initial_state.AddTokenToState(newline=node.newline, dry_run=False)
NESTED_DEPTH = []

def _FormatFirstToken(first_token, indent_depth, prev_line, final_lines):
    if False:
        print('Hello World!')
    "Format the first token in the logical line.\n\n  Add a newline and the required indent before the first token of the logical\n  line.\n\n  Arguments:\n    first_token: (format_token.FormatToken) The first token in the logical line.\n    indent_depth: (int) The line's indentation depth.\n    prev_line: (list of logical_line.LogicalLine) The logical line previous to\n      this line.\n    final_lines: (list of logical_line.LogicalLine) The logical lines that have\n      already been processed.\n  "
    global NESTED_DEPTH
    while NESTED_DEPTH and NESTED_DEPTH[-1] > indent_depth:
        NESTED_DEPTH.pop()
    first_nested = False
    if _IsClassOrDef(first_token):
        if not NESTED_DEPTH:
            NESTED_DEPTH = [indent_depth]
        elif NESTED_DEPTH[-1] < indent_depth:
            first_nested = True
            NESTED_DEPTH.append(indent_depth)
    first_token.AddWhitespacePrefix(_CalculateNumberOfNewlines(first_token, indent_depth, prev_line, final_lines, first_nested), indent_level=indent_depth)
NO_BLANK_LINES = 1
ONE_BLANK_LINE = 2
TWO_BLANK_LINES = 3

def _IsClassOrDef(tok):
    if False:
        while True:
            i = 10
    if tok.value in {'class', 'def', '@'}:
        return True
    return tok.next_token and tok.value == 'async' and (tok.next_token.value == 'def')

def _CalculateNumberOfNewlines(first_token, indent_depth, prev_line, final_lines, first_nested):
    if False:
        while True:
            i = 10
    "Calculate the number of newlines we need to add.\n\n  Arguments:\n    first_token: (format_token.FormatToken) The first token in the logical\n      line.\n    indent_depth: (int) The line's indentation depth.\n    prev_line: (list of logical_line.LogicalLine) The logical line previous to\n      this line.\n    final_lines: (list of logical_line.LogicalLine) The logical lines that have\n      already been processed.\n    first_nested: (boolean) Whether this is the first nested class or function.\n\n  Returns:\n    The number of newlines needed before the first token.\n  "
    if prev_line is None:
        if first_token.newlines is not None:
            first_token.newlines = None
        return 0
    if first_token.is_docstring:
        if prev_line.first.value == 'class' and style.Get('BLANK_LINE_BEFORE_CLASS_DOCSTRING'):
            return ONE_BLANK_LINE
        elif prev_line.first.value.startswith('#') and style.Get('BLANK_LINE_BEFORE_MODULE_DOCSTRING'):
            return ONE_BLANK_LINE
        return NO_BLANK_LINES
    if first_token.is_name and (not indent_depth):
        if prev_line.first.value in {'from', 'import'}:
            return 1 + style.Get('BLANK_LINES_BETWEEN_TOP_LEVEL_IMPORTS_AND_VARIABLES')
    prev_last_token = prev_line.last
    if prev_last_token.is_docstring:
        if not indent_depth and first_token.value in {'class', 'def', 'async'}:
            return 1 + style.Get('BLANK_LINES_AROUND_TOP_LEVEL_DEFINITION')
        if first_nested and (not style.Get('BLANK_LINE_BEFORE_NESTED_CLASS_OR_DEF')) and _IsClassOrDef(first_token):
            first_token.newlines = None
            return NO_BLANK_LINES
        if _NoBlankLinesBeforeCurrentToken(prev_last_token.value, first_token, prev_last_token):
            return NO_BLANK_LINES
        else:
            return ONE_BLANK_LINE
    if _IsClassOrDef(first_token):
        if not indent_depth:
            is_inline_comment = prev_last_token.whitespace_prefix.count('\n') == 0
            if not prev_line.disable and prev_last_token.is_comment and (not is_inline_comment):
                if _NoBlankLinesBeforeCurrentToken(prev_last_token.value, first_token, prev_last_token):
                    index = len(final_lines) - 1
                    while index > 0:
                        if not final_lines[index - 1].is_comment:
                            break
                        index -= 1
                    if final_lines[index - 1].first.value == '@':
                        final_lines[index].first.AdjustNewlinesBefore(NO_BLANK_LINES)
                    else:
                        prev_last_token.AdjustNewlinesBefore(1 + style.Get('BLANK_LINES_AROUND_TOP_LEVEL_DEFINITION'))
                    if first_token.newlines is not None:
                        first_token.newlines = None
                    return NO_BLANK_LINES
        elif _IsClassOrDef(prev_line.first):
            if first_nested and (not style.Get('BLANK_LINE_BEFORE_NESTED_CLASS_OR_DEF')):
                first_token.newlines = None
                return NO_BLANK_LINES
    if first_token.is_comment:
        first_token_lineno = first_token.lineno - first_token.value.count('\n')
    else:
        first_token_lineno = first_token.lineno
    prev_last_token_lineno = prev_last_token.lineno
    if prev_last_token.is_multiline_string:
        prev_last_token_lineno += prev_last_token.value.count('\n')
    if first_token_lineno - prev_last_token_lineno > 1:
        return ONE_BLANK_LINE
    return NO_BLANK_LINES

def _SingleOrMergedLines(lines):
    if False:
        print('Hello World!')
    'Generate the lines we want to format.\n\n  Arguments:\n    lines: (list of logical_line.LogicalLine) Lines we want to format.\n\n  Yields:\n    Either a single line, if the current line cannot be merged with the\n    succeeding line, or the next two lines merged into one line.\n  '
    index = 0
    last_was_merged = False
    while index < len(lines):
        if lines[index].disable:
            line = lines[index]
            index += 1
            while index < len(lines):
                column = line.last.column + 2
                if lines[index].lineno != line.lineno:
                    break
                if line.last.value != ':':
                    leaf = pytree.Leaf(type=token.SEMI, value=';', context=('', (line.lineno, column)))
                    line.AppendToken(format_token.FormatToken(leaf, pytree_utils.NodeName(leaf)))
                for tok in lines[index].tokens:
                    line.AppendToken(tok)
                index += 1
            yield line
        elif line_joiner.CanMergeMultipleLines(lines[index:], last_was_merged):
            next_line = lines[index + 1]
            for tok in next_line.tokens:
                lines[index].AppendToken(tok)
            if len(next_line.tokens) == 1 and next_line.first.is_multiline_string:
                lines[index].disable = True
            yield lines[index]
            index += 2
            last_was_merged = True
        else:
            yield lines[index]
            index += 1
            last_was_merged = False

def _NoBlankLinesBeforeCurrentToken(text, cur_token, prev_token):
    if False:
        for i in range(10):
            print('nop')
    "Determine if there are no blank lines before the current token.\n\n  The previous token is a docstring or comment. The prev_token_lineno is the\n  start of the text of that token. Counting the number of newlines in its text\n  gives us the extent and thus where the line number of the end of the\n  docstring or comment. After that, we just compare it to the current token's\n  line number to see if there are blank lines between them.\n\n  Arguments:\n    text: (unicode) The text of the docstring or comment before the current\n      token.\n    cur_token: (format_token.FormatToken) The current token in the logical line.\n    prev_token: (format_token.FormatToken) The previous token in the logical\n      line.\n\n  Returns:\n    True if there is no blank line before the current token.\n  "
    cur_token_lineno = cur_token.lineno
    if cur_token.is_comment:
        cur_token_lineno -= cur_token.value.count('\n')
    num_newlines = text.count('\n') if not prev_token.is_comment else 0
    return prev_token.lineno + num_newlines == cur_token_lineno - 1