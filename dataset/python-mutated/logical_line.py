"""LogicalLine primitive for formatting.

A logical line is the containing data structure produced by the parser. It
collects all nodes (stored in FormatToken objects) that could appear on a single
line if there were no line length restrictions. It's then used by the parser to
perform the wrapping required to comply with the style guide.
"""
from yapf_third_party._ylib2to3.fixer_util import syms as python_symbols
from yapf.pytree import pytree_utils
from yapf.pytree import split_penalty
from yapf.yapflib import format_token
from yapf.yapflib import style
from yapf.yapflib import subtypes

class LogicalLine(object):
    """Represents a single logical line in the output.

  Attributes:
    depth: indentation depth of this line. This is just a numeric value used to
      distinguish lines that are more deeply nested than others. It is not the
      actual amount of spaces, which is style-dependent.
  """

    def __init__(self, depth, tokens=None):
        if False:
            print('Hello World!')
        'Constructor.\n\n    Creates a new logical line with the given depth an initial list of tokens.\n    Constructs the doubly-linked lists for format tokens using their built-in\n    next_token and previous_token attributes.\n\n    Arguments:\n      depth: indentation depth of this line\n      tokens: initial list of tokens\n    '
        self.depth = depth
        self._tokens = tokens or []
        self.disable = False
        if self._tokens:
            for (index, tok) in enumerate(self._tokens[1:]):
                tok.previous_token = self._tokens[index]
                self._tokens[index].next_token = tok

    def CalculateFormattingInformation(self):
        if False:
            print('Hello World!')
        'Calculate the split penalty and total length for the tokens.'
        self.first.spaces_required_before = 1
        self.first.total_length = len(self.first.value)
        prev_token = self.first
        prev_length = self.first.total_length
        for token in self._tokens[1:]:
            if token.spaces_required_before == 0 and _SpaceRequiredBetween(prev_token, token, self.disable):
                token.spaces_required_before = 1
            tok_len = len(token.value) if not token.is_pseudo else 0
            spaces_required_before = token.spaces_required_before
            if isinstance(spaces_required_before, list):
                assert token.is_comment, token
                spaces_required_before = 0
            token.total_length = prev_length + tok_len + spaces_required_before
            token.split_penalty += _SplitPenalty(prev_token, token)
            token.must_break_before = _MustBreakBefore(prev_token, token)
            token.can_break_before = token.must_break_before or _CanBreakBefore(prev_token, token)
            prev_length = token.total_length
            prev_token = token

    def Split(self):
        if False:
            i = 10
            return i + 15
        'Split the line at semicolons.'
        if not self.has_semicolon or self.disable:
            return [self]
        llines = []
        lline = LogicalLine(self.depth)
        for tok in self._tokens:
            if tok.value == ';':
                llines.append(lline)
                lline = LogicalLine(self.depth)
            else:
                lline.AppendToken(tok)
        if lline.tokens:
            llines.append(lline)
        for lline in llines:
            lline.first.previous_token = None
            lline.last.next_token = None
        return llines

    def AppendToken(self, token):
        if False:
            i = 10
            return i + 15
        'Append a new FormatToken to the tokens contained in this line.'
        if self._tokens:
            token.previous_token = self.last
            self.last.next_token = token
        self._tokens.append(token)

    @property
    def first(self):
        if False:
            print('Hello World!')
        'Returns the first non-whitespace token.'
        return self._tokens[0]

    @property
    def last(self):
        if False:
            while True:
                i = 10
        'Returns the last non-whitespace token.'
        return self._tokens[-1]

    def AsCode(self, indent_per_depth=2):
        if False:
            for i in range(10):
                print('nop')
        'Return a "code" representation of this line.\n\n    The code representation shows how the line would be printed out as code.\n\n    TODO(eliben): for now this is rudimentary for debugging - once we add\n    formatting capabilities, this method will have other uses (not all tokens\n    have spaces around them, for example).\n\n    Arguments:\n      indent_per_depth: how much spaces to indent per depth level.\n\n    Returns:\n      A string representing the line as code.\n    '
        indent = ' ' * indent_per_depth * self.depth
        tokens_str = ' '.join((tok.value for tok in self._tokens))
        return indent + tokens_str

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.AsCode()

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        tokens_repr = ','.join(('{0}({1!r})'.format(tok.name, tok.value) for tok in self._tokens))
        return 'LogicalLine(depth={0}, tokens=[{1}])'.format(self.depth, tokens_repr)

    @property
    def tokens(self):
        if False:
            while True:
                i = 10
        'Access the tokens contained within this line.\n\n    The caller must not modify the tokens list returned by this method.\n\n    Returns:\n      List of tokens in this line.\n    '
        return self._tokens

    @property
    def lineno(self):
        if False:
            i = 10
            return i + 15
        'Return the line number of this logical line.\n\n    Returns:\n      The line number of the first token in this logical line.\n    '
        return self.first.lineno

    @property
    def start(self):
        if False:
            for i in range(10):
                print('nop')
        'The start of the logical line.\n\n    Returns:\n      A tuple of the starting line number and column.\n    '
        return (self.first.lineno, self.first.column)

    @property
    def end(self):
        if False:
            for i in range(10):
                print('nop')
        'The end of the logical line.\n\n    Returns:\n      A tuple of the ending line number and column.\n    '
        return (self.last.lineno, self.last.column + len(self.last.value))

    @property
    def is_comment(self):
        if False:
            for i in range(10):
                print('nop')
        return self.first.is_comment

    @property
    def has_semicolon(self):
        if False:
            return 10
        return any((tok.value == ';' for tok in self._tokens))

def _IsIdNumberStringToken(tok):
    if False:
        return 10
    return tok.is_keyword or tok.is_name or tok.is_number or tok.is_string

def _IsUnaryOperator(tok):
    if False:
        print('Hello World!')
    return subtypes.UNARY_OPERATOR in tok.subtypes

def _HasPrecedence(tok):
    if False:
        for i in range(10):
            print('nop')
    'Whether a binary operation has precedence within its context.'
    node = tok.node
    ancestor = node.parent.parent
    while ancestor is not None:
        predecessor_type = pytree_utils.NodeName(ancestor)
        if predecessor_type in ['arith_expr', 'term']:
            return True
        if predecessor_type != 'atom':
            return False
        ancestor = ancestor.parent

def _PriorityIndicatingNoSpace(tok):
    if False:
        i = 10
        return i + 15
    'Whether to remove spaces around an operator due to precedence.'
    if not tok.is_arithmetic_op or not tok.is_simple_expr:
        return False
    return _HasPrecedence(tok)

def _IsSubscriptColonAndValuePair(token1, token2):
    if False:
        i = 10
        return i + 15
    return (token1.is_number or token1.is_name) and token2.is_subscript_colon

def _SpaceRequiredBetween(left, right, is_line_disabled):
    if False:
        return 10
    'Return True if a space is required between the left and right token.'
    lval = left.value
    rval = right.value
    if left.is_pseudo and _IsIdNumberStringToken(right) and left.previous_token and _IsIdNumberStringToken(left.previous_token):
        return True
    if left.is_pseudo or right.is_pseudo:
        if left.OpensScope():
            return True
        return False
    if left.is_continuation or right.is_continuation:
        return False
    if right.name in pytree_utils.NONSEMANTIC_TOKENS:
        return False
    if _IsIdNumberStringToken(left) and _IsIdNumberStringToken(right):
        return True
    if lval == ',' and rval == ':':
        return True
    if style.Get('SPACE_INSIDE_BRACKETS'):
        if left.OpensScope() and rval == ':':
            return True
        if right.ClosesScope() and lval == ':':
            return True
    if style.Get('SPACES_AROUND_SUBSCRIPT_COLON') and (_IsSubscriptColonAndValuePair(left, right) or _IsSubscriptColonAndValuePair(right, left)):
        return True
    if rval in ':,':
        return False
    if lval == ',' and rval in ']})':
        return style.Get('SPACE_BETWEEN_ENDING_COMMA_AND_CLOSING_BRACKET')
    if lval == ',':
        return True
    if lval == 'from' and rval == '.':
        return True
    if lval == '.' and rval == 'import':
        return True
    if lval == '=' and rval in {'.', ',,,'} and (subtypes.DEFAULT_OR_NAMED_ASSIGN not in left.subtypes):
        return True
    if lval == ':' and rval in {'.', '...'}:
        return True
    if (right.is_keyword or right.is_name) and (left.is_keyword or left.is_name):
        return True
    if subtypes.SUBSCRIPT_COLON in left.subtypes or subtypes.SUBSCRIPT_COLON in right.subtypes:
        return False
    if subtypes.TYPED_NAME in left.subtypes or subtypes.TYPED_NAME in right.subtypes:
        return True
    if left.is_string:
        if rval == '=' and subtypes.DEFAULT_OR_NAMED_ASSIGN_ARG_LIST in right.subtypes:
            return False
        if rval not in '[)]}.' and (not right.is_binary_op):
            return True
        if right.ClosesScope():
            return style.Get('SPACE_INSIDE_BRACKETS')
        if subtypes.SUBSCRIPT_BRACKET in right.subtypes:
            return False
    if left.is_binary_op and lval != '**' and _IsUnaryOperator(right):
        return True
    if left.is_keyword and _IsUnaryOperator(right):
        return True
    if _IsUnaryOperator(left) and _IsUnaryOperator(right):
        return False
    if left.is_binary_op or right.is_binary_op:
        if lval == '**' or rval == '**':
            return style.Get('SPACES_AROUND_POWER_OPERATOR')
        block_list = style.Get('NO_SPACES_AROUND_SELECTED_BINARY_OPERATORS')
        if lval in block_list or rval in block_list:
            return False
        if style.Get('ARITHMETIC_PRECEDENCE_INDICATION'):
            if _PriorityIndicatingNoSpace(left) or _PriorityIndicatingNoSpace(right):
                return False
            else:
                return True
        else:
            return True
    if _IsUnaryOperator(left) and lval != 'not' and (right.is_name or right.is_number or rval == '('):
        return False
    if subtypes.DEFAULT_OR_NAMED_ASSIGN in left.subtypes and subtypes.TYPED_NAME not in right.subtypes:
        return style.Get('SPACES_AROUND_DEFAULT_OR_NAMED_ASSIGN')
    if subtypes.DEFAULT_OR_NAMED_ASSIGN in right.subtypes and subtypes.TYPED_NAME not in left.subtypes:
        return style.Get('SPACES_AROUND_DEFAULT_OR_NAMED_ASSIGN')
    if subtypes.VARARGS_LIST in left.subtypes or subtypes.VARARGS_LIST in right.subtypes:
        return False
    if subtypes.VARARGS_STAR in left.subtypes or subtypes.KWARGS_STAR_STAR in left.subtypes:
        return False
    if lval == '@' and subtypes.DECORATOR in left.subtypes:
        return False
    if left.is_keyword and rval == '.':
        return lval not in {'None', 'print'}
    if lval == '.' and right.is_keyword:
        return rval not in {'None', 'print'}
    if lval == '.' or rval == '.':
        return False
    if lval == '(' and rval == ')' or (lval == '[' and rval == ']') or (lval == '{' and rval == '}'):
        return False
    if not is_line_disabled and (left.OpensScope() or right.ClosesScope()):
        if style.GetOrDefault('SPACES_AROUND_DICT_DELIMITERS', False) and (lval == '{' and _IsDictListTupleDelimiterTok(left, is_opening=True) or (rval == '}' and _IsDictListTupleDelimiterTok(right, is_opening=False))):
            return True
        if style.GetOrDefault('SPACES_AROUND_LIST_DELIMITERS', False) and (lval == '[' and _IsDictListTupleDelimiterTok(left, is_opening=True) or (rval == ']' and _IsDictListTupleDelimiterTok(right, is_opening=False))):
            return True
        if style.GetOrDefault('SPACES_AROUND_TUPLE_DELIMITERS', False) and (lval == '(' and _IsDictListTupleDelimiterTok(left, is_opening=True) or (rval == ')' and _IsDictListTupleDelimiterTok(right, is_opening=False))):
            return True
    if left.OpensScope() and right.OpensScope():
        return style.Get('SPACE_INSIDE_BRACKETS')
    if left.ClosesScope() and right.ClosesScope():
        return style.Get('SPACE_INSIDE_BRACKETS')
    if left.ClosesScope() and rval in '([':
        return False
    if left.OpensScope() and _IsIdNumberStringToken(right):
        return style.Get('SPACE_INSIDE_BRACKETS')
    if left.is_name and rval in '([':
        return False
    if right.ClosesScope():
        return style.Get('SPACE_INSIDE_BRACKETS')
    if lval == 'print' and rval == '(':
        return False
    if left.OpensScope() and _IsUnaryOperator(right):
        return style.Get('SPACE_INSIDE_BRACKETS')
    if left.OpensScope() and (subtypes.VARARGS_STAR in right.subtypes or subtypes.KWARGS_STAR_STAR in right.subtypes):
        return style.Get('SPACE_INSIDE_BRACKETS')
    if rval == ';':
        return False
    if lval == '(' and rval == 'await':
        return style.Get('SPACE_INSIDE_BRACKETS')
    return True

def _MustBreakBefore(prev_token, cur_token):
    if False:
        i = 10
        return i + 15
    'Return True if a line break is required before the current token.'
    if prev_token.is_comment or (prev_token.previous_token and prev_token.is_pseudo and prev_token.previous_token.is_comment):
        return True
    if cur_token.is_string and prev_token.is_string and IsSurroundedByBrackets(cur_token):
        return True
    return cur_token.must_break_before

def _CanBreakBefore(prev_token, cur_token):
    if False:
        return 10
    'Return True if a line break may occur before the current token.'
    pval = prev_token.value
    cval = cur_token.value
    if pval == 'yield' and cval == 'from':
        return False
    if pval in {'async', 'await'} and cval in {'def', 'with', 'for'}:
        return False
    if cur_token.split_penalty >= split_penalty.UNBREAKABLE:
        return False
    if pval == '@':
        return False
    if cval == ':':
        return False
    if cval == ',':
        return False
    if prev_token.is_name and cval == '(':
        return False
    if prev_token.is_name and cval == '[':
        return False
    if cur_token.is_comment and prev_token.lineno == cur_token.lineno:
        return False
    if subtypes.UNARY_OPERATOR in prev_token.subtypes:
        return False
    if not style.Get('ALLOW_SPLIT_BEFORE_DEFAULT_OR_NAMED_ASSIGNS'):
        if subtypes.DEFAULT_OR_NAMED_ASSIGN in cur_token.subtypes or subtypes.DEFAULT_OR_NAMED_ASSIGN in prev_token.subtypes:
            return False
    return True

def IsSurroundedByBrackets(tok):
    if False:
        return 10
    'Return True if the token is surrounded by brackets.'
    paren_count = 0
    brace_count = 0
    sq_bracket_count = 0
    previous_token = tok.previous_token
    while previous_token:
        if previous_token.value == ')':
            paren_count -= 1
        elif previous_token.value == '}':
            brace_count -= 1
        elif previous_token.value == ']':
            sq_bracket_count -= 1
        if previous_token.value == '(':
            if paren_count == 0:
                return previous_token
            paren_count += 1
        elif previous_token.value == '{':
            if brace_count == 0:
                return previous_token
            brace_count += 1
        elif previous_token.value == '[':
            if sq_bracket_count == 0:
                return previous_token
            sq_bracket_count += 1
        previous_token = previous_token.previous_token
    return None

def _IsDictListTupleDelimiterTok(tok, is_opening):
    if False:
        return 10
    assert tok
    if tok.matching_bracket is None:
        return False
    if is_opening:
        open_tok = tok
        close_tok = tok.matching_bracket
    else:
        open_tok = tok.matching_bracket
        close_tok = tok
    if open_tok.next_token == close_tok:
        return False
    assert open_tok.next_token.node
    assert open_tok.next_token.node.parent
    return open_tok.next_token.node.parent.type in [python_symbols.dictsetmaker, python_symbols.listmaker, python_symbols.testlist_gexp]
_LOGICAL_OPERATORS = frozenset({'and', 'or'})
_BITWISE_OPERATORS = frozenset({'&', '|', '^'})
_ARITHMETIC_OPERATORS = frozenset({'+', '-', '*', '/', '%', '//', '@'})

def _SplitPenalty(prev_token, cur_token):
    if False:
        print('Hello World!')
    'Return the penalty for breaking the line before the current token.'
    pval = prev_token.value
    cval = cur_token.value
    if pval == 'not':
        return split_penalty.UNBREAKABLE
    if cur_token.node_split_penalty > 0:
        return cur_token.node_split_penalty
    if style.Get('SPLIT_BEFORE_LOGICAL_OPERATOR'):
        if pval in _LOGICAL_OPERATORS:
            return style.Get('SPLIT_PENALTY_LOGICAL_OPERATOR')
        if cval in _LOGICAL_OPERATORS:
            return 0
    else:
        if pval in _LOGICAL_OPERATORS:
            return 0
        if cval in _LOGICAL_OPERATORS:
            return style.Get('SPLIT_PENALTY_LOGICAL_OPERATOR')
    if style.Get('SPLIT_BEFORE_BITWISE_OPERATOR'):
        if pval in _BITWISE_OPERATORS:
            return style.Get('SPLIT_PENALTY_BITWISE_OPERATOR')
        if cval in _BITWISE_OPERATORS:
            return 0
    else:
        if pval in _BITWISE_OPERATORS:
            return 0
        if cval in _BITWISE_OPERATORS:
            return style.Get('SPLIT_PENALTY_BITWISE_OPERATOR')
    if subtypes.COMP_FOR in cur_token.subtypes or subtypes.COMP_IF in cur_token.subtypes:
        return 0
    if subtypes.UNARY_OPERATOR in prev_token.subtypes:
        return style.Get('SPLIT_PENALTY_AFTER_UNARY_OPERATOR')
    if pval == ',':
        return 0
    if pval == '**' or cval == '**':
        return split_penalty.STRONGLY_CONNECTED
    if subtypes.VARARGS_STAR in prev_token.subtypes or subtypes.KWARGS_STAR_STAR in prev_token.subtypes:
        return split_penalty.UNBREAKABLE
    if prev_token.OpensScope() and cval != '(':
        return style.Get('SPLIT_PENALTY_AFTER_OPENING_BRACKET')
    if cval == ':':
        return split_penalty.UNBREAKABLE
    if cval == '=':
        return split_penalty.UNBREAKABLE
    if subtypes.DEFAULT_OR_NAMED_ASSIGN in prev_token.subtypes or subtypes.DEFAULT_OR_NAMED_ASSIGN in cur_token.subtypes:
        return split_penalty.UNBREAKABLE
    if cval == '==':
        return split_penalty.STRONGLY_CONNECTED
    if cur_token.ClosesScope():
        return 100
    return 0