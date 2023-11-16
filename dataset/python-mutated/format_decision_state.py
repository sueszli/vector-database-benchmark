"""Implements a format decision state object that manages whitespace decisions.

Each token is processed one at a time, at which point its whitespace formatting
decisions are made. A graph of potential whitespace formatting is created,
where each node in the graph is a format decision state object. The heuristic
tries formatting the token with and without a newline before it to determine
which one has the least penalty. Therefore, the format decision state object for
each decision needs to be its own unique copy.

Once the heuristic determines the best formatting, it makes a non-dry run pass
through the code to commit the whitespace formatting.

  FormatDecisionState: main class exported by this module.
"""
from yapf.pytree import split_penalty
from yapf.pytree.pytree_utils import NodeName
from yapf.yapflib import logical_line
from yapf.yapflib import object_state
from yapf.yapflib import style
from yapf.yapflib import subtypes

class FormatDecisionState(object):
    """The current state when indenting a logical line.

  The FormatDecisionState object is meant to be copied instead of referenced.

  Attributes:
    first_indent: The indent of the first token.
    column: The number of used columns in the current line.
    line: The logical line we're currently processing.
    next_token: The next token to be formatted.
    paren_level: The level of nesting inside (), [], and {}.
    lowest_level_on_line: The lowest paren_level on the current line.
    stack: A stack (of _ParenState) keeping track of properties applying to
      parenthesis levels.
    comp_stack: A stack (of ComprehensionState) keeping track of properties
      applying to comprehensions.
    param_list_stack: A stack (of ParameterListState) keeping track of
      properties applying to function parameter lists.
    ignore_stack_for_comparison: Ignore the stack of _ParenState for state
      comparison.
    column_limit: The column limit specified by the style.
  """

    def __init__(self, line, first_indent):
        if False:
            return 10
        "Initializer.\n\n    Initializes to the state after placing the first token from 'line' at\n    'first_indent'.\n\n    Arguments:\n      line: (LogicalLine) The logical line we're currently processing.\n      first_indent: (int) The indent of the first token.\n    "
        self.next_token = line.first
        self.column = first_indent
        self.line = line
        self.paren_level = 0
        self.lowest_level_on_line = 0
        self.ignore_stack_for_comparison = False
        self.stack = [_ParenState(first_indent, first_indent)]
        self.comp_stack = []
        self.param_list_stack = []
        self.first_indent = first_indent
        self.column_limit = style.Get('COLUMN_LIMIT')

    def Clone(self):
        if False:
            while True:
                i = 10
        'Clones a FormatDecisionState object.'
        new = FormatDecisionState(self.line, self.first_indent)
        new.next_token = self.next_token
        new.column = self.column
        new.line = self.line
        new.paren_level = self.paren_level
        new.line.depth = self.line.depth
        new.lowest_level_on_line = self.lowest_level_on_line
        new.ignore_stack_for_comparison = self.ignore_stack_for_comparison
        new.first_indent = self.first_indent
        new.stack = [state.Clone() for state in self.stack]
        new.comp_stack = [state.Clone() for state in self.comp_stack]
        new.param_list_stack = [state.Clone() for state in self.param_list_stack]
        return new

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.next_token == other.next_token and self.column == other.column and (self.paren_level == other.paren_level) and (self.line.depth == other.line.depth) and (self.lowest_level_on_line == other.lowest_level_on_line) and (self.ignore_stack_for_comparison or other.ignore_stack_for_comparison or (self.stack == other.stack and self.comp_stack == other.comp_stack and (self.param_list_stack == other.param_list_stack)))

    def __ne__(self, other):
        if False:
            return 10
        return not self == other

    def __hash__(self):
        if False:
            return 10
        return hash((self.next_token, self.column, self.paren_level, self.line.depth, self.lowest_level_on_line))

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'column::%d, next_token::%s, paren_level::%d, stack::[\n\t%s' % (self.column, repr(self.next_token), self.paren_level, '\n\t'.join((repr(s) for s in self.stack)) + ']')

    def CanSplit(self, must_split):
        if False:
            print('Hello World!')
        'Determine if we can split before the next token.\n\n    Arguments:\n      must_split: (bool) A newline was required before this token.\n\n    Returns:\n      True if the line can be split before the next token.\n    '
        current = self.next_token
        previous = current.previous_token
        if current.is_pseudo:
            return False
        if not must_split and subtypes.DICTIONARY_KEY_PART in current.subtypes and (subtypes.DICTIONARY_KEY not in current.subtypes) and (not style.Get('ALLOW_MULTILINE_DICTIONARY_KEYS')):
            return False
        if not must_split and subtypes.DICTIONARY_VALUE in current.subtypes and (not style.Get('ALLOW_SPLIT_BEFORE_DICT_VALUE')):
            return False
        if previous and previous.value == '(' and (current.value == ')'):
            token = previous.previous_token
            while token:
                prev = token.previous_token
                if not prev or prev.name not in {'NAME', 'DOT'}:
                    break
                token = token.previous_token
            if token and subtypes.DICTIONARY_VALUE in token.subtypes:
                if not style.Get('ALLOW_SPLIT_BEFORE_DICT_VALUE'):
                    return False
        if previous and previous.value == '.' and (current.value == '.'):
            return False
        return current.can_break_before

    def MustSplit(self):
        if False:
            print('Hello World!')
        'Returns True if the line must split before the next token.'
        current = self.next_token
        previous = current.previous_token
        if current.is_pseudo:
            return False
        if current.must_break_before:
            return True
        if not previous:
            return False
        if style.Get('SPLIT_ALL_COMMA_SEPARATED_VALUES') and previous.value == ',':
            if subtypes.COMP_FOR in current.subtypes or subtypes.LAMBDEF in current.subtypes:
                return False
            return True
        if style.Get('FORCE_MULTILINE_DICT') and subtypes.DICTIONARY_KEY in current.subtypes and (not current.is_comment):
            return True
        if style.Get('SPLIT_ALL_TOP_LEVEL_COMMA_SEPARATED_VALUES') and previous.value == ',':
            if subtypes.COMP_FOR in current.subtypes or subtypes.LAMBDEF in current.subtypes:
                return False
            opening = _GetOpeningBracket(current)
            if not opening:
                return True
            if current.is_comment:
                return False
            if current != opening.matching_bracket:
                return not self._ContainerFitsOnStartLine(opening)
        if self.stack[-1].split_before_closing_bracket and (current.value in '}]' and style.Get('SPLIT_BEFORE_CLOSING_BRACKET') or (current.value in '}])' and style.Get('INDENT_CLOSING_BRACKETS'))):
            if subtypes.SUBSCRIPT_BRACKET not in current.subtypes or (previous.value == ',' and (not style.Get('DISABLE_ENDING_COMMA_HEURISTIC'))):
                return current.node_split_penalty != split_penalty.UNBREAKABLE
        if current.value == ')' and previous.value == ',' and (not _IsSingleElementTuple(current.matching_bracket)):
            return True
        if style.Get('SPLIT_BEFORE_FIRST_ARGUMENT') and _IsCompoundStatement(self.line.first) and (not _IsFunctionDef(self.line.first)):
            return False
        if style.Get('DEDENT_CLOSING_BRACKETS') or style.Get('INDENT_CLOSING_BRACKETS') or style.Get('SPLIT_BEFORE_FIRST_ARGUMENT'):
            bracket = current if current.ClosesScope() else previous
            if subtypes.SUBSCRIPT_BRACKET not in bracket.subtypes:
                if bracket.OpensScope():
                    if style.Get('COALESCE_BRACKETS'):
                        if current.OpensScope():
                            return False
                    if not _IsLastScopeInLine(bracket) or logical_line.IsSurroundedByBrackets(bracket):
                        last_token = bracket.matching_bracket
                    else:
                        last_token = _LastTokenInLine(bracket.matching_bracket)
                    if not self._FitsOnLine(bracket, last_token):
                        self.stack[-1].split_before_closing_bracket = True
                        return True
                elif (style.Get('DEDENT_CLOSING_BRACKETS') or style.Get('INDENT_CLOSING_BRACKETS')) and current.ClosesScope():
                    return self.stack[-1].split_before_closing_bracket
        if style.Get('SPLIT_BEFORE_EXPRESSION_AFTER_OPENING_PAREN') and current.is_name:

            def SurroundedByParens(token):
                if False:
                    for i in range(10):
                        print('nop')
                "Check if it's an expression surrounded by parentheses."
                while token:
                    if token.value == ',':
                        return False
                    if token.value == ')':
                        return not token.next_token
                    if token.OpensScope():
                        token = token.matching_bracket.next_token
                    else:
                        token = token.next_token
                return False
            if previous.value == '(' and (not previous.is_pseudo) and (not logical_line.IsSurroundedByBrackets(previous)):
                pptoken = previous.previous_token
                if pptoken and (not pptoken.is_name) and (not pptoken.is_keyword) and SurroundedByParens(current):
                    return True
        if (current.is_name or current.is_string) and previous.value == ',':
            func_call_or_string_format = False
            tok = current.next_token
            if current.is_name:
                while tok and (tok.is_name or tok.value == '.'):
                    tok = tok.next_token
                func_call_or_string_format = tok and tok.value == '('
            elif current.is_string:
                while tok and tok.is_string:
                    tok = tok.next_token
                func_call_or_string_format = tok and tok.value == '%'
            if func_call_or_string_format:
                open_bracket = logical_line.IsSurroundedByBrackets(current)
                if open_bracket:
                    if open_bracket.value in '[{':
                        if not self._FitsOnLine(open_bracket, open_bracket.matching_bracket):
                            return True
                    elif tok.value == '(':
                        if not self._FitsOnLine(current, tok.matching_bracket):
                            return True
        if current.OpensScope() and previous.value == ',' and (subtypes.DICTIONARY_KEY not in current.next_token.subtypes):
            open_bracket = logical_line.IsSurroundedByBrackets(current)
            if open_bracket and open_bracket.value in '[{' and (subtypes.SUBSCRIPT_BRACKET not in open_bracket.subtypes):
                if not self._FitsOnLine(current, current.matching_bracket):
                    return True
        if style.Get('EACH_DICT_ENTRY_ON_SEPARATE_LINE') and subtypes.DICTIONARY_KEY in current.subtypes and (not current.is_comment):
            if previous.value == '{' and previous.previous_token:
                opening = _GetOpeningBracket(previous.previous_token)
                if opening and opening.value == '(' and opening.previous_token and opening.previous_token.is_name:
                    if self._FitsOnLine(previous, previous.matching_bracket) and previous.matching_bracket.next_token and (not opening.matching_bracket.next_token or opening.matching_bracket.next_token.value != '.') and _ScopeHasNoCommas(previous):
                        return False
            return True
        if style.Get('SPLIT_BEFORE_DICT_SET_GENERATOR') and subtypes.DICT_SET_GENERATOR in current.subtypes:
            return True
        if subtypes.DICTIONARY_VALUE in current.subtypes or (previous.is_pseudo and previous.value == '(' and (not current.is_comment)):
            if not current.OpensScope():
                opening = _GetOpeningBracket(current)
                if not self._EachDictEntryFitsOnOneLine(opening):
                    return style.Get('ALLOW_SPLIT_BEFORE_DICT_VALUE')
        if previous.value == '{':
            closing = previous.matching_bracket
            if not self._FitsOnLine(previous, closing) and closing.previous_token.value == ',':
                self.stack[-1].split_before_closing_bracket = True
                return True
        if style.Get('SPLIT_ARGUMENTS_WHEN_COMMA_TERMINATED'):
            opening = _GetOpeningBracket(current)
            if opening and opening.previous_token and opening.previous_token.is_name:
                if previous.value in '(,':
                    if opening.matching_bracket.previous_token.value == ',':
                        return True
        if style.Get('SPLIT_BEFORE_NAMED_ASSIGNS') and (not current.is_comment) and (subtypes.DEFAULT_OR_NAMED_ASSIGN_ARG_LIST in current.subtypes):
            if previous.value not in {'=', ':', '*', '**'} and current.value not in ':=,)' and (not _IsFunctionDefinition(previous)):
                if previous.value == '(':
                    if self._FitsOnLine(previous, previous.matching_bracket) and logical_line.IsSurroundedByBrackets(previous):
                        return False
                    if not style.Get('SPLIT_BEFORE_EXPRESSION_AFTER_OPENING_PAREN') and (not style.Get('SPLIT_BEFORE_FIRST_ARGUMENT')):
                        return False
                    column = self.column - self.stack[-1].last_space
                    return column > style.Get('CONTINUATION_INDENT_WIDTH')
                opening = _GetOpeningBracket(current)
                if opening:
                    return not self._ContainerFitsOnStartLine(opening)
        if current.value not in '{)' and previous.value == '(' and self._ArgumentListHasDictionaryEntry(current):
            return True
        if (current.is_name or current.value in {'*', '**'}) and previous.value == ',':
            opening = _GetOpeningBracket(current)
            if opening and opening.value == '(' and opening.previous_token and (opening.previous_token.is_name or opening.previous_token.value in {'*', '**'}):
                is_func_call = False
                opening = current
                while opening:
                    if opening.value == '(':
                        is_func_call = True
                        break
                    if not (opening.is_name or opening.value in {'*', '**'}) and opening.value != '.':
                        break
                    opening = opening.next_token
                if is_func_call:
                    if not self._FitsOnLine(current, opening.matching_bracket) or (opening.matching_bracket.next_token and opening.matching_bracket.next_token.value != ',' and (not opening.matching_bracket.next_token.ClosesScope())):
                        return True
        pprevious = previous.previous_token
        if current.value == '{' and previous.value == '(' and pprevious and pprevious.is_name:
            dict_end = current.matching_bracket
            next_token = dict_end.next_token
            if next_token.value == ',' and (not self._FitsOnLine(current, dict_end)):
                return True
        if current.is_name and pprevious and pprevious.is_name and (previous.value == '('):
            if not self._FitsOnLine(previous, previous.matching_bracket) and _IsFunctionCallWithArguments(current):
                if style.Get('SPLIT_BEFORE_EXPRESSION_AFTER_OPENING_PAREN') or style.Get('SPLIT_BEFORE_FIRST_ARGUMENT'):
                    return True
                opening = _GetOpeningBracket(current)
                if opening and opening.value == '(' and opening.previous_token and (opening.previous_token.is_name or opening.previous_token.value in {'*', '**'}):
                    is_func_call = False
                    opening = current
                    while opening:
                        if opening.value == '(':
                            is_func_call = True
                            break
                        if not (opening.is_name or opening.value in {'*', '**'}) and opening.value != '.':
                            break
                        opening = opening.next_token
                    if is_func_call:
                        if not self._FitsOnLine(current, opening.matching_bracket) or (opening.matching_bracket.next_token and opening.matching_bracket.next_token.value != ',' and (not opening.matching_bracket.next_token.ClosesScope())):
                            return True
        if previous.OpensScope() and (not current.OpensScope()) and (not current.is_comment) and (subtypes.SUBSCRIPT_BRACKET not in previous.subtypes):
            if pprevious and (not pprevious.is_keyword) and (not pprevious.is_name):
                token = current
                while token != previous.matching_bracket:
                    if token.is_comment:
                        return True
                    token = token.next_token
            if previous.value == '(':
                pptoken = previous.previous_token
                if not pptoken or not pptoken.is_name:
                    if self._FitsOnLine(previous, previous.matching_bracket):
                        return False
                elif not self._FitsOnLine(previous, previous.matching_bracket):
                    if len(previous.container_elements) == 1:
                        return False
                    elements = previous.container_elements + [previous.matching_bracket]
                    i = 1
                    while i < len(elements):
                        if not elements[i - 1].OpensScope() and (not self._FitsOnLine(elements[i - 1], elements[i])):
                            return True
                        i += 1
                    if (self.column_limit - self.column) / float(self.column_limit) < 0.3:
                        return True
            elif not self._FitsOnLine(previous, previous.matching_bracket):
                return True
        if style.Get('SPLIT_BEFORE_BITWISE_OPERATOR') and current.value in '&|' and (previous.lineno < current.lineno):
            return True
        if current.is_comment and previous.lineno < current.lineno - current.value.count('\n'):
            return True
        return False

    def AddTokenToState(self, newline, dry_run, must_split=False):
        if False:
            while True:
                i = 10
        "Add a token to the format decision state.\n\n    Allow the heuristic to try out adding the token with and without a newline.\n    Later on, the algorithm will determine which one has the lowest penalty.\n\n    Arguments:\n      newline: (bool) Add the token on a new line if True.\n      dry_run: (bool) Don't commit whitespace changes to the FormatToken if\n        True.\n      must_split: (bool) A newline was required before this token.\n\n    Returns:\n      The penalty of splitting after the current token.\n    "
        self._PushParameterListState(newline)
        penalty = 0
        if newline:
            penalty = self._AddTokenOnNewline(dry_run, must_split)
        else:
            self._AddTokenOnCurrentLine(dry_run)
        penalty += self._CalculateComprehensionState(newline)
        penalty += self._CalculateParameterListState(newline)
        return self.MoveStateToNextToken() + penalty

    def _AddTokenOnCurrentLine(self, dry_run):
        if False:
            i = 10
            return i + 15
        'Puts the token on the current line.\n\n    Appends the next token to the state and updates information necessary for\n    indentation.\n\n    Arguments:\n      dry_run: (bool) Commit whitespace changes to the FormatToken if True.\n    '
        current = self.next_token
        previous = current.previous_token
        spaces = current.spaces_required_before
        if isinstance(spaces, list):
            spaces = 0
        if not dry_run:
            current.AddWhitespacePrefix(newlines_before=0, spaces=spaces)
        if previous.OpensScope():
            if not current.is_comment:
                self.stack[-1].closing_scope_indent = self.column - 1
                if style.Get('ALIGN_CLOSING_BRACKET_WITH_VISUAL_INDENT'):
                    self.stack[-1].closing_scope_indent += 1
                self.stack[-1].indent = self.column + spaces
            else:
                self.stack[-1].closing_scope_indent = self.stack[-1].indent - style.Get('CONTINUATION_INDENT_WIDTH')
        self.column += spaces

    def _AddTokenOnNewline(self, dry_run, must_split):
        if False:
            return 10
        "Adds a line break and necessary indentation.\n\n    Appends the next token to the state and updates information necessary for\n    indentation.\n\n    Arguments:\n      dry_run: (bool) Don't commit whitespace changes to the FormatToken if\n        True.\n      must_split: (bool) A newline was required before this token.\n\n    Returns:\n      The split penalty for splitting after the current state.\n    "
        current = self.next_token
        previous = current.previous_token
        self.column = self._GetNewlineColumn()
        if not dry_run:
            indent_level = self.line.depth
            spaces = self.column
            if spaces:
                spaces -= indent_level * style.Get('INDENT_WIDTH')
            current.AddWhitespacePrefix(newlines_before=1, spaces=spaces, indent_level=indent_level)
        if not current.is_comment:
            self.stack[-1].last_space = self.column
        self.lowest_level_on_line = self.paren_level
        if previous.OpensScope() or (previous.is_comment and previous.previous_token is not None and previous.previous_token.OpensScope()):
            dedent = (style.Get('CONTINUATION_INDENT_WIDTH'), 0)[style.Get('INDENT_CLOSING_BRACKETS')]
            self.stack[-1].closing_scope_indent = max(0, self.stack[-1].indent - dedent)
            self.stack[-1].split_before_closing_bracket = True
        penalty = current.split_penalty
        if must_split:
            return penalty
        if previous.is_pseudo and previous.value == '(':
            penalty += 50
        if current.value not in {'if', 'for'}:
            last = self.stack[-1]
            last.num_line_splits += 1
            penalty += style.Get('SPLIT_PENALTY_FOR_ADDED_LINE_SPLIT') * last.num_line_splits
        if current.OpensScope() and previous.OpensScope():
            pprev = previous.previous_token
            if not pprev or not pprev.is_name:
                penalty += 10
        return penalty + 10

    def MoveStateToNextToken(self):
        if False:
            i = 10
            return i + 15
        'Calculate format decision state information and move onto the next token.\n\n    Before moving onto the next token, we first calculate the format decision\n    state given the current token and its formatting decisions. Then the format\n    decision state is set up so that the next token can be added.\n\n    Returns:\n      The penalty for the number of characters over the column limit.\n    '
        current = self.next_token
        if not current.OpensScope() and (not current.ClosesScope()):
            self.lowest_level_on_line = min(self.lowest_level_on_line, self.paren_level)
        if current.OpensScope():
            last = self.stack[-1]
            new_indent = style.Get('CONTINUATION_INDENT_WIDTH') + last.last_space
            self.stack.append(_ParenState(new_indent, self.stack[-1].last_space))
            self.paren_level += 1
        if len(self.stack) > 1 and current.ClosesScope():
            if subtypes.DICTIONARY_KEY_PART in current.subtypes:
                self.stack[-2].last_space = self.stack[-2].indent
            else:
                self.stack[-2].last_space = self.stack[-1].last_space
            self.stack.pop()
            self.paren_level -= 1
        is_multiline_string = current.is_string and '\n' in current.value
        if is_multiline_string:
            self.column += len(current.value.split('\n')[0])
        elif not current.is_pseudo:
            self.column += len(current.value)
        self.next_token = self.next_token.next_token
        penalty = 0
        if not current.is_pylint_comment and (not current.is_pytype_comment) and (not current.is_copybara_comment) and (self.column > self.column_limit):
            excess_characters = self.column - self.column_limit
            penalty += style.Get('SPLIT_PENALTY_EXCESS_CHARACTER') * excess_characters
        if is_multiline_string:
            self.column = len(current.value.split('\n')[-1])
        return penalty

    def _CalculateComprehensionState(self, newline):
        if False:
            i = 10
            return i + 15
        'Makes required changes to comprehension state.\n\n    Args:\n      newline: Whether the current token is to be added on a newline.\n\n    Returns:\n      The penalty for the token-newline combination given the current\n      comprehension state.\n    '
        current = self.next_token
        previous = current.previous_token
        top_of_stack = self.comp_stack[-1] if self.comp_stack else None
        penalty = 0
        if top_of_stack is not None:
            if current == top_of_stack.closing_bracket:
                last = self.comp_stack.pop()
                if last.has_interior_split:
                    penalty += style.Get('SPLIT_PENALTY_COMPREHENSION')
                return penalty
            if newline:
                top_of_stack.has_interior_split = True
        if subtypes.COMP_EXPR in current.subtypes and subtypes.COMP_EXPR not in previous.subtypes:
            self.comp_stack.append(object_state.ComprehensionState(current))
            return penalty
        if current.value == 'for' and subtypes.COMP_FOR in current.subtypes:
            if top_of_stack.for_token is not None:
                if style.Get('SPLIT_COMPLEX_COMPREHENSION') and top_of_stack.has_split_at_for != newline and (top_of_stack.has_split_at_for or not top_of_stack.HasTrivialExpr()):
                    penalty += split_penalty.UNBREAKABLE
            else:
                top_of_stack.for_token = current
                top_of_stack.has_split_at_for = newline
                if style.Get('SPLIT_COMPLEX_COMPREHENSION') and newline and top_of_stack.HasTrivialExpr():
                    penalty += split_penalty.CONNECTED
        if subtypes.COMP_IF in current.subtypes and subtypes.COMP_IF not in previous.subtypes:
            if style.Get('SPLIT_COMPLEX_COMPREHENSION') and top_of_stack.has_split_at_for != newline and (top_of_stack.has_split_at_for or not top_of_stack.HasTrivialExpr()):
                penalty += split_penalty.UNBREAKABLE
        return penalty

    def _PushParameterListState(self, newline):
        if False:
            print('Hello World!')
        'Push a new parameter list state for a function definition.\n\n    Args:\n      newline: Whether the current token is to be added on a newline.\n    '
        current = self.next_token
        previous = current.previous_token
        if _IsFunctionDefinition(previous):
            first_param_column = previous.total_length + self.stack[-2].indent
            self.param_list_stack.append(object_state.ParameterListState(previous, newline, first_param_column))

    def _CalculateParameterListState(self, newline):
        if False:
            for i in range(10):
                print('nop')
        'Makes required changes to parameter list state.\n\n    Args:\n      newline: Whether the current token is to be added on a newline.\n\n    Returns:\n      The penalty for the token-newline combination given the current\n      parameter state.\n    '
        current = self.next_token
        previous = current.previous_token
        penalty = 0
        if _IsFunctionDefinition(previous):
            first_param_column = previous.total_length + self.stack[-2].indent
            if not newline:
                param_list = self.param_list_stack[-1]
                if param_list.parameters and param_list.has_typed_return:
                    last_param = param_list.parameters[-1].first_token
                    last_token = _LastTokenInLine(previous.matching_bracket)
                    total_length = last_token.total_length
                    total_length -= last_param.total_length - len(last_param.value)
                    if total_length + self.column > self.column_limit:
                        penalty += split_penalty.VERY_STRONGLY_CONNECTED
                return penalty
            if first_param_column <= self.column:
                penalty += split_penalty.VERY_STRONGLY_CONNECTED
            return penalty
        if not self.param_list_stack:
            return penalty
        param_list = self.param_list_stack[-1]
        if current == self.param_list_stack[-1].closing_bracket:
            self.param_list_stack.pop()
            if newline and param_list.has_typed_return:
                if param_list.split_before_closing_bracket:
                    penalty -= split_penalty.STRONGLY_CONNECTED
                elif param_list.LastParamFitsOnLine(self.column):
                    penalty += split_penalty.STRONGLY_CONNECTED
            if not newline and param_list.has_typed_return and param_list.has_split_before_first_param:
                penalty += split_penalty.STRONGLY_CONNECTED
            return penalty
        if not param_list.parameters:
            return penalty
        if newline:
            if self._FitsOnLine(param_list.parameters[0].first_token, _LastTokenInLine(param_list.closing_bracket)):
                penalty += split_penalty.STRONGLY_CONNECTED
        if not newline and style.Get('SPLIT_BEFORE_NAMED_ASSIGNS') and param_list.has_default_values and (current != param_list.parameters[0].first_token) and (current != param_list.closing_bracket) and (subtypes.PARAMETER_START in current.subtypes):
            penalty += split_penalty.STRONGLY_CONNECTED
        return penalty

    def _IndentWithContinuationAlignStyle(self, column):
        if False:
            while True:
                i = 10
        if column == 0:
            return column
        align_style = style.Get('CONTINUATION_ALIGN_STYLE')
        if align_style == 'FIXED':
            return self.line.depth * style.Get('INDENT_WIDTH') + style.Get('CONTINUATION_INDENT_WIDTH')
        if align_style == 'VALIGN-RIGHT':
            indent_width = style.Get('INDENT_WIDTH')
            return indent_width * int((column + indent_width - 1) / indent_width)
        return column

    def _GetNewlineColumn(self):
        if False:
            while True:
                i = 10
        'Return the new column on the newline.'
        current = self.next_token
        previous = current.previous_token
        top_of_stack = self.stack[-1]
        if isinstance(current.spaces_required_before, list):
            return 0
        elif current.spaces_required_before > 2 or self.line.disable:
            return current.spaces_required_before
        cont_aligned_indent = self._IndentWithContinuationAlignStyle(top_of_stack.indent)
        if current.OpensScope():
            return cont_aligned_indent if self.paren_level else self.first_indent
        if current.ClosesScope():
            if previous.OpensScope() or (previous.is_comment and previous.previous_token is not None and previous.previous_token.OpensScope()):
                return max(0, top_of_stack.indent - style.Get('CONTINUATION_INDENT_WIDTH'))
            return top_of_stack.closing_scope_indent
        if previous and previous.is_string and current.is_string and (subtypes.DICTIONARY_VALUE in current.subtypes):
            return previous.column
        if style.Get('INDENT_DICTIONARY_VALUE'):
            if previous and (previous.value == ':' or previous.is_pseudo):
                if subtypes.DICTIONARY_VALUE in current.subtypes:
                    return top_of_stack.indent
        if not self.param_list_stack and _IsCompoundStatement(self.line.first) and (not (style.Get('DEDENT_CLOSING_BRACKETS') or style.Get('INDENT_CLOSING_BRACKETS')) or style.Get('SPLIT_BEFORE_FIRST_ARGUMENT')):
            token_indent = len(self.line.first.whitespace_prefix.split('\n')[-1]) + style.Get('INDENT_WIDTH')
            if token_indent == top_of_stack.indent:
                return token_indent + style.Get('CONTINUATION_INDENT_WIDTH')
        if self.param_list_stack and (not self.param_list_stack[-1].SplitBeforeClosingBracket(top_of_stack.indent)) and (top_of_stack.indent == (self.line.depth + 1) * style.Get('INDENT_WIDTH')):
            if subtypes.PARAMETER_START in current.subtypes or (previous.is_comment and subtypes.PARAMETER_START in previous.subtypes):
                return top_of_stack.indent + style.Get('CONTINUATION_INDENT_WIDTH')
        return cont_aligned_indent

    def _FitsOnLine(self, start, end):
        if False:
            for i in range(10):
                print('nop')
        'Determines if line between start and end can fit on the current line.'
        length = end.total_length - start.total_length
        if not start.is_pseudo:
            length += len(start.value)
        return length + self.column <= self.column_limit

    def _EachDictEntryFitsOnOneLine(self, opening):
        if False:
            i = 10
            return i + 15
        'Determine if each dict elems can fit on one line.'

        def PreviousNonCommentToken(tok):
            if False:
                i = 10
                return i + 15
            tok = tok.previous_token
            while tok.is_comment:
                tok = tok.previous_token
            return tok

        def ImplicitStringConcatenation(tok):
            if False:
                for i in range(10):
                    print('nop')
            num_strings = 0
            if tok.is_pseudo:
                tok = tok.next_token
            while tok.is_string:
                num_strings += 1
                tok = tok.next_token
            return num_strings > 1

        def DictValueIsContainer(opening, closing):
            if False:
                print('Hello World!')
            'Return true if the dictionary value is a container.'
            if not opening or not closing:
                return False
            colon = opening.previous_token
            while colon:
                if not colon.is_pseudo:
                    break
                colon = colon.previous_token
            if not colon or colon.value != ':':
                return False
            key = colon.previous_token
            if not key:
                return False
            return subtypes.DICTIONARY_KEY_PART in key.subtypes
        closing = opening.matching_bracket
        entry_start = opening.next_token
        current = opening.next_token.next_token
        while current and current != closing:
            if subtypes.DICT_SET_GENERATOR in current.subtypes:
                break
            if subtypes.DICTIONARY_KEY in current.subtypes:
                prev = PreviousNonCommentToken(current)
                if prev.value == ',':
                    prev = PreviousNonCommentToken(prev.previous_token)
                if not DictValueIsContainer(prev.matching_bracket, prev):
                    length = prev.total_length - entry_start.total_length
                    length += len(entry_start.value)
                    if length + self.stack[-2].indent >= self.column_limit:
                        return False
                entry_start = current
            if current.OpensScope():
                if (current.value == '{' or ((current.is_pseudo and current.next_token.value == '{') and subtypes.DICTIONARY_VALUE in current.subtypes)) or ImplicitStringConcatenation(current):
                    if current.matching_bracket:
                        current = current.matching_bracket
                    while current:
                        if current == closing:
                            return True
                        if subtypes.DICTIONARY_KEY in current.subtypes:
                            entry_start = current
                            break
                        current = current.next_token
                else:
                    current = current.matching_bracket
            else:
                current = current.next_token
        current = PreviousNonCommentToken(current)
        length = current.total_length - entry_start.total_length
        length += len(entry_start.value)
        return length + self.stack[-2].indent <= self.column_limit

    def _ArgumentListHasDictionaryEntry(self, token):
        if False:
            while True:
                i = 10
        'Check if the function argument list has a dictionary as an arg.'
        if _IsArgumentToFunction(token):
            while token:
                if token.value == '{':
                    length = token.matching_bracket.total_length - token.total_length
                    return length + self.stack[-2].indent > self.column_limit
                if token.ClosesScope():
                    break
                if token.OpensScope():
                    token = token.matching_bracket
                token = token.next_token
        return False

    def _ContainerFitsOnStartLine(self, opening):
        if False:
            while True:
                i = 10
        'Check if the container can fit on its starting line.'
        return opening.matching_bracket.total_length - opening.total_length + self.stack[-1].indent <= self.column_limit
_COMPOUND_STMTS = frozenset({'for', 'while', 'if', 'elif', 'with', 'except', 'def', 'class'})

def _IsCompoundStatement(token):
    if False:
        return 10
    value = token.value
    if value == 'async':
        token = token.next_token
    if token.value in _COMPOUND_STMTS:
        return True
    parent_name = NodeName(token.node.parent)
    return value == 'match' and parent_name == 'match_stmt' or (value == 'case' and parent_name == 'case_stmt')

def _IsFunctionDef(token):
    if False:
        while True:
            i = 10
    if token.value == 'async':
        token = token.next_token
    return token.value == 'def'

def _IsFunctionCallWithArguments(token):
    if False:
        return 10
    while token:
        if token.value == '(':
            token = token.next_token
            return token and token.value != ')'
        elif token.name not in {'NAME', 'DOT', 'EQUAL'}:
            break
        token = token.next_token
    return False

def _IsArgumentToFunction(token):
    if False:
        i = 10
        return i + 15
    bracket = logical_line.IsSurroundedByBrackets(token)
    if not bracket or bracket.value != '(':
        return False
    previous = bracket.previous_token
    return previous and previous.is_name

def _GetOpeningBracket(current):
    if False:
        i = 10
        return i + 15
    'Get the opening bracket containing the current token.'
    if current.matching_bracket and (not current.is_pseudo):
        return current if current.OpensScope() else current.matching_bracket
    while current:
        if current.ClosesScope():
            current = current.matching_bracket
        elif current.is_pseudo:
            current = current.previous_token
        elif current.OpensScope():
            return current
        current = current.previous_token
    return None

def _LastTokenInLine(current):
    if False:
        print('Hello World!')
    while not current.is_comment and current.next_token:
        current = current.next_token
    return current

def _IsFunctionDefinition(current):
    if False:
        for i in range(10):
            print('nop')
    prev = current.previous_token
    return current.value == '(' and prev and (subtypes.FUNC_DEF in prev.subtypes)

def _IsLastScopeInLine(current):
    if False:
        return 10
    current = current.matching_bracket
    while current:
        current = current.next_token
        if current and current.OpensScope():
            return False
    return True

def _IsSingleElementTuple(token):
    if False:
        while True:
            i = 10
    "Check if it's a single-element tuple."
    close = token.matching_bracket
    token = token.next_token
    num_commas = 0
    while token != close:
        if token.value == ',':
            num_commas += 1
        token = token.matching_bracket if token.OpensScope() else token.next_token
    return num_commas == 1

def _ScopeHasNoCommas(token):
    if False:
        for i in range(10):
            print('nop')
    'Check if the scope has no commas.'
    close = token.matching_bracket
    token = token.next_token
    while token != close:
        if token.value == ',':
            return False
        token = token.matching_bracket if token.OpensScope() else token.next_token
    return True

class _ParenState(object):
    """Maintains the state of the bracket enclosures.

  A stack of _ParenState objects are kept so that we know how to indent relative
  to the brackets.

  Attributes:
    indent: The column position to which a specified parenthesis level needs to
      be indented.
    last_space: The column position of the last space on each level.
    closing_scope_indent: The column position of the closing indentation.
    split_before_closing_bracket: Whether a newline needs to be inserted before
      the closing bracket. We only want to insert a newline before the closing
      bracket if there also was a newline after the beginning left bracket.
    num_line_splits: Number of line splits this _ParenState contains already.
      Each subsequent line split gets an increasing penalty.
  """

    def __init__(self, indent, last_space):
        if False:
            while True:
                i = 10
        self.indent = indent
        self.last_space = last_space
        self.closing_scope_indent = 0
        self.split_before_closing_bracket = False
        self.num_line_splits = 0

    def Clone(self):
        if False:
            for i in range(10):
                print('nop')
        state = _ParenState(self.indent, self.last_space)
        state.closing_scope_indent = self.closing_scope_indent
        state.split_before_closing_bracket = self.split_before_closing_bracket
        state.num_line_splits = self.num_line_splits
        return state

    def __repr__(self):
        if False:
            return 10
        return '[indent::%d, last_space::%d, closing_scope_indent::%d]' % (self.indent, self.last_space, self.closing_scope_indent)

    def __eq__(self, other):
        if False:
            return 10
        return hash(self) == hash(other)

    def __ne__(self, other):
        if False:
            i = 10
            return i + 15
        return not self == other

    def __hash__(self, *args, **kwargs):
        if False:
            return 10
        return hash((self.indent, self.last_space, self.closing_scope_indent, self.split_before_closing_bracket, self.num_line_splits))