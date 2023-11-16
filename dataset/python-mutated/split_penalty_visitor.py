import ast
from yapf.pyparser import pyparser_utils as pyutils
from yapf.yapflib import split_penalty
from yapf.yapflib import style
from yapf.yapflib import subtypes

class SplitPenalty(ast.NodeVisitor):
    """Compute split penalties between tokens."""

    def __init__(self, logical_lines):
        if False:
            i = 10
            return i + 15
        super(SplitPenalty, self).__init__()
        self.logical_lines = logical_lines
        for logical_line in logical_lines:
            for token in logical_line.tokens:
                if token.value in frozenset({',', ':'}):
                    token.split_penalty = split_penalty.UNBREAKABLE

    def _GetTokens(self, node):
        if False:
            while True:
                i = 10
        return pyutils.GetLogicalLine(self.logical_lines, node)

    def visit_FunctionDef(self, node):
        if False:
            for i in range(10):
                print('nop')
        tokens = self._GetTokens(node)
        for decorator in node.decorator_list:
            decorator_range = self._GetTokens(decorator)
            decorator_range[0].split_penalty = split_penalty.UNBREAKABLE
        for token in tokens[1:]:
            if token.value == '(':
                break
            _SetPenalty(token, split_penalty.UNBREAKABLE)
        if node.returns:
            start_index = pyutils.GetTokenIndex(tokens, pyutils.TokenStart(node.returns))
            _IncreasePenalty(tokens[start_index - 1:start_index + 1], split_penalty.VERY_STRONGLY_CONNECTED)
            end_index = pyutils.GetTokenIndex(tokens, pyutils.TokenEnd(node.returns))
            _IncreasePenalty(tokens[start_index + 1:end_index], split_penalty.STRONGLY_CONNECTED)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        if False:
            return 10
        return self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        if False:
            i = 10
            return i + 15
        for base in node.bases:
            tokens = self._GetTokens(base)
            _IncreasePenalty(tokens[1:], split_penalty.EXPR)
        for decorator in node.decorator_list:
            tokens = self._GetTokens(decorator)
            tokens[0].split_penalty = split_penalty.UNBREAKABLE
        return self.generic_visit(node)

    def visit_Return(self, node):
        if False:
            i = 10
            return i + 15
        tokens = self._GetTokens(node)
        _IncreasePenalty(tokens[1:], split_penalty.EXPR)
        return self.generic_visit(node)

    def visit_Delete(self, node):
        if False:
            while True:
                i = 10
        for target in node.targets:
            tokens = self._GetTokens(target)
            _IncreasePenalty(tokens[1:], split_penalty.EXPR)
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if False:
            print('Hello World!')
        tokens = self._GetTokens(node)
        _IncreasePenalty(tokens[1:], split_penalty.EXPR)
        return self.generic_visit(node)

    def visit_AugAssign(self, node):
        if False:
            i = 10
            return i + 15
        return self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if False:
            return 10
        return self.generic_visit(node)

    def visit_For(self, node):
        if False:
            i = 10
            return i + 15
        return self.generic_visit(node)

    def visit_AsyncFor(self, node):
        if False:
            i = 10
            return i + 15
        return self.generic_visit(node)

    def visit_While(self, node):
        if False:
            i = 10
            return i + 15
        return self.generic_visit(node)

    def visit_If(self, node):
        if False:
            while True:
                i = 10
        return self.generic_visit(node)

    def visit_With(self, node):
        if False:
            i = 10
            return i + 15
        return self.generic_visit(node)

    def visit_AsyncWith(self, node):
        if False:
            while True:
                i = 10
        return self.generic_visit(node)

    def visit_Match(self, node):
        if False:
            while True:
                i = 10
        return self.generic_visit(node)

    def visit_Raise(self, node):
        if False:
            while True:
                i = 10
        return self.generic_visit(node)

    def visit_Try(self, node):
        if False:
            i = 10
            return i + 15
        return self.generic_visit(node)

    def visit_Assert(self, node):
        if False:
            for i in range(10):
                print('nop')
        return self.generic_visit(node)

    def visit_Import(self, node):
        if False:
            while True:
                i = 10
        return self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if False:
            print('Hello World!')
        return self.generic_visit(node)

    def visit_Global(self, node):
        if False:
            while True:
                i = 10
        return self.generic_visit(node)

    def visit_Nonlocal(self, node):
        if False:
            print('Hello World!')
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if False:
            i = 10
            return i + 15
        return self.generic_visit(node)

    def visit_Pass(self, node):
        if False:
            i = 10
            return i + 15
        return self.generic_visit(node)

    def visit_Break(self, node):
        if False:
            while True:
                i = 10
        return self.generic_visit(node)

    def visit_Continue(self, node):
        if False:
            return 10
        return self.generic_visit(node)

    def visit_BoolOp(self, node):
        if False:
            return 10
        tokens = self._GetTokens(node)
        _IncreasePenalty(tokens[1:], split_penalty.EXPR)
        split_before_operator = style.Get('SPLIT_BEFORE_LOGICAL_OPERATOR')
        operator_indices = [pyutils.GetNextTokenIndex(tokens, pyutils.TokenEnd(value)) for value in node.values[:-1]]
        for operator_index in operator_indices:
            if not split_before_operator:
                operator_index += 1
            _DecreasePenalty(tokens[operator_index], split_penalty.EXPR * 2)
        return self.generic_visit(node)

    def visit_NamedExpr(self, node):
        if False:
            i = 10
            return i + 15
        tokens = self._GetTokens(node)
        _IncreasePenalty(tokens[1:], split_penalty.EXPR)
        return self.generic_visit(node)

    def visit_BinOp(self, node):
        if False:
            while True:
                i = 10
        tokens = self._GetTokens(node)
        _IncreasePenalty(tokens[1:], split_penalty.EXPR)
        operator_index = pyutils.GetNextTokenIndex(tokens, pyutils.TokenEnd(node.left))
        if not style.Get('SPLIT_BEFORE_ARITHMETIC_OPERATOR'):
            operator_index += 1
        _DecreasePenalty(tokens[operator_index], split_penalty.EXPR * 2)
        return self.generic_visit(node)

    def visit_UnaryOp(self, node):
        if False:
            while True:
                i = 10
        tokens = self._GetTokens(node)
        _IncreasePenalty(tokens[1:], split_penalty.EXPR)
        _IncreasePenalty(tokens[1], style.Get('SPLIT_PENALTY_AFTER_UNARY_OPERATOR'))
        return self.generic_visit(node)

    def visit_Lambda(self, node):
        if False:
            print('Hello World!')
        tokens = self._GetTokens(node)
        _IncreasePenalty(tokens[1:], split_penalty.LAMBDA)
        if style.Get('ALLOW_MULTILINE_LAMBDAS'):
            _SetPenalty(self._GetTokens(node.body), split_penalty.MULTIPLINE_LAMBDA)
        return self.generic_visit(node)

    def visit_IfExp(self, node):
        if False:
            return 10
        tokens = self._GetTokens(node)
        _IncreasePenalty(tokens[1:], split_penalty.EXPR)
        return self.generic_visit(node)

    def visit_Dict(self, node):
        if False:
            for i in range(10):
                print('nop')
        tokens = self._GetTokens(node)
        for key in node.keys:
            subrange = pyutils.GetTokensInSubRange(tokens, key)
            _IncreasePenalty(subrange[1:], split_penalty.DICT_KEY_EXPR)
        for value in node.values:
            subrange = pyutils.GetTokensInSubRange(tokens, value)
            _IncreasePenalty(subrange[1:], split_penalty.DICT_VALUE_EXPR)
        return self.generic_visit(node)

    def visit_Set(self, node):
        if False:
            while True:
                i = 10
        tokens = self._GetTokens(node)
        for element in node.elts:
            subrange = pyutils.GetTokensInSubRange(tokens, element)
            _IncreasePenalty(subrange[1:], split_penalty.EXPR)
        return self.generic_visit(node)

    def visit_ListComp(self, node):
        if False:
            for i in range(10):
                print('nop')
        tokens = self._GetTokens(node)
        element = pyutils.GetTokensInSubRange(tokens, node.elt)
        _IncreasePenalty(element[1:], split_penalty.EXPR)
        for comp in node.generators:
            subrange = pyutils.GetTokensInSubRange(tokens, comp.iter)
            _IncreasePenalty(subrange[1:], split_penalty.EXPR)
            for if_expr in comp.ifs:
                subrange = pyutils.GetTokensInSubRange(tokens, if_expr)
                _IncreasePenalty(subrange[1:], split_penalty.EXPR)
        return self.generic_visit(node)

    def visit_SetComp(self, node):
        if False:
            i = 10
            return i + 15
        tokens = self._GetTokens(node)
        element = pyutils.GetTokensInSubRange(tokens, node.elt)
        _IncreasePenalty(element[1:], split_penalty.EXPR)
        for comp in node.generators:
            subrange = pyutils.GetTokensInSubRange(tokens, comp.iter)
            _IncreasePenalty(subrange[1:], split_penalty.EXPR)
            for if_expr in comp.ifs:
                subrange = pyutils.GetTokensInSubRange(tokens, if_expr)
                _IncreasePenalty(subrange[1:], split_penalty.EXPR)
        return self.generic_visit(node)

    def visit_DictComp(self, node):
        if False:
            i = 10
            return i + 15
        tokens = self._GetTokens(node)
        key = pyutils.GetTokensInSubRange(tokens, node.key)
        _IncreasePenalty(key[1:], split_penalty.EXPR)
        value = pyutils.GetTokensInSubRange(tokens, node.value)
        _IncreasePenalty(value[1:], split_penalty.EXPR)
        for comp in node.generators:
            subrange = pyutils.GetTokensInSubRange(tokens, comp.iter)
            _IncreasePenalty(subrange[1:], split_penalty.EXPR)
            for if_expr in comp.ifs:
                subrange = pyutils.GetTokensInSubRange(tokens, if_expr)
                _IncreasePenalty(subrange[1:], split_penalty.EXPR)
        return self.generic_visit(node)

    def visit_GeneratorExp(self, node):
        if False:
            for i in range(10):
                print('nop')
        tokens = self._GetTokens(node)
        element = pyutils.GetTokensInSubRange(tokens, node.elt)
        _IncreasePenalty(element[1:], split_penalty.EXPR)
        for comp in node.generators:
            subrange = pyutils.GetTokensInSubRange(tokens, comp.iter)
            _IncreasePenalty(subrange[1:], split_penalty.EXPR)
            for if_expr in comp.ifs:
                subrange = pyutils.GetTokensInSubRange(tokens, if_expr)
                _IncreasePenalty(subrange[1:], split_penalty.EXPR)
        return self.generic_visit(node)

    def visit_Await(self, node):
        if False:
            for i in range(10):
                print('nop')
        tokens = self._GetTokens(node)
        _IncreasePenalty(tokens[1:], split_penalty.EXPR)
        return self.generic_visit(node)

    def visit_Yield(self, node):
        if False:
            while True:
                i = 10
        tokens = self._GetTokens(node)
        _IncreasePenalty(tokens[1:], split_penalty.EXPR)
        return self.generic_visit(node)

    def visit_YieldFrom(self, node):
        if False:
            i = 10
            return i + 15
        tokens = self._GetTokens(node)
        _IncreasePenalty(tokens[1:], split_penalty.EXPR)
        tokens[2].split_penalty = split_penalty.UNBREAKABLE
        return self.generic_visit(node)

    def visit_Compare(self, node):
        if False:
            i = 10
            return i + 15
        tokens = self._GetTokens(node)
        _IncreasePenalty(tokens[1:], split_penalty.EXPR)
        operator_indices = [pyutils.GetNextTokenIndex(tokens, pyutils.TokenEnd(node.left))] + [pyutils.GetNextTokenIndex(tokens, pyutils.TokenEnd(comparator)) for comparator in node.comparators[:-1]]
        split_before = style.Get('SPLIT_BEFORE_ARITHMETIC_OPERATOR')
        for operator_index in operator_indices:
            if not split_before:
                operator_index += 1
            _DecreasePenalty(tokens[operator_index], split_penalty.EXPR * 2)
        return self.generic_visit(node)

    def visit_Call(self, node):
        if False:
            while True:
                i = 10
        tokens = self._GetTokens(node)
        paren_index = pyutils.GetNextTokenIndex(tokens, pyutils.TokenEnd(node.func))
        _IncreasePenalty(tokens[paren_index], split_penalty.UNBREAKABLE)
        for arg in node.args:
            subrange = pyutils.GetTokensInSubRange(tokens, arg)
            _IncreasePenalty(subrange[1:], split_penalty.EXPR)
        return self.generic_visit(node)

    def visit_FormattedValue(self, node):
        if False:
            for i in range(10):
                print('nop')
        return node

    def visit_JoinedStr(self, node):
        if False:
            while True:
                i = 10
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if False:
            print('Hello World!')
        return self.generic_visit(node)

    def visit_Attribute(self, node):
        if False:
            for i in range(10):
                print('nop')
        tokens = self._GetTokens(node)
        split_before = style.Get('SPLIT_BEFORE_DOT')
        dot_indices = pyutils.GetNextTokenIndex(tokens, pyutils.TokenEnd(node.value))
        if not split_before:
            dot_indices += 1
        _IncreasePenalty(tokens[dot_indices], split_penalty.VERY_STRONGLY_CONNECTED)
        return self.generic_visit(node)

    def visit_Subscript(self, node):
        if False:
            return 10
        tokens = self._GetTokens(node)
        bracket_index = pyutils.GetNextTokenIndex(tokens, pyutils.TokenEnd(node.value))
        _IncreasePenalty(tokens[bracket_index], split_penalty.UNBREAKABLE)
        return self.generic_visit(node)

    def visit_Starred(self, node):
        if False:
            while True:
                i = 10
        return self.generic_visit(node)

    def visit_Name(self, node):
        if False:
            for i in range(10):
                print('nop')
        tokens = self._GetTokens(node)
        _IncreasePenalty(tokens[1:], split_penalty.UNBREAKABLE)
        return self.generic_visit(node)

    def visit_List(self, node):
        if False:
            i = 10
            return i + 15
        tokens = self._GetTokens(node)
        for element in node.elts:
            subrange = pyutils.GetTokensInSubRange(tokens, element)
            _IncreasePenalty(subrange[1:], split_penalty.EXPR)
            _DecreasePenalty(subrange[0], split_penalty.EXPR // 2)
        return self.generic_visit(node)

    def visit_Tuple(self, node):
        if False:
            for i in range(10):
                print('nop')
        tokens = self._GetTokens(node)
        for element in node.elts:
            subrange = pyutils.GetTokensInSubRange(tokens, element)
            _IncreasePenalty(subrange[1:], split_penalty.EXPR)
            _DecreasePenalty(subrange[0], split_penalty.EXPR // 2)
        return self.generic_visit(node)

    def visit_Slice(self, node):
        if False:
            for i in range(10):
                print('nop')
        tokens = self._GetTokens(node)
        if hasattr(node, 'lower') and node.lower:
            subrange = pyutils.GetTokensInSubRange(tokens, node.lower)
            _IncreasePenalty(subrange, split_penalty.EXPR)
            _DecreasePenalty(subrange[0], split_penalty.EXPR // 2)
        if hasattr(node, 'upper') and node.upper:
            colon_index = pyutils.GetPrevTokenIndex(tokens, pyutils.TokenStart(node.upper))
            _IncreasePenalty(tokens[colon_index], split_penalty.UNBREAKABLE)
            subrange = pyutils.GetTokensInSubRange(tokens, node.upper)
            _IncreasePenalty(subrange, split_penalty.EXPR)
            _DecreasePenalty(subrange[0], split_penalty.EXPR // 2)
        if hasattr(node, 'step') and node.step:
            colon_index = pyutils.GetPrevTokenIndex(tokens, pyutils.TokenStart(node.step))
            _IncreasePenalty(tokens[colon_index], split_penalty.UNBREAKABLE)
            subrange = pyutils.GetTokensInSubRange(tokens, node.step)
            _IncreasePenalty(subrange, split_penalty.EXPR)
            _DecreasePenalty(subrange[0], split_penalty.EXPR // 2)
        return self.generic_visit(node)

    def visit_Load(self, node):
        if False:
            i = 10
            return i + 15
        return self.generic_visit(node)

    def visit_Store(self, node):
        if False:
            return 10
        return self.generic_visit(node)

    def visit_Del(self, node):
        if False:
            i = 10
            return i + 15
        return self.generic_visit(node)

    def visit_And(self, node):
        if False:
            while True:
                i = 10
        return self.generic_visit(node)

    def visit_Or(self, node):
        if False:
            while True:
                i = 10
        return self.generic_visit(node)

    def visit_Add(self, node):
        if False:
            print('Hello World!')
        return self.generic_visit(node)

    def visit_Sub(self, node):
        if False:
            while True:
                i = 10
        return self.generic_visit(node)

    def visit_Mult(self, node):
        if False:
            while True:
                i = 10
        return self.generic_visit(node)

    def visit_MatMult(self, node):
        if False:
            return 10
        return self.generic_visit(node)

    def visit_Div(self, node):
        if False:
            print('Hello World!')
        return self.generic_visit(node)

    def visit_Mod(self, node):
        if False:
            print('Hello World!')
        return self.generic_visit(node)

    def visit_Pow(self, node):
        if False:
            for i in range(10):
                print('nop')
        return self.generic_visit(node)

    def visit_LShift(self, node):
        if False:
            return 10
        return self.generic_visit(node)

    def visit_RShift(self, node):
        if False:
            while True:
                i = 10
        return self.generic_visit(node)

    def visit_BitOr(self, node):
        if False:
            return 10
        return self.generic_visit(node)

    def visit_BitXor(self, node):
        if False:
            return 10
        return self.generic_visit(node)

    def visit_BitAnd(self, node):
        if False:
            return 10
        return self.generic_visit(node)

    def visit_FloorDiv(self, node):
        if False:
            while True:
                i = 10
        return self.generic_visit(node)

    def visit_Invert(self, node):
        if False:
            for i in range(10):
                print('nop')
        return self.generic_visit(node)

    def visit_Not(self, node):
        if False:
            for i in range(10):
                print('nop')
        return self.generic_visit(node)

    def visit_UAdd(self, node):
        if False:
            while True:
                i = 10
        return self.generic_visit(node)

    def visit_USub(self, node):
        if False:
            i = 10
            return i + 15
        return self.generic_visit(node)

    def visit_Eq(self, node):
        if False:
            while True:
                i = 10
        return self.generic_visit(node)

    def visit_NotEq(self, node):
        if False:
            for i in range(10):
                print('nop')
        return self.generic_visit(node)

    def visit_Lt(self, node):
        if False:
            print('Hello World!')
        return self.generic_visit(node)

    def visit_LtE(self, node):
        if False:
            for i in range(10):
                print('nop')
        return self.generic_visit(node)

    def visit_Gt(self, node):
        if False:
            for i in range(10):
                print('nop')
        return self.generic_visit(node)

    def visit_GtE(self, node):
        if False:
            print('Hello World!')
        return self.generic_visit(node)

    def visit_Is(self, node):
        if False:
            i = 10
            return i + 15
        return self.generic_visit(node)

    def visit_IsNot(self, node):
        if False:
            i = 10
            return i + 15
        return self.generic_visit(node)

    def visit_In(self, node):
        if False:
            print('Hello World!')
        return self.generic_visit(node)

    def visit_NotIn(self, node):
        if False:
            while True:
                i = 10
        return self.generic_visit(node)

    def visit_ExceptionHandler(self, node):
        if False:
            print('Hello World!')
        return self.generic_visit(node)

    def visit_MatchValue(self, node):
        if False:
            for i in range(10):
                print('nop')
        return self.generic_visit(node)

    def visit_MatchSingleton(self, node):
        if False:
            for i in range(10):
                print('nop')
        return self.generic_visit(node)

    def visit_MatchSequence(self, node):
        if False:
            print('Hello World!')
        return self.generic_visit(node)

    def visit_MatchMapping(self, node):
        if False:
            return 10
        return self.generic_visit(node)

    def visit_MatchClass(self, node):
        if False:
            i = 10
            return i + 15
        return self.generic_visit(node)

    def visit_MatchStar(self, node):
        if False:
            print('Hello World!')
        return self.generic_visit(node)

    def visit_MatchAs(self, node):
        if False:
            i = 10
            return i + 15
        return self.generic_visit(node)

    def visit_MatchOr(self, node):
        if False:
            for i in range(10):
                print('nop')
        return self.generic_visit(node)

    def visit_TypeIgnore(self, node):
        if False:
            for i in range(10):
                print('nop')
        return self.generic_visit(node)

    def visit_comprehension(self, node):
        if False:
            print('Hello World!')
        return self.generic_visit(node)

    def visit_arguments(self, node):
        if False:
            i = 10
            return i + 15
        return self.generic_visit(node)

    def visit_arg(self, node):
        if False:
            for i in range(10):
                print('nop')
        tokens = self._GetTokens(node)
        if hasattr(node, 'annotation') and node.annotation:
            annotation = node.annotation
            subrange = pyutils.GetTokensInSubRange(tokens, annotation)
            _IncreasePenalty(subrange, split_penalty.ANNOTATION)
        return self.generic_visit(node)

    def visit_keyword(self, node):
        if False:
            i = 10
            return i + 15
        return self.generic_visit(node)

    def visit_alias(self, node):
        if False:
            i = 10
            return i + 15
        return self.generic_visit(node)

    def visit_withitem(self, node):
        if False:
            i = 10
            return i + 15
        return self.generic_visit(node)

    def visit_match_case(self, node):
        if False:
            print('Hello World!')
        return self.generic_visit(node)

def _IncreasePenalty(tokens, amt):
    if False:
        print('Hello World!')
    if not isinstance(tokens, list):
        tokens = [tokens]
    for token in tokens:
        token.split_penalty += amt

def _DecreasePenalty(tokens, amt):
    if False:
        i = 10
        return i + 15
    if not isinstance(tokens, list):
        tokens = [tokens]
    for token in tokens:
        token.split_penalty -= amt

def _SetPenalty(tokens, amt):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(tokens, list):
        tokens = [tokens]
    for token in tokens:
        token.split_penalty = amt