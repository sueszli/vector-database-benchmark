"""Handles changes since PY310

handle
- import-alias requiring lineno
- match statement
"""
import ast
from xonsh.parsers.v39 import Parser as ThreeNineParser
from xonsh.ply.ply import yacc

class Parser(ThreeNineParser):

    def p_import_from_post_times(self, p):
        if False:
            return 10
        'import_from_post : TIMES'
        p[0] = [ast.alias(name=p[1], asname=None, **self.get_line_cols(p, 1))]

    def p_import_as_name(self, p):
        if False:
            while True:
                i = 10
        'import_as_name : name_str as_name_opt'
        self.p_dotted_as_name(p)

    def p_dotted_as_name(self, p: yacc.YaccProduction):
        if False:
            print('Hello World!')
        'dotted_as_name : dotted_name as_name_opt'
        alias_idx = 2
        p[0] = ast.alias(name=p[1], asname=p[alias_idx], **self.get_line_cols(p, alias_idx))

    @staticmethod
    def get_line_cols(p: yacc.YaccProduction, idx: int):
        if False:
            print('Hello World!')
        (line_no, end_line_no) = p.linespan(idx)
        (col_offset, end_col_offset) = p.lexspan(idx)
        return dict(lineno=line_no, end_lineno=end_line_no, col_offset=col_offset, end_col_offset=end_col_offset)

    def _set_error_at_production_index(self, msg, p, i):
        if False:
            return 10
        error_loc = self.get_line_cols(p, i)
        err_lineno = error_loc['lineno']
        err_column = error_loc['col_offset'] + 1
        self._set_error(msg, self.currloc(lineno=err_lineno, column=err_column))

    def p_compound_stmt_match(self, p):
        if False:
            return 10
        '\n        compound_stmt : match_stmt\n        '
        p[0] = p[1]

    def p_match_stmt(self, p):
        if False:
            print('Hello World!')
        '\n        match_stmt : match_tok subject_expr COLON NEWLINE INDENT case_block_list_nonempty DEDENT\n        '
        (_, _, subject_expr, _, _, _, case_block_list_nonempty, _) = p
        p[0] = [ast.Match(**self.get_line_cols(p, 1), subject=subject_expr, cases=case_block_list_nonempty)]

    def p_case_block(self, p):
        if False:
            print('Hello World!')
        '\n        case_block : case_tok patterns COLON suite\n                   | case_tok patterns IF test COLON suite\n        '
        loc = self.get_line_cols(p, 1)
        match list(p):
            case [_, _, pattern, _, suite]:
                p[0] = ast.match_case(pattern=pattern, body=suite, **loc)
            case [_, _, pattern, _, guard, _, suite]:
                p[0] = ast.match_case(pattern=pattern, body=suite, guard=guard, **loc)
            case _:
                raise AssertionError()

    def p_case_block_list_nonempty(self, p):
        if False:
            return 10
        '\n        case_block_list_nonempty : case_block\n                                 | case_block case_block_list_nonempty\n        '
        match list(p):
            case [_, case_block]:
                p[0] = [case_block]
            case [_, case_block, case_block_list_nonempty]:
                p[0] = [case_block] + case_block_list_nonempty
            case _:
                raise AssertionError()

    def p_subject_expr_single_value(self, p):
        if False:
            i = 10
            return i + 15
        '\n        subject_expr : test_or_star_expr comma_opt\n        '
        match list(p):
            case [_, test_or_star_expr, None]:
                p[0] = test_or_star_expr
            case [_, test_or_star_expr, ',']:
                p[0] = ast.Tuple(elts=[test_or_star_expr], ctx=ast.Load(), **self.get_line_cols(p, 1))
            case _:
                raise AssertionError()

    def p_subject_expr_multiple_values(self, p):
        if False:
            return 10
        '\n        subject_expr : test_or_star_expr comma_test_or_star_expr_list comma_opt\n        '
        match list(p):
            case [_, test_or_star_expr, comma_test_or_star_expr_list, ',' | None]:
                p[0] = ast.Tuple(elts=[test_or_star_expr] + comma_test_or_star_expr_list, ctx=ast.Load(), **self.get_line_cols(p, 1))
            case _:
                raise AssertionError()

    def p_closed_pattern(self, p):
        if False:
            for i in range(10):
                print('nop')
        '\n        closed_pattern : literal_pattern\n                       | capture_and_wildcard_pattern\n                       | group_pattern\n                       | sequence_pattern\n                       | value_pattern\n                       | class_pattern\n                       | mapping_pattern\n        '
        p[0] = p[1]

    def p_patterns(self, p):
        if False:
            i = 10
            return i + 15
        '\n        patterns : pattern\n                 | open_sequence_pattern\n        '
        p[0] = p[1]

    def p_pattern(self, p):
        if False:
            for i in range(10):
                print('nop')
        '\n        pattern : or_pattern\n                | as_pattern\n        '
        p[0] = p[1]

    def p_or_pattern(self, p):
        if False:
            while True:
                i = 10
        '\n        or_pattern : or_pattern_list\n        '
        (_, or_pattern_list) = p
        match or_pattern_list:
            case [single_value]:
                p[0] = single_value
            case multiple_values:
                p[0] = ast.MatchOr(patterns=multiple_values, **self.get_line_cols(p, 1))

    def p_or_pattern_list(self, p):
        if False:
            print('Hello World!')
        '\n        or_pattern_list : closed_pattern\n                        | closed_pattern PIPE or_pattern_list\n        '
        match list(p):
            case [_, closed_pattern]:
                p[0] = [closed_pattern]
            case [_, closed_pattern, '|', or_pattern_list]:
                p[0] = [closed_pattern] + or_pattern_list

    def p_group_pattern(self, p):
        if False:
            i = 10
            return i + 15
        '\n        group_pattern : LPAREN pattern RPAREN\n        '
        (_, _, pattern, _) = p
        p[0] = pattern

    def p_literal_pattern(self, p):
        if False:
            i = 10
            return i + 15
        '\n        literal_pattern : literal_expr\n        '
        match p[1]:
            case None | True | False:
                p[0] = ast.MatchSingleton(value=p[1], **self.get_line_cols(p, 1))
            case _:
                p[0] = ast.MatchValue(value=p[1], **self.get_line_cols(p, 1))

    def p_literal_expr_number_or_string_literal_list(self, p):
        if False:
            while True:
                i = 10
        '\n        literal_expr : complex_number\n                     | string_literal_list\n        '
        p[0] = p[1]
        match p[1]:
            case ast.JoinedStr():
                raise AssertionError('patterns may not match formatted string literals')

    def p_literal_expr_none_or_true_or_false(self, p):
        if False:
            for i in range(10):
                print('nop')
        '\n        literal_expr : none_tok\n                     | true_tok\n                     | false_tok\n        '
        match p[1].value:
            case 'None':
                value = None
            case 'True':
                value = True
            case 'False':
                value = False
            case _:
                raise AssertionError()
        p[0] = value

    def p_complex_number(self, p):
        if False:
            for i in range(10):
                print('nop')
        '\n        complex_number : number\n                       | MINUS number\n                       | number PLUS number\n                       | number MINUS number\n                       | MINUS number PLUS number\n                       | MINUS number MINUS number\n        '
        ops = {'+': ast.Add(), '-': ast.Sub()}
        build_complex = False
        loc = self.get_line_cols(p, 1)
        match list(p):
            case [_, x]:
                p[0] = x
            case [_, '-', x]:
                p[0] = ast.UnaryOp(op=ast.USub(), operand=x, **loc)
            case [_, left, '+' | '-' as op_char, right]:
                build_complex = True
                negate_left_side = False
            case [_, '-', left, '+' | '-' as op_char, right]:
                build_complex = True
                negate_left_side = True
            case _:
                raise AssertionError()
        if build_complex:
            assert isinstance(right.value, complex), 'right part of complex literal must be imaginary'
            if negate_left_side:
                left = ast.UnaryOp(op=ast.USub(), operand=left, **loc)
            p[0] = ast.BinOp(left=left, op=ops[op_char], right=right, **loc)

    def p_as_pattern(self, p):
        if False:
            return 10
        '\n        as_pattern : or_pattern AS capture_target_name\n        '
        (_, or_pattern, _, name) = p
        p[0] = ast.MatchAs(pattern=or_pattern, name=name, **self.get_line_cols(p, 1))

    def p_capture_target_name(self, p):
        if False:
            while True:
                i = 10
        '\n        capture_target_name : name_str\n        '
        name = p[1]
        if name == '_':
            self._set_error_at_production_index("can't capture name '_' in patterns", p, 1)
        p[0] = name

    def p_capture_and_wildcard_pattern(self, p):
        if False:
            i = 10
            return i + 15
        '\n        capture_and_wildcard_pattern : name_str\n        '
        (_, name) = p
        target = name if name != '_' else None
        p[0] = ast.MatchAs(name=target, **self.get_line_cols(p, 1))

    def p_sequence_pattern_square_brackets(self, p):
        if False:
            i = 10
            return i + 15
        '\n        sequence_pattern : LBRACKET maybe_sequence_pattern RBRACKET\n                         | LBRACKET RBRACKET\n                         | LPAREN open_sequence_pattern RPAREN\n                         | LPAREN RPAREN\n        '
        match list(p):
            case [_, _, ast.MatchSequence() as seq, _]:
                p[0] = seq
            case [_, _, single_item, _]:
                p[0] = ast.MatchSequence(patterns=[single_item], **self.get_line_cols(p, 1))
            case [_, _, _]:
                p[0] = ast.MatchSequence(patterns=[], **self.get_line_cols(p, 1))
            case _:
                raise AssertionError()

    def p_maybe_sequence_pattern(self, p):
        if False:
            for i in range(10):
                print('nop')
        '\n        maybe_sequence_pattern : maybe_star_pattern comma_opt\n                               | maybe_star_pattern COMMA maybe_sequence_pattern\n        '
        match list(p):
            case [_, maybe_star_pattern, ',']:
                p[0] = ast.MatchSequence(patterns=[maybe_star_pattern], **self.get_line_cols(p, 1))
            case [_, maybe_star_pattern, None]:
                p[0] = maybe_star_pattern
            case [_, maybe_star_pattern, ',', ast.MatchSequence(patterns=list(maybe_sequence_pattern))]:
                p[0] = ast.MatchSequence(patterns=[maybe_star_pattern] + maybe_sequence_pattern, **self.get_line_cols(p, 1))
            case [_, maybe_star_pattern, ',', maybe_sequence_pattern]:
                p[0] = ast.MatchSequence(patterns=[maybe_star_pattern, maybe_sequence_pattern], **self.get_line_cols(p, 1))
            case _:
                raise AssertionError()

    def p_open_sequence_pattern(self, p):
        if False:
            for i in range(10):
                print('nop')
        '\n        open_sequence_pattern : maybe_star_pattern COMMA\n                              | maybe_star_pattern COMMA maybe_sequence_pattern\n        '
        self.p_maybe_sequence_pattern(p)

    def p_maybe_star_pattern(self, p):
        if False:
            i = 10
            return i + 15
        '\n        maybe_star_pattern : pattern\n                           | star_pattern\n        '
        p[0] = p[1]

    def p_star_pattern(self, p):
        if False:
            i = 10
            return i + 15
        '\n        star_pattern : TIMES name_str\n        '
        (_, _, name) = p
        target = name if name != '_' else None
        p[0] = ast.MatchStar(name=target, **self.get_line_cols(p, 1))

    def p_value_pattern(self, p):
        if False:
            for i in range(10):
                print('nop')
        '\n        value_pattern : attr_name_with\n        '
        p[0] = ast.MatchValue(value=p[1], **self.get_line_cols(p, 1))

    def p_class_pattern(self, p):
        if False:
            for i in range(10):
                print('nop')
        '\n        class_pattern : attr_name LPAREN class_pattern_positional_part_start RPAREN\n        '
        (positional_patterns, keyword_patterns_key_value_tuple_list) = p[3]
        if keyword_patterns_key_value_tuple_list:
            (kwd_attrs, kwd_patterns) = list(zip(*keyword_patterns_key_value_tuple_list))
        else:
            (kwd_attrs, kwd_patterns) = ([], [])
        p[0] = ast.MatchClass(cls=p[1], patterns=positional_patterns, kwd_attrs=list(kwd_attrs), kwd_patterns=list(kwd_patterns), **self.get_line_cols(p, 1))

    def p_class_pattern_positional_part_start(self, p):
        if False:
            i = 10
            return i + 15
        '\n        class_pattern_positional_part_start :\n                                            | pattern\n                                            | pattern COMMA class_pattern_positional_part\n                                            | name_str EQUALS pattern\n                                            | name_str EQUALS pattern COMMA class_pattern_keyword_part\n        '
        match list(p):
            case [_]:
                p[0] = ([], [])
            case [_, pattern]:
                p[0] = ([pattern], [])
            case [_, pattern, ',', [names, patterns]]:
                p[0] = ([pattern] + names, patterns)
            case [_, name, '=', pattern]:
                p[0] = ([], [(name, pattern)])
            case [_, name, '=', pattern, ',', class_pattern_keyword_part]:
                p[0] = ([], [(name, pattern)] + class_pattern_keyword_part)
            case _:
                raise AssertionError()

    def p_class_pattern_positional_part_skip(self, p):
        if False:
            print('Hello World!')
        '\n        class_pattern_positional_part : class_pattern_keyword_part\n        '
        p[0] = ([], p[1])

    def p_class_pattern_positional_part(self, p):
        if False:
            for i in range(10):
                print('nop')
        '\n        class_pattern_positional_part : pattern\n                                      | pattern COMMA class_pattern_positional_part\n        '
        match list(p):
            case [_, pattern]:
                p[0] = ([pattern], [])
            case [_, pattern, ',', [names, patterns]]:
                p[0] = ([pattern] + names, patterns)
            case _:
                raise AssertionError()

    def p_class_pattern_keyword_part(self, p):
        if False:
            return 10
        '\n        class_pattern_keyword_part :\n                                   | COMMA\n                                   | name_str EQUALS pattern\n                                   | name_str EQUALS pattern COMMA class_pattern_keyword_part\n        '
        match list(p):
            case [_] | [_, ',']:
                p[0] = []
            case [_, name, '=', pattern]:
                p[0] = [(name, pattern)]
            case [_, name, '=', pattern, ',', class_pattern_keyword_part]:
                p[0] = [(name, pattern)] + class_pattern_keyword_part
            case _:
                raise AssertionError()

    def p_mapping_pattern(self, p):
        if False:
            i = 10
            return i + 15
        '\n        mapping_pattern : LBRACE mapping_pattern_args_start RBRACE\n        '
        (_, _, (keys, patterns, rest), _) = p
        p[0] = ast.MatchMapping(keys=keys, patterns=patterns, rest=rest, **self.get_line_cols(p, 1))

    def p_mapping_pattern_args_start(self, p):
        if False:
            i = 10
            return i + 15
        '\n        mapping_pattern_args_start :\n                                   | key_value_pattern\n                                   | key_value_pattern COMMA mapping_pattern_args_item_part\n                                   | double_star_pattern\n        '
        match list(p):
            case [_]:
                p[0] = ([], [], None)
            case [_, [key, value]]:
                p[0] = ([key], [value], None)
            case [_, [key, value], ',', [keys, values, rest]]:
                p[0] = ([key] + keys, [value] + values, rest)
            case [_, str(double_star_pattern)]:
                p[0] = ([], [], double_star_pattern)
            case _:
                raise AssertionError()

    def p_mapping_pattern_args_item_part_skip(self, p):
        if False:
            while True:
                i = 10
        '\n        mapping_pattern_args_item_part :\n                                       | double_star_pattern\n        '
        match list(p):
            case [_]:
                p[0] = ([], [], None)
            case [_, rest]:
                p[0] = ([], [], rest)
            case _:
                raise AssertionError()

    def p_mapping_pattern_args_item_part(self, p):
        if False:
            print('Hello World!')
        '\n        mapping_pattern_args_item_part : key_value_pattern\n                                       | key_value_pattern COMMA mapping_pattern_args_item_part\n        '
        match list(p):
            case [_, [key, value]]:
                p[0] = ([key], [value], None)
            case [_, [key, value], ',', [keys, values, rest]]:
                p[0] = ([key] + keys, [value] + values, rest)
            case _:
                raise AssertionError()

    def p_double_star_pattern(self, p):
        if False:
            print('Hello World!')
        '\n        double_star_pattern : POW capture_target_name comma_opt\n        '
        p[0] = p[2]

    def p_key_value_pattern(self, p):
        if False:
            while True:
                i = 10
        '\n        key_value_pattern : literal_expr COLON pattern\n                          | attr_name_with COLON pattern\n        '
        (_, key, _, value) = p
        p[0] = (key, value)