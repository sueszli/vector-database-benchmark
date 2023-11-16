"""Tokenizers for three string formatting methods"""
from __future__ import annotations
from enum import Enum, unique
from typing import Final
from mypy.checkstrformat import ConversionSpecifier, parse_conversion_specifiers, parse_format_value
from mypy.errors import Errors
from mypy.messages import MessageBuilder
from mypy.nodes import Context, Expression
from mypy.options import Options
from mypyc.ir.ops import Integer, Value
from mypyc.ir.rtypes import c_pyssize_t_rprimitive, is_bytes_rprimitive, is_int_rprimitive, is_short_int_rprimitive, is_str_rprimitive
from mypyc.irbuild.builder import IRBuilder
from mypyc.primitives.bytes_ops import bytes_build_op
from mypyc.primitives.int_ops import int_to_str_op
from mypyc.primitives.str_ops import str_build_op, str_op

@unique
class FormatOp(Enum):
    """FormatOp represents conversion operations of string formatting during
    compile time.

    Compare to ConversionSpecifier, FormatOp has fewer attributes.
    For example, to mark a conversion from any object to string,
    ConversionSpecifier may have several representations, like '%s', '{}'
    or '{:{}}'. However, there would only exist one corresponding FormatOp.
    """
    STR = 's'
    INT = 'd'
    BYTES = 'b'

def generate_format_ops(specifiers: list[ConversionSpecifier]) -> list[FormatOp] | None:
    if False:
        return 10
    'Convert ConversionSpecifier to FormatOp.\n\n    Different ConversionSpecifiers may share a same FormatOp.\n    '
    format_ops = []
    for spec in specifiers:
        if spec.whole_seq == '%s' or spec.whole_seq == '{:{}}':
            format_op = FormatOp.STR
        elif spec.whole_seq == '%d':
            format_op = FormatOp.INT
        elif spec.whole_seq == '%b':
            format_op = FormatOp.BYTES
        elif spec.whole_seq:
            return None
        else:
            format_op = FormatOp.STR
        format_ops.append(format_op)
    return format_ops

def tokenizer_printf_style(format_str: str) -> tuple[list[str], list[FormatOp]] | None:
    if False:
        i = 10
        return i + 15
    'Tokenize a printf-style format string using regex.\n\n    Return:\n        A list of string literals and a list of FormatOps.\n    '
    literals: list[str] = []
    specifiers: list[ConversionSpecifier] = parse_conversion_specifiers(format_str)
    format_ops = generate_format_ops(specifiers)
    if format_ops is None:
        return None
    last_end = 0
    for spec in specifiers:
        cur_start = spec.start_pos
        literals.append(format_str[last_end:cur_start])
        last_end = cur_start + len(spec.whole_seq)
    literals.append(format_str[last_end:])
    return (literals, format_ops)
EMPTY_CONTEXT: Final = Context()

def tokenizer_format_call(format_str: str) -> tuple[list[str], list[FormatOp]] | None:
    if False:
        while True:
            i = 10
    'Tokenize a str.format() format string.\n\n    The core function parse_format_value() is shared with mypy.\n    With these specifiers, we then parse the literal substrings\n    of the original format string and convert `ConversionSpecifier`\n    to `FormatOp`.\n\n    Return:\n        A list of string literals and a list of FormatOps. The literals\n        are interleaved with FormatOps and the length of returned literals\n        should be exactly one more than FormatOps.\n        Return None if it cannot parse the string.\n    '
    specifiers = parse_format_value(format_str, EMPTY_CONTEXT, MessageBuilder(Errors(Options()), {}))
    if specifiers is None:
        return None
    format_ops = generate_format_ops(specifiers)
    if format_ops is None:
        return None
    literals: list[str] = []
    last_end = 0
    for spec in specifiers:
        literals.append(format_str[last_end:spec.start_pos - 1])
        last_end = spec.start_pos + len(spec.whole_seq) + 1
    literals.append(format_str[last_end:])
    literals = [x.replace('{{', '{').replace('}}', '}') for x in literals]
    return (literals, format_ops)

def convert_format_expr_to_str(builder: IRBuilder, format_ops: list[FormatOp], exprs: list[Expression], line: int) -> list[Value] | None:
    if False:
        print('Hello World!')
    'Convert expressions into string literal objects with the guidance\n    of FormatOps. Return None when fails.'
    if len(format_ops) != len(exprs):
        return None
    converted = []
    for (x, format_op) in zip(exprs, format_ops):
        node_type = builder.node_type(x)
        if format_op == FormatOp.STR:
            if is_str_rprimitive(node_type):
                var_str = builder.accept(x)
            elif is_int_rprimitive(node_type) or is_short_int_rprimitive(node_type):
                var_str = builder.call_c(int_to_str_op, [builder.accept(x)], line)
            else:
                var_str = builder.call_c(str_op, [builder.accept(x)], line)
        elif format_op == FormatOp.INT:
            if is_int_rprimitive(node_type) or is_short_int_rprimitive(node_type):
                var_str = builder.call_c(int_to_str_op, [builder.accept(x)], line)
            else:
                return None
        else:
            return None
        converted.append(var_str)
    return converted

def join_formatted_strings(builder: IRBuilder, literals: list[str] | None, substitutions: list[Value], line: int) -> Value:
    if False:
        return 10
    "Merge the list of literals and the list of substitutions\n    alternatively using 'str_build_op'.\n\n    `substitutions` is the result value of formatting conversions.\n\n    If the `literals` is set to None, we simply join the substitutions;\n    Otherwise, the `literals` is the literal substrings of the original\n    format string and its length should be exactly one more than\n    substitutions.\n\n    For example:\n    (1)    'This is a %s and the value is %d'\n        -> literals: ['This is a ', ' and the value is', '']\n    (2)    '{} and the value is {}'\n        -> literals: ['', ' and the value is', '']\n    "
    result_list: list[Value] = [Integer(0, c_pyssize_t_rprimitive)]
    if literals is not None:
        for (a, b) in zip(literals, substitutions):
            if a:
                result_list.append(builder.load_str(a))
            result_list.append(b)
        if literals[-1]:
            result_list.append(builder.load_str(literals[-1]))
    else:
        result_list.extend(substitutions)
    if len(result_list) == 1:
        return builder.load_str('')
    if not substitutions and len(result_list) == 2:
        return result_list[1]
    result_list[0] = Integer(len(result_list) - 1, c_pyssize_t_rprimitive)
    return builder.call_c(str_build_op, result_list, line)

def convert_format_expr_to_bytes(builder: IRBuilder, format_ops: list[FormatOp], exprs: list[Expression], line: int) -> list[Value] | None:
    if False:
        print('Hello World!')
    'Convert expressions into bytes literal objects with the guidance\n    of FormatOps. Return None when fails.'
    if len(format_ops) != len(exprs):
        return None
    converted = []
    for (x, format_op) in zip(exprs, format_ops):
        node_type = builder.node_type(x)
        if format_op == FormatOp.BYTES or format_op == FormatOp.STR:
            if is_bytes_rprimitive(node_type):
                var_bytes = builder.accept(x)
            else:
                return None
        else:
            return None
        converted.append(var_bytes)
    return converted

def join_formatted_bytes(builder: IRBuilder, literals: list[str], substitutions: list[Value], line: int) -> Value:
    if False:
        print('Hello World!')
    "Merge the list of literals and the list of substitutions\n    alternatively using 'bytes_build_op'."
    result_list: list[Value] = [Integer(0, c_pyssize_t_rprimitive)]
    for (a, b) in zip(literals, substitutions):
        if a:
            result_list.append(builder.load_bytes_from_str_literal(a))
        result_list.append(b)
    if literals[-1]:
        result_list.append(builder.load_bytes_from_str_literal(literals[-1]))
    if len(result_list) == 1:
        return builder.load_bytes_from_str_literal('')
    if not substitutions and len(result_list) == 2:
        return result_list[1]
    result_list[0] = Integer(len(result_list) - 1, c_pyssize_t_rprimitive)
    return builder.call_c(bytes_build_op, result_list, line)