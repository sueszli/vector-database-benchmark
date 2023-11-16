import dataclasses
import enum
import functools
import math
import re
from typing import Any, List, Text
EMPTY_ANSWER = 'none'
EMPTY_ANSWER_AGG = 'none'

def _split_thousands(delimiter, value):
    if False:
        return 10
    split = value.split(delimiter)
    return len(split) > 1 and any((len(x) == 3 for x in split))

def convert_to_float(value):
    if False:
        return 10
    'Converts value to a float using a series of increasingly complex heuristics.\n    Args:\n      value: object that needs to be converted. Allowed types include\n        float/int/strings.\n    Returns:\n      A float interpretation of value.\n    Raises:\n      ValueError if the float conversion of value fails.\n    '
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    if not isinstance(value, str):
        raise ValueError("Argument value is not a string. Can't parse it as float")
    sanitized = value
    try:
        if '.' in sanitized and ',' in sanitized:
            return float(sanitized.replace(',', ''))
        if ',' in sanitized and _split_thousands(',', sanitized):
            return float(sanitized.replace(',', ''))
        if ',' in sanitized and sanitized.count(',') == 1 and (not _split_thousands(',', sanitized)):
            return float(sanitized.replace(',', '.'))
        if sanitized.count('.') > 1:
            return float(sanitized.replace('.', ''))
        if sanitized.count(',') > 1:
            return float(sanitized.replace(',', ''))
        return float(sanitized)
    except ValueError:
        raise ValueError('Unable to convert value to float')

def _normalize_float(answer):
    if False:
        print('Hello World!')
    if answer is None:
        return None
    try:
        value = convert_to_float(answer)
        if isinstance(value, float) and math.isnan(value):
            return None
        return value
    except ValueError:
        return answer.lower()
_TYPE_CONVERTER = {'text': lambda x: x, 'real': convert_to_float}

class _Aggregation(enum.Enum):
    """Aggregations as defined by WikiSQL. Indexes match the data."""
    NONE = 0
    MAX = 1
    MIN = 2
    COUNT = 3
    SUM = 4
    AVERAGE = 5

class _Operator(enum.Enum):
    """The boolean operators used by WikiSQL. Indexes match the data."""
    EQUALS = 0
    GREATER = 1
    LESSER = 2

@dataclasses.dataclass
class _Condition:
    """Represents an SQL where clauses (e.g A = "a" or B > 5)."""
    column: Text
    operator: _Operator
    cmp_value: Any
_TOKENIZER = re.compile('\\w+|[^\\w\\s]+', re.UNICODE | re.MULTILINE | re.DOTALL)

def _normalize_for_match(x):
    if False:
        i = 10
        return i + 15
    return list(_TOKENIZER.findall(x.lower()))

def _compare(operator, src, tgt):
    if False:
        print('Hello World!')
    if operator == _Operator.EQUALS:
        return src == tgt
    elif operator == _Operator.GREATER:
        return src > tgt
    elif operator == _Operator.LESSER:
        return src < tgt
    raise ValueError(f'Unknown operator: {operator}')

def _parse_value(table, column, cell_value):
    if False:
        for i in range(10):
            print('nop')
    'Convert numeric values to floats and keeps everything else as string.'
    types = table['types']
    return _TYPE_CONVERTER[types[column]](cell_value)

def _is_string(x):
    if False:
        i = 10
        return i + 15
    return isinstance(x, str)

def _respect_conditions(table, row, conditions):
    if False:
        while True:
            i = 10
    "True if 'row' satisfies all 'conditions'."
    for cond in conditions:
        table_value = row[cond.column]
        cmp_value = _parse_value(table, cond.column, cond.cmp_value)
        if _is_string(table_value) and _is_string(cmp_value):
            table_value = _normalize_for_match(table_value)
            cmp_value = _normalize_for_match(cmp_value)
        if not isinstance(table_value, type(cmp_value)):
            raise ValueError('Type difference {} != {}'.format(type(table_value), type(cmp_value)))
        if not _compare(cond.operator, table_value, cmp_value):
            return False
    return True

def _get_float_answer(table, answer_coordinates, aggregation_op):
    if False:
        return 10
    'Applies operation to produce reference float answer.'
    if not answer_coordinates:
        if aggregation_op == _Aggregation.COUNT:
            return 0.0
        else:
            return EMPTY_ANSWER_AGG
    if aggregation_op == _Aggregation.COUNT:
        return float(len(answer_coordinates))
    values = [table['rows'][i][j] for (i, j) in answer_coordinates]
    if len(answer_coordinates) == 1:
        try:
            return convert_to_float(values[0])
        except ValueError as e:
            if aggregation_op != _Aggregation.NONE:
                raise e
    if aggregation_op == _Aggregation.NONE:
        return None
    if not all((isinstance(v, (int, float)) for v in values)):
        return None
    if aggregation_op == _Aggregation.SUM:
        return float(sum(values))
    elif aggregation_op == _Aggregation.AVERAGE:
        return sum(values) / len(answer_coordinates)
    else:
        raise ValueError(f'Unknown aggregation: {aggregation_op}')

def _get_answer_coordinates(table, sql_query):
    if False:
        for i in range(10):
            print('nop')
    'Retrieves references coordinates by executing SQL.'
    aggregation_op_index = sql_query['agg']
    if aggregation_op_index >= 3:
        aggregation_op = _Aggregation(aggregation_op_index)
    else:
        aggregation_op = _Aggregation.NONE
    target_column = sql_query['sel']
    conditions = [_Condition(column, _Operator(operator), cmp_value) for (column, operator, cmp_value) in zip(sql_query['conds']['column_index'], sql_query['conds']['operator_index'], sql_query['conds']['condition'])]
    indices = []
    for row in range(len(table['rows'])):
        if _respect_conditions(table, table['rows'][row], conditions):
            indices.append((row, target_column))
    if not indices:
        return ([], aggregation_op)
    if len(indices) == 1:
        return (indices, aggregation_op)
    if aggregation_op_index in (1, 2):
        operators = {2: min, 1: max}
        values = [(table['rows'][i][j], index) for (index, (i, j)) in enumerate(indices)]
        reduced = functools.reduce(operators[sql_query['agg']], values)
        ret = [indices[reduced[1]]]
        return (ret, _Aggregation.NONE)
    return (indices, aggregation_op)

def _get_answer_text(table, answer_coordinates, float_answer):
    if False:
        for i in range(10):
            print('nop')
    if float_answer is not None:
        return [str(float_answer)]
    return [str(table['real_rows'][r][c]) for (r, c) in answer_coordinates]

def retrieve_wikisql_query_answer_tapas(table, example) -> List:
    if False:
        while True:
            i = 10
    (answer_coordinates, aggregation_op) = _get_answer_coordinates(table, example)
    float_answer = _get_float_answer(table, answer_coordinates, aggregation_op)
    answer_text = _get_answer_text(table, answer_coordinates, float_answer)
    if len(answer_text) == 0:
        answer_text = [EMPTY_ANSWER]
    return answer_text