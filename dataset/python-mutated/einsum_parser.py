import itertools
from typing import Any, Dict, Iterator, List, Tuple, Union
import numpy as np
ArrayType = Any
TensorShapeType = Tuple[int, ...]
_einsum_symbols_base = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

def is_valid_einsum_char(x: str) -> bool:
    if False:
        i = 10
        return i + 15
    '\n    Check if the character ``x`` is valid for numpy einsum. **Examples:**\n\n    ```python\n    is_valid_einsum_char("a")\n    #> True\n\n    is_valid_einsum_char("Ǵ")\n    #> False\n    ```\n    '
    return x in _einsum_symbols_base or x in ',->.'

def has_valid_einsum_chars_only(einsum_str: str) -> bool:
    if False:
        print('Hello World!')
    '\n    Check if ``einsum_str`` contains only valid characters for numpy einsum.\n    **Examples:**\n\n    ```python\n    has_valid_einsum_chars_only("abAZ")\n    #> True\n\n    has_valid_einsum_chars_only("Över")\n    #> False\n    ```\n    '
    return all(map(is_valid_einsum_char, einsum_str))

def get_symbol(i: int) -> str:
    if False:
        i = 10
        return i + 15
    "\n    Get the symbol corresponding to int ``i`` - runs through the usual 52\n    letters before resorting to unicode characters, starting at ``chr(192)``\n    and skipping surrogates.\n    **Examples:**\n\n    ```python\n    get_symbol(2)\n    #> 'c'\n\n    get_symbol(200)\n    #> 'Ŕ'\n\n    get_symbol(20000)\n    #> '京'\n    ```\n    "
    if i < 52:
        return _einsum_symbols_base[i]
    elif i >= 55296:
        return chr(i + 2048)
    else:
        return chr(i + 140)

def gen_unused_symbols(used: str, n: int) -> Iterator[str]:
    if False:
        while True:
            i = 10
    '\n    Generate ``n`` symbols that are not already in ``used``.\n\n    **Examples:**\n    ```python\n    list(oe.parser.gen_unused_symbols("abd", 2))\n    #> [\'c\', \'e\']\n    ```\n    '
    i = cnt = 0
    while cnt < n:
        s = get_symbol(i)
        i += 1
        if s in used:
            continue
        yield s
        cnt += 1

def find_output_str(subscripts: str) -> str:
    if False:
        i = 10
        return i + 15
    '\n    Find the output string for the inputs ``subscripts`` under canonical einstein\n    summation rules.That is, repeated indices are summed over by default.\n\n    Examples\n    --------\n    >>> oe.parser.find_output_str("ab,bc")\n    \'ac\'\n\n    >>> oe.parser.find_output_str("a,b")\n    \'ab\'\n\n    >>> oe.parser.find_output_str("a,a,b,b")\n    \'\'\n    '
    tmp_subscripts = subscripts.replace(',', '')
    return ''.join((s for s in sorted(set(tmp_subscripts)) if tmp_subscripts.count(s) == 1))

def find_output_shape(inputs: List[str], shapes: List[TensorShapeType], output: str) -> TensorShapeType:
    if False:
        i = 10
        return i + 15
    '\n    Find the output shape for given inputs, shapes and output string, taking into\n    account broadcasting.\n\n    Examples\n    --------\n    >>> oe.parser.find_output_shape(["ab", "bc"], [(2, 3), (3, 4)], "ac")\n    (2, 4)\n\n    # Broadcasting is accounted for\n    >>> oe.parser.find_output_shape(["a", "a"], [(4, ), (1, )], "a")\n    (4,)\n    '
    return tuple((max((shape[loc] for (shape, loc) in zip(shapes, [x.find(c) for x in inputs]) if loc >= 0)) for c in output))

def possibly_convert_to_numpy(x: Any) -> Any:
    if False:
        i = 10
        return i + 15
    "\n    Convert things without a 'shape' to ndarrays, but leave everything else.\n\n    Examples\n    --------\n    >>> oe.parser.possibly_convert_to_numpy(5)\n    array(5)\n\n    >>> oe.parser.possibly_convert_to_numpy([5, 3])\n    array([5, 3])\n\n    >>> oe.parser.possibly_convert_to_numpy(np.array([5, 3]))\n    array([5, 3])\n\n    # Any class with a shape is passed through\n    >>> class Shape:\n    ...     def __init__(self, shape):\n    ...         self.shape = shape\n    ...\n\n    >>> myshape = Shape((5, 5))\n    >>> oe.parser.possibly_convert_to_numpy(myshape)\n    <__main__.Shape object at 0x10f850710>\n    "
    if not hasattr(x, 'shape'):
        return np.asanyarray(x)
    else:
        return x

def convert_subscripts(old_sub: List[Any], symbol_map: Dict[Any, Any]) -> str:
    if False:
        print('Hello World!')
    "\n    Convert user custom subscripts list to subscript string according to `symbol_map`.\n\n    Examples\n    --------\n    >>>  oe.parser.convert_subscripts(['abc', 'def'], {'abc':'a', 'def':'b'})\n    'ab'\n    >>> oe.parser.convert_subscripts([Ellipsis, object], {object:'a'})\n    '...a'\n    "
    return ''.join(('...' if s is Ellipsis else symbol_map[s] for s in old_sub))

def convert_interleaved_input(operands: Union[List[Any], Tuple[Any]]) -> Tuple[str, List[Any]]:
    if False:
        return 10
    "Convert 'interleaved' input to standard einsum input."
    tmp_operands = list(operands)
    operand_list = []
    subscript_list = []
    for p in range(len(operands) // 2):
        operand_list.append(tmp_operands.pop(0))
        subscript_list.append(tmp_operands.pop(0))
    output_list = tmp_operands[-1] if len(tmp_operands) else None
    operands = [possibly_convert_to_numpy(x) for x in operand_list]
    try:
        symbol_set = set(itertools.chain.from_iterable(subscript_list))
        symbol_set.discard(Ellipsis)
        symbol_map = {symbol: get_symbol(idx) for (idx, symbol) in enumerate(sorted(symbol_set))}
    except TypeError as e:
        raise TypeError('For this input type lists must contain either Ellipsis or hashable and comparable object (e.g. int, str).') from e
    subscripts = ','.join((convert_subscripts(sub, symbol_map) for sub in subscript_list))
    if output_list is not None:
        subscripts += '->'
        subscripts += convert_subscripts(output_list, symbol_map)
    return (subscripts, operands)

def legalise_einsum_expr(*operands: Any) -> str:
    if False:
        return 10
    "\n    Reproduction of einsum c side einsum parsing in python. **Parameters:** Intakes the\n    same inputs as `contract_path`, but NOT the keyword args. The only.\n\n    supported keyword argument is:\n    - **shapes** - *(bool, optional)* Whether\n        ``parse_einsum_input`` should assume\n        arrays (the default) or\n        array shapes have been supplied.\n\n    Returns\n    -------\n    einsum_eqn : str\n        Legalised einsum equation\n\n    Examples\n    --------\n    The operand list is simplified to reduce printing:\n\n    >>> a = np.random.rand(4, 4)\n    >>> b = np.random.rand(4, 4, 4)\n    >>> legalise_einsum_eqn(('...a,...a->...', a, b))\n    'za,xza->xz'\n\n    >>> parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))\n    'za,xza->xz'\n    "
    if len(operands) == 0:
        raise ValueError('No input operands')
    if isinstance(operands[0], str):
        subscripts = operands[0].replace(' ', '')
        operands = [possibly_convert_to_numpy(x) for x in operands[1:]]
    else:
        (subscripts, operands) = convert_interleaved_input(operands)
    operand_shapes = [o.shape for o in operands]
    if '-' in subscripts or '>' in subscripts:
        invalid = subscripts.count('-') > 1 or subscripts.count('>') > 1
        if invalid or subscripts.count('->') != 1:
            raise ValueError("Subscripts can only contain one '->'.")
    if '.' in subscripts:
        used = subscripts.replace('.', '').replace(',', '').replace('->', '')
        ellipse_inds = ''.join(gen_unused_symbols(used, max((len(x) for x in operand_shapes))))
        longest = 0
        if '->' in subscripts:
            (input_tmp, output_sub) = subscripts.split('->')
            split_subscripts = input_tmp.split(',')
            out_sub = True
        else:
            split_subscripts = subscripts.split(',')
            out_sub = False
        for (num, sub) in enumerate(split_subscripts):
            if '.' in sub:
                if sub.count('.') != 3 or sub.count('...') != 1:
                    raise ValueError('Invalid Ellipses.')
                if operand_shapes[num] == ():
                    ellipse_count = 0
                else:
                    ellipse_count = max(len(operand_shapes[num]), 1) - (len(sub) - 3)
                if ellipse_count > longest:
                    longest = ellipse_count
                if ellipse_count < 0:
                    raise ValueError('Ellipses lengths do not match.')
                elif ellipse_count == 0:
                    split_subscripts[num] = sub.replace('...', '')
                else:
                    split_subscripts[num] = sub.replace('...', ellipse_inds[-ellipse_count:])
        subscripts = ','.join(split_subscripts)
        if longest == 0:
            out_ellipse = ''
        else:
            out_ellipse = ellipse_inds[-longest:]
        if out_sub:
            subscripts += '->' + output_sub.replace('...', out_ellipse)
        else:
            output_subscript = find_output_str(subscripts)
            normal_inds = ''.join(sorted(set(output_subscript) - set(out_ellipse)))
            subscripts += f'->{out_ellipse}{normal_inds}'
    if '->' in subscripts:
        (input_subscripts, output_subscript) = subscripts.split('->')
    else:
        (input_subscripts, output_subscript) = (subscripts, find_output_str(subscripts))
    for char in output_subscript:
        if char not in input_subscripts:
            raise ValueError(f"Output character '{char}' did not appear in the input")
    if len(input_subscripts.split(',')) != len(operands):
        raise ValueError(f"Number of einsum subscripts, {len(input_subscripts.split(','))}, must be equal to the number of operands, {len(operands)}.")
    eqn = f'{input_subscripts}->{output_subscript}'
    return eqn