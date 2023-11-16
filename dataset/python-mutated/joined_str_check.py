def joined_str_invalid(self, lhs: str, n: int, rule, tree, tokens: list, first: int, last: int) -> bool:
    if False:
        while True:
            i = 10
    expr1 = tree[0]
    if expr1 != 'expr':
        return False
    load_str = expr1[0]
    if load_str != 'LOAD_STR':
        return False
    format_value_equal = load_str.attr
    if format_value_equal[-1] != '=':
        return False
    expr2 = tree[1]
    if expr2 != 'expr':
        return False
    formatted_value = expr2[0]
    if not formatted_value.kind.startswith('formatted_value'):
        return False
    expr2a = formatted_value[0]
    if expr2a != 'expr':
        return False
    load_const = expr2a[0]
    if load_const == 'LOAD_CONST':
        format_value2 = load_const.attr
        return str(format_value2) == format_value_equal[:-1]
    return True