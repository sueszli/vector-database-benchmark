import miasm.expression.expression as m2_expr
jok1 = m2_expr.ExprId('jok1', 32)
jok2 = m2_expr.ExprId('jok2', 32)
jok3 = m2_expr.ExprId('jok3', 32)
jok_small = m2_expr.ExprId('jok_small', 1)

def __ExprOp_cond(op, arg1, arg2):
    if False:
        return 10
    'Return an ExprOp standing for arg1 op arg2 with size to 1'
    ec = m2_expr.ExprOp(op, arg1, arg2)
    return ec

def ExprOp_inf_signed(arg1, arg2):
    if False:
        return 10
    'Return an ExprOp standing for arg1 <s arg2'
    return __ExprOp_cond(m2_expr.TOK_INF_SIGNED, arg1, arg2)

def ExprOp_inf_unsigned(arg1, arg2):
    if False:
        print('Hello World!')
    'Return an ExprOp standing for arg1 <s arg2'
    return __ExprOp_cond(m2_expr.TOK_INF_UNSIGNED, arg1, arg2)

def ExprOp_equal(arg1, arg2):
    if False:
        i = 10
        return i + 15
    'Return an ExprOp standing for arg1 == arg2'
    return __ExprOp_cond(m2_expr.TOK_EQUAL, arg1, arg2)

def __check_msb(e):
    if False:
        i = 10
        return i + 15
    'If @e stand for the most significant bit of its arg, return the arg;\n    False otherwise'
    if not isinstance(e, m2_expr.ExprSlice):
        return False
    arg = e.arg
    if e.start != arg.size - 1 or e.stop != arg.size:
        return False
    return arg

def __match_expr_wrap(e, to_match, jok_list):
    if False:
        return 10
    'Wrapper around match_expr to canonize pattern'
    to_match = to_match.canonize()
    r = m2_expr.match_expr(e, to_match, jok_list)
    if r is False:
        return False
    if r == {}:
        return False
    return r

def expr_simp_inf_signed(expr_simp, e):
    if False:
        return 10
    '((x - y) ^ ((x ^ y) & ((x - y) ^ x))) [31:32] == x <s y'
    arg = __check_msb(e)
    if arg is False:
        return e
    to_match = jok3 ^ (jok1 ^ jok2) & (jok3 ^ jok1)
    r = __match_expr_wrap(arg, to_match, [jok1, jok2, jok3])
    if r is False:
        return e
    new_j3 = expr_simp(r[jok3])
    sub = expr_simp(r[jok1] - r[jok2])
    if new_j3 == sub:
        return ExprOp_inf_signed(r[jok1], r[jok2])
    else:
        return e

def expr_simp_inf_unsigned_inversed(expr_simp, e):
    if False:
        print('Hello World!')
    '((x - y) ^ ((x ^ y) & ((x - y) ^ x))) ^ x ^ y [31:32] == x <u y'
    arg = __check_msb(e)
    if arg is False:
        return e
    to_match = jok3 ^ (jok1 ^ jok2) & (jok3 ^ jok1) ^ jok1 ^ jok2
    r = __match_expr_wrap(arg, to_match, [jok1, jok2, jok3])
    if r is False:
        return e
    new_j3 = expr_simp(r[jok3])
    sub = expr_simp(r[jok1] - r[jok2])
    if new_j3 == sub:
        return ExprOp_inf_unsigned(r[jok1], r[jok2])
    else:
        return e

def expr_simp_inverse(expr_simp, e):
    if False:
        print('Hello World!')
    '(x <u y) ^ ((x ^ y) [31:32]) == x <s y,\n    (x <s y) ^ ((x ^ y) [31:32]) == x <u y'
    to_match = ExprOp_inf_unsigned(jok1, jok2) ^ jok_small
    r = __match_expr_wrap(e, to_match, [jok1, jok2, jok_small])
    if r is False:
        to_match = ExprOp_inf_signed(jok1, jok2) ^ jok_small
        r = __match_expr_wrap(e, to_match, [jok1, jok2, jok_small])
        if r is False:
            return e
        cur_sig = m2_expr.TOK_INF_SIGNED
    else:
        cur_sig = m2_expr.TOK_INF_UNSIGNED
    arg = __check_msb(r[jok_small])
    if arg is False:
        return e
    if not isinstance(arg, m2_expr.ExprOp) or arg.op != '^':
        return e
    op_args = arg.args
    if len(op_args) != 2:
        return e
    if r[jok1] not in op_args or r[jok2] not in op_args:
        return e
    if cur_sig == m2_expr.TOK_INF_UNSIGNED:
        return ExprOp_inf_signed(r[jok1], r[jok2])
    else:
        return ExprOp_inf_unsigned(r[jok1], r[jok2])

def expr_simp_equal(expr_simp, e):
    if False:
        return 10
    '(x - y)?(0:1) == (x == y)'
    to_match = m2_expr.ExprCond(jok1 + jok2, m2_expr.ExprInt(0, 1), m2_expr.ExprInt(1, 1))
    r = __match_expr_wrap(e, to_match, [jok1, jok2])
    if r is False:
        return e
    return ExprOp_equal(r[jok1], expr_simp(-r[jok2]))