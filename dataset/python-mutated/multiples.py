from lark import Lark, UnexpectedInput
parser = Lark.open('multiples.lark', rel_to=__file__, parser='lalr')

def is_in_grammar(data):
    if False:
        while True:
            i = 10
    try:
        parser.parse(data)
    except UnexpectedInput:
        return False
    return True
for n_dec in range(100):
    n_bin = bin(n_dec)[2:]
    assert is_in_grammar('2:{}'.format(n_bin)) == (n_dec % 2 == 0)
    assert is_in_grammar('3:{}'.format(n_bin)) == (n_dec % 3 == 0)