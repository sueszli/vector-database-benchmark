import tokenize
import string

def parse_signature(sig):
    if False:
        return 10
    "Parse generalized ufunc signature.\n\n    NOTE: ',' (COMMA) is a delimiter; not separator.\n          This means trailing comma is legal.\n    "

    def stripws(s):
        if False:
            return 10
        return ''.join((c for c in s if c not in string.whitespace))

    def tokenizer(src):
        if False:
            i = 10
            return i + 15

        def readline():
            if False:
                return 10
            yield src
        gen = readline()
        return tokenize.generate_tokens(lambda : next(gen))

    def parse(src):
        if False:
            return 10
        tokgen = tokenizer(src)
        while True:
            tok = next(tokgen)
            if tok[1] == '(':
                symbols = []
                while True:
                    tok = next(tokgen)
                    if tok[1] == ')':
                        break
                    elif tok[0] == tokenize.NAME:
                        symbols.append(tok[1])
                    elif tok[1] == ',':
                        continue
                    else:
                        raise ValueError('bad token in signature "%s"' % tok[1])
                yield tuple(symbols)
                tok = next(tokgen)
                if tok[1] == ',':
                    continue
                elif tokenize.ISEOF(tok[0]):
                    break
            elif tokenize.ISEOF(tok[0]):
                break
            else:
                raise ValueError('bad token in signature "%s"' % tok[1])
    (ins, _, outs) = stripws(sig).partition('->')
    inputs = list(parse(ins))
    outputs = list(parse(outs))
    isym = set()
    osym = set()
    for grp in inputs:
        isym |= set(grp)
    for grp in outputs:
        osym |= set(grp)
    diff = osym.difference(isym)
    if diff:
        raise NameError('undefined output symbols: %s' % ','.join(sorted(diff)))
    return (inputs, outputs)