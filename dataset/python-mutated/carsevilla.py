keys = ' -,.?!-ABC-DEF-GHI-JKL-MNO-PQRS-TUV-WXYZ'

def T9(input: str) -> str:
    if False:
        i = 10
        return i + 15
    return ''.join([keys.split('-')[int(v[0])][len(v) - 1] for v in input.split('-')])