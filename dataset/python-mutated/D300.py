def with_backslash():
    if False:
        for i in range(10):
            print('nop')
    'Sum\\mary.'

def ends_in_quote():
    if False:
        i = 10
        return i + 15
    'Sum\\mary."'

def contains_quote():
    if False:
        i = 10
        return i + 15
    'Sum"\\mary.'

def contains_triples(t):
    if False:
        return 10
    '(\'\'\'|""")'

def contains_triples(t):
    if False:
        i = 10
        return i + 15
    '(\'\'\'|""")'

def contains_triples(t):
    if False:
        return 10
    '(""")'