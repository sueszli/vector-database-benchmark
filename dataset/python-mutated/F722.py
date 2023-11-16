class A:
    pass

def f() -> 'A':
    if False:
        i = 10
        return i + 15
    pass

def g() -> '///':
    if False:
        i = 10
        return i + 15
    pass
X: 'List[int]☃' = []