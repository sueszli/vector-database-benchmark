from .Format import Format, Group, Scheme, Sign, Symbol

def money(decimals, sign=Sign.default):
    if False:
        while True:
            i = 10
    return Format(group=Group.yes, precision=decimals, scheme=Scheme.fixed, sign=sign, symbol=Symbol.yes)

def percentage(decimals, rounded=False):
    if False:
        print('Hello World!')
    if not isinstance(rounded, bool):
        raise TypeError('expected rounded to be a boolean')
    rounded = Scheme.percentage_rounded if rounded else Scheme.percentage
    return Format(scheme=rounded, precision=decimals)