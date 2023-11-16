""" Tools for construct tests.

"""

def generateConstructCases(construct_source_code):
    if False:
        print('Hello World!')
    inside = False
    case = 0
    case_1 = []
    case_2 = []
    for line in construct_source_code.splitlines():
        if not inside or case == 1:
            case_1.append(line)
        else:
            case_1.append('')
        if '# construct_end' in line:
            inside = False
        if '# construct_alternative' in line:
            case = 2
        if not inside or case == 2:
            case_2.append(line)
        else:
            case_2.append('')
        if '# construct_begin' in line:
            inside = True
            case = 1
    return ('\n'.join(case_1), '\n'.join(case_2))