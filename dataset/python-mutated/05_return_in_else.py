def parseline(self, line):
    if False:
        for i in range(10):
            print('nop')
    if not line:
        return 5
    elif line:
        if hasattr(self, 'do_shell'):
            line = 'shell'
        else:
            return 3 if line[3] else 4
    return 6

def find(domain):
    if False:
        for i in range(10):
            print('nop')
    for lang in domain:
        if lang:
            if all:
                domain.append(5)
            else:
                return lang
    return domain