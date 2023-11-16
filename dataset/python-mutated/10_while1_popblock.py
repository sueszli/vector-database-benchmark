def readmailcapfile(line):
    if False:
        for i in range(10):
            print('nop')
    while 1:
        if not line:
            break
        if line[0] == '#' or line.strip() == '':
            continue
        if not line:
            continue
        for j in range(3):
            line[j] = line[j].strip()
        if '/' in line:
            line['/'].append('a')
        else:
            line['/'] = 'a'
    return