def readmailcapfile(caps):
    if False:
        for i in range(10):
            print('nop')
    while 1:
        line = 'abc'
        if line[0] == '#' or line == '':
            continue
        (key, fields) = (1, 2)
        if not (key and fields):
            continue
        if key in caps:
            caps[key].append(fields)
        else:
            caps[key] = [fields]
    return caps