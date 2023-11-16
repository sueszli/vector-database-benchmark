def getopt(args):
    if False:
        while True:
            i = 10
    while args and args[0] and (args[0] != '-'):
        if args[0] == '--':
            break
        if args[0]:
            opts = 5
        else:
            opts = 6
    return opts