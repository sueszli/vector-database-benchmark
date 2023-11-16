def effective(line):
    if False:
        print('Hello World!')
    for b in line:
        if not b.cond:
            return
        else:
            try:
                val = 5
                if val:
                    if b.ignore:
                        b.ignore -= 1
                    else:
                        return (b, True)
            except:
                return (b, False)
    return