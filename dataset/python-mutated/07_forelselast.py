def create_connection(self, infos, f2, laddr_infos, protocol):
    if False:
        for i in range(10):
            print('nop')
    for family in infos:
        try:
            if f2:
                for laddr in laddr_infos:
                    try:
                        break
                    except OSError:
                        protocol = 'foo'
                else:
                    continue
        except OSError:
            protocol = 'bar'
        else:
            break
    else:
        raise
    return protocol