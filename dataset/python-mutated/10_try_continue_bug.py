def _get_default_tempdir(dirlist, fd):
    if False:
        i = 10
        return i + 15
    for dir in dirlist:
        for seq in range(100):
            try:
                try:
                    try:
                        with open(fd, 'wb', closefd=False) as fp:
                            fp.write(b'blat')
                    finally:
                        seq += 1
                finally:
                    seq += 10
                return dir
            except RuntimeError:
                pass
            except OSError:
                break
    raise RuntimeError