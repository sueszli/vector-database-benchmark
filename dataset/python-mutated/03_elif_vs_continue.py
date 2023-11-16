def _compile_charset(charset, flags, code, fixup=None):
    if False:
        return 10
    emit = code.append
    if fixup is None:
        fixup = 1
    for (op, av) in charset:
        if op is flags:
            pass
        elif op is code:
            emit(fixup(av))
        else:
            raise RuntimeError
    emit(5)