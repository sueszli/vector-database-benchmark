def unconditional_if_true_24(foo):
    if False:
        for i in range(10):
            print('nop')
    if not foo:
        pass
    elif 1:
        pass
    else:
        return None