def not_append_func10(default=[]):
    if False:
        for i in range(10):
            print('nop')
    default = [str(x) for x in default]
    default.append(5)