def f():
    if False:
        return 10
    match (0, 1, 2):
        case [*x]:
            pass