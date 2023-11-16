from ast import literal_eval
eval('3 + 4')
literal_eval({1: 2})

def fn() -> None:
    if False:
        return 10
    eval('3 + 4')