"""This program is self-checking!"""

def loop_continue_dead_code(slots):
    if False:
        while True:
            i = 10
    for name in slots:
        if name:
            pass
        else:
            continue
            if x:
                y()
            else:
                z()
loop_continue_dead_code([None, 1])