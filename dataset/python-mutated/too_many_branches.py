"""
Test for too many branches.
Taken from the pylint source 2023-02-03
"""

def wrong():
    if False:
        print('Hello World!')
    ' Has too many branches. '
    if 1:
        pass
    elif 1:
        pass
    elif 1:
        pass
    elif 1:
        pass
    elif 1:
        pass
    elif 1:
        pass
    try:
        pass
    finally:
        pass
    if 2:
        pass
    while True:
        pass
    if 1:
        pass
    elif 2:
        pass
    elif 3:
        pass

def good():
    if False:
        print('Hello World!')
    ' Too many branches only if we take\n    into consideration the nested functions.\n    '

    def nested_1():
        if False:
            for i in range(10):
                print('nop')
        ' empty '
        if 1:
            pass
        elif 2:
            pass
        elif 3:
            pass
        elif 4:
            pass
    nested_1()
    try:
        pass
    finally:
        pass
    try:
        pass
    finally:
        pass
    if 1:
        pass
    elif 2:
        pass
    elif 3:
        pass
    elif 4:
        pass
    elif 5:
        pass
    elif 6:
        pass
    elif 7:
        pass