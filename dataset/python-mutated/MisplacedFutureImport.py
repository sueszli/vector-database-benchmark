def f():
    if False:
        print('Hello World!')
    from __future__ import print_function
    print(locals())