def foo(x):
    if False:
        for i in range(10):
            print('nop')
    for i in range(x):
        try:
            pass
        finally:
            try:
                try:
                    print(x, i)
                finally:
                    try:
                        1 / 0
                    finally:
                        return 42
            finally:
                print('return')
                return 43
print(foo(4))