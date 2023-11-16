def foo(x):
    if False:
        return 10
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
                print('continue')
                continue
print(foo(4))