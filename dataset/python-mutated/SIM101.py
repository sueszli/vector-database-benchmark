if isinstance(a, int) or isinstance(a, float):
    pass
if isinstance(a, (int, float)) or isinstance(a, bool):
    pass
if isinstance(a, int) or isinstance(a, float) or isinstance(b, bool):
    pass
if isinstance(b, bool) or isinstance(a, int) or isinstance(a, float):
    pass
if isinstance(a, int) or isinstance(b, bool) or isinstance(a, float):
    pass
if (isinstance(a, int) or isinstance(a, float)) and isinstance(b, bool):
    pass
if isinstance(a.b, int) or isinstance(a.b, float):
    pass
if isinstance(a(), int) or isinstance(a(), float):
    pass
if isinstance(a, int) and isinstance(b, bool) or isinstance(a, float):
    pass
if isinstance(a, bool) or isinstance(b, str):
    pass
if isinstance(a, int) or isinstance(a.b, float):
    pass
if isinstance(a, int) or unrelated_condition or isinstance(a, float):
    pass
if x or isinstance(a, int) or isinstance(a, float):
    pass
if x or y or isinstance(a, int) or isinstance(a, float) or z:
    pass

def f():
    if False:
        print('Hello World!')

    def isinstance(a, b):
        if False:
            while True:
                i = 10
        return False
    if isinstance(a, int) or isinstance(a, float):
        pass