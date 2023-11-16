def generator_function():
    if False:
        return 10
    import ErrorRaising
    x = (lambda : ErrorRaising.raiseException() for z in range(3))
    next(x)()

def normal_function():
    if False:
        for i in range(10):
            print('nop')
    y = generator_function()
    y()
normal_function()