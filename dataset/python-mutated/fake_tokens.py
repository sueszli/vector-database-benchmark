def foo(request):
    if False:
        for i in range(10):
            print('nop')
    redirect(request.foo)
    return