def foo(request):
    if False:
        return 10
    path = request.get('unsafe')
    open(path)

def bar(request):
    if False:
        while True:
            i = 10
    foo = request.get('unsafe')
    path = 'safe_path'
    open(path)
z = request.get('unsafe')

def baz():
    if False:
        return 10
    open(z)