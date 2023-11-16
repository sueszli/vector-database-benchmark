"""
Topic: 在局部变量域中执行代码
Desc : 
"""

def test3():
    if False:
        while True:
            i = 10
    x = 0
    loc = locals()
    print(loc)
    exec('x += 1')
    print(loc)
    locals()
    print(loc)

def test4():
    if False:
        for i in range(10):
            print('nop')
    a = 13
    loc = {'a': a}
    glb = {}
    exec('b = a + 1', glb, loc)
    b = loc['b']
    print(b)