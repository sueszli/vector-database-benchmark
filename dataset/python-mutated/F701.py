for i in range(10):
    break
else:
    break
i = 0
while i < 10:
    i += 1
    break

def f():
    if False:
        for i in range(10):
            print('nop')
    for i in range(10):
        break
    break

class Foo:
    break
break