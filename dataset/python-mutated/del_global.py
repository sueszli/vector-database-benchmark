def do_del():
    if False:
        return 10
    global x
    del x
x = 1
print(x)
do_del()
try:
    print(x)
except NameError:
    print('NameError')
try:
    do_del()
except:
    print('NameError')
a = 1
del (a,)
try:
    print(a)
except NameError:
    print('NameError')
a = 2
b = 3
del (a, b)
try:
    print(a)
except NameError:
    print('NameError')
try:
    print(b)
except NameError:
    print('NameError')
a = 1
b = 2
c = 3
del (a, b, c)
try:
    print(a)
except NameError:
    print('NameError')
try:
    print(b)
except NameError:
    print('NameError')
try:
    print(c)
except NameError:
    print('NameError')
a = 1
b = 2
c = 3
del (a, (b, c))