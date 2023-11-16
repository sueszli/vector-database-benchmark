def f():
    if False:
        while True:
            i = 10
    for x in range(2):
        for y in range(2):
            for z in range(2):
                print(x, y, z)
f()
for i in range(3, -1, -1):
    print(i)
a = -1
for i in range(3, -1, a):
    print(i)