@micropython.asm_thumb
def f(r0):
    if False:
        return 10
    b(entry)
    label(func1)
    add(r0, 2)
    bx(lr)
    label(func2)
    sub(r0, 1)
    bx(lr)
    label(entry)
    bl(func1)
    bl(func2)
print(f(0))
print(f(1))