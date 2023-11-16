@micropython.asm_thumb
def arg0():
    if False:
        i = 10
        return i + 15
    mov(r0, 1)
print(arg0())

@micropython.asm_thumb
def arg1(r0):
    if False:
        return 10
    add(r0, r0, 1)
print(arg1(1))

@micropython.asm_thumb
def arg2(r0, r1):
    if False:
        i = 10
        return i + 15
    add(r0, r0, r1)
print(arg2(1, 2))

@micropython.asm_thumb
def arg3(r0, r1, r2):
    if False:
        for i in range(10):
            print('nop')
    add(r0, r0, r1)
    add(r0, r0, r2)
print(arg3(1, 2, 3))

@micropython.asm_thumb
def arg4(r0, r1, r2, r3):
    if False:
        for i in range(10):
            print('nop')
    add(r0, r0, r1)
    add(r0, r0, r2)
    add(r0, r0, r3)
print(arg4(1, 2, 3, 4))