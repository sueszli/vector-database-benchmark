import micropython

@micropython.asm_thumb
def asm_add(r0, r1):
    if False:
        for i in range(10):
            print('nop')
    add(r0, r0, r1)

@micropython.asm_thumb
def asm_add1(r0) -> object:
    if False:
        return 10
    lsl(r0, r0, 1)
    add(r0, r0, 3)

@micropython.asm_thumb
def asm_cast_bool(r0) -> bool:
    if False:
        print('Hello World!')
    pass

@micropython.asm_thumb
def asm_shift_int(r0) -> int:
    if False:
        for i in range(10):
            print('nop')
    lsl(r0, r0, 29)

@micropython.asm_thumb
def asm_shift_uint(r0) -> uint:
    if False:
        while True:
            i = 10
    lsl(r0, r0, 29)