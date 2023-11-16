@micropython.asm_thumb
def f(r0, r1, r2):
    if False:
        while True:
            i = 10
    push({r0})
    push({r1, r2})
    pop({r0})
    pop({r1, r2})

@micropython.asm_thumb
def g():
    if False:
        return 10
    b(START)
    label(SUBROUTINE)
    push({lr})
    mov(r0, 7)
    pop({pc})
    label(START)
    bl(SUBROUTINE)
print(f(0, 1, 2))
print(g())