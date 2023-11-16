@micropython.asm_thumb
def c1():
    if False:
        return 10
    movwt(r0, 4294967295)
    movwt(r1, 4026531840)
    sub(r0, r0, r1)
print(hex(c1()))