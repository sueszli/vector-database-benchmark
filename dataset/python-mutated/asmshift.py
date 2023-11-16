@micropython.asm_thumb
def lsl1(r0):
    if False:
        i = 10
        return i + 15
    lsl(r0, r0, 1)
print(hex(lsl1(291)))

@micropython.asm_thumb
def lsl23(r0):
    if False:
        i = 10
        return i + 15
    lsl(r0, r0, 23)
print(hex(lsl23(1)))

@micropython.asm_thumb
def lsr1(r0):
    if False:
        print('Hello World!')
    lsr(r0, r0, 1)
print(hex(lsr1(291)))

@micropython.asm_thumb
def lsr31(r0):
    if False:
        for i in range(10):
            print('nop')
    lsr(r0, r0, 31)
print(hex(lsr31(2147483648)))

@micropython.asm_thumb
def asr1(r0):
    if False:
        for i in range(10):
            print('nop')
    asr(r0, r0, 1)
print(hex(asr1(291)))

@micropython.asm_thumb
def asr31(r0):
    if False:
        i = 10
        return i + 15
    asr(r0, r0, 31)
print(hex(asr31(2147483648)))