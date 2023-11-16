@micropython.asm_thumb
def clz(r0):
    if False:
        print('Hello World!')
    clz(r0, r0)
print(clz(240))
print(clz(32768))

@micropython.asm_thumb
def rbit(r0):
    if False:
        for i in range(10):
            print('nop')
    rbit(r0, r0)
print(hex(rbit(240)))
print(hex(rbit(32768)))