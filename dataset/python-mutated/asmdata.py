@micropython.asm_thumb
def ret_num(r0) -> uint:
    if False:
        return 10
    lsl(r0, r0, 2)
    mov(r1, pc)
    add(r0, r0, r1)
    ldr(r0, [r0, 4])
    b(HERE)
    data(4, 305419896, 536870912, 1073741824, 2147483647 + 1, (1 << 32) - 2)
    label(HERE)
for i in range(5):
    print(hex(ret_num(i)))