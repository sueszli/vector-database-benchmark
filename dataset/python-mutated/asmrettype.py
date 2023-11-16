@micropython.asm_thumb
def ret_obj(r0) -> object:
    if False:
        return 10
    pass
ret_obj(print)(1)

@micropython.asm_thumb
def ret_bool(r0) -> bool:
    if False:
        for i in range(10):
            print('nop')
    pass
print(ret_bool(0), ret_bool(1))

@micropython.asm_thumb
def ret_int(r0) -> int:
    if False:
        return 10
    lsl(r0, r0, 29)
print(ret_int(0), hex(ret_int(1)), hex(ret_int(2)), hex(ret_int(4)))

@micropython.asm_thumb
def ret_uint(r0) -> uint:
    if False:
        for i in range(10):
            print('nop')
    lsl(r0, r0, 29)
print(ret_uint(0), hex(ret_uint(1)), hex(ret_uint(2)), hex(ret_uint(4)))