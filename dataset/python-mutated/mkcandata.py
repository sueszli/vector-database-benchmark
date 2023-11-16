def defines(name, suffix):
    if False:
        while True:
            i = 10
    print(f'mcu_pin_function_t {name} [] = {{')
    for instance in (0, 1):
        for function in 'HI':
            for port in 'ABCD':
                for idx in range(32):
                    pin = f'P{port}{idx:02d}'
                    pinmux = f'PINMUX_{pin}{function}_CAN{instance}_{suffix}'
                    print(f'#if defined({pinmux}) && ! defined(IGNORE_PIN_{pin})\n    {{&pin_{pin}, {instance}, PIN_{pin}, {pinmux} & 0xffff}},\n#endif')
    print(f'{{NULL, 0, 0}}')
    print(f'}};')
    print()
print('#include <stdint.h>\n#include "py/obj.h"\n#include "sam.h"\n#include "samd/pins.h"\n#include "mpconfigport.h"\n#include "atmel_start_pins.h"\n#include "hal/include/hal_gpio.h"\n#include "common-hal/microcontroller/Pin.h"\n')
defines('can_rx', 'RX')
defines('can_tx', 'TX')