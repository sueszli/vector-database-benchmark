def defines(name, function):
    if False:
        print('Hello World!')
    print(f'mcu_pin_function_t {name} [] = {{')
    for instance in (0, 1):
        for port in 'ABCD':
            for idx in range(32):
                pin = f'P{port}{idx:02d}'
                pinmux = f'PINMUX_{pin}I_SDHC{instance}_{function}'
                print(f'#if defined({pinmux}) && ! defined(IGNORE_PIN_{pin})\n    {{&pin_{pin}, {instance}, PIN_{pin}, {pinmux} & 0xffff}},\n#endif')
    print(f'{{NULL, 0, 0}}')
    print(f'}};')
    print()
print('#include <stdint.h>\n#include "py/obj.h"\n#include "sam.h"\n#include "samd/pins.h"\n#include "mpconfigport.h"\n#include "atmel_start_pins.h"\n#include "hal/include/hal_gpio.h"\n#include "common-hal/microcontroller/Pin.h"\n\n')
defines('sdio_ck', 'SDCK')
defines('sdio_cmd', 'SDCMD')
defines('sdio_dat0', 'SDDAT0')
defines('sdio_dat1', 'SDDAT1')
defines('sdio_dat2', 'SDDAT2')
defines('sdio_dat3', 'SDDAT3')