import micropython

@micropython.viper
def viper_uint() -> uint:
    if False:
        print('Hello World!')
    return uint(-1)
import sys
print(viper_uint() == sys.maxsize << 1 | 1)