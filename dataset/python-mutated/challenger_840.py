import board
import digitalio
_LDO_PIN = digitalio.DigitalInOut(board.LDO_CONTROL)
_LDO_PIN.direction = digitalio.Direction.OUTPUT
_LDO_PIN.value = True

def ldo_on():
    if False:
        i = 10
        return i + 15
    global _LDO_PIN
    _LDO_PIN.value = True

def ldo_off():
    if False:
        i = 10
        return i + 15
    global _LDO_PIN
    _LDO_PIN.value = False