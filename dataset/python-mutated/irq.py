import pyb

def test_irq():
    if False:
        while True:
            i = 10
    i1 = pyb.disable_irq()
    print(i1)
    pyb.enable_irq()
    pyb.delay(10)
    i1 = pyb.disable_irq()
    i2 = pyb.disable_irq()
    print(i1, i2)
    pyb.enable_irq(i2)
    pyb.enable_irq(i1)
    pyb.delay(10)
test_irq()