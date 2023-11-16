@micropython.asm_thumb
def getIPSR():
    if False:
        while True:
            i = 10
    mrs(r0, IPSR)

@micropython.asm_thumb
def getBASEPRI():
    if False:
        return 10
    mrs(r0, BASEPRI)
print(getBASEPRI())
print(getIPSR())