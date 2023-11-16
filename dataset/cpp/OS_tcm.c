#include "OS_tcm.h"
#include "function_target.h"
#include "code32.h"

asm u32 OS_GetDTCMAddress(void) {
    mrc p15, 0x0, r0, c9, c1, 0x0 //Data TCM Base
    ldr r1, =OSi_TCM_REGION_BASE_MASK
    and r0, r0, r1
    bx lr
}
