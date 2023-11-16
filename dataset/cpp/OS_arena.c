#include "consts.h"
#include "OS_arena.h"
#include "code32.h"

static BOOL OSi_Initialized = FALSE;

extern void SDK_SUBPRIV_ARENA_LO(void);
extern void SDK_WRAM_ARENA_LO(void);
extern void SDK_IRQ_STACKSIZE(void);
extern void SDK_SYS_STACKSIZE(void);

void OS_InitArena(void)
{
    if (OSi_Initialized)
    {
        return;
    }
    OSi_Initialized = TRUE;

    OS_SetArenaHi(OS_ARENA_MAIN_SUBPRIV, OS_GetInitArenaHi(OS_ARENA_MAIN_SUBPRIV));
    OS_SetArenaLo(OS_ARENA_MAIN_SUBPRIV, OS_GetInitArenaLo(OS_ARENA_MAIN_SUBPRIV));

    OS_SetArenaHi(OS_ARENA_WRAM_SUB, OS_GetInitArenaHi(OS_ARENA_WRAM_SUB));
    OS_SetArenaLo(OS_ARENA_WRAM_SUB, OS_GetInitArenaLo(OS_ARENA_WRAM_SUB));

    OS_SetArenaHi(OS_ARENA_WRAM_SUBPRIV, OS_GetInitArenaHi(OS_ARENA_WRAM_SUBPRIV));
    OS_SetArenaLo(OS_ARENA_WRAM_SUBPRIV, OS_GetInitArenaLo(OS_ARENA_WRAM_SUBPRIV));
}

void* OS_GetArenaHi(OSArenaId id)
{
    return OSi_GetArenaInfo().hi[id];
}

void* OS_GetArenaLo(OSArenaId id)
{
    return OSi_GetArenaInfo().lo[id];
}

void* OS_GetInitArenaHi(OSArenaId id)
{
    switch (id) {
        case OS_ARENA_MAIN_SUBPRIV:
            return (void *)HW_MAIN_MEM_SUB_END;
        case OS_ARENA_WRAM_SUB:
            return (void *)HW_WRAM_END;
        case OS_ARENA_WRAM_SUBPRIV:
        {
            u32 irqStackLo = (u32)HW_PRV_WRAM_IRQ_STACK_END - (s32)SDK_IRQ_STACKSIZE;
            u32 sysStackLo = (u32)HW_PRV_WRAM;

            if ((u32)SDK_WRAM_ARENA_LO > (u32)HW_PRV_WRAM)
            {
                sysStackLo = (u32)SDK_WRAM_ARENA_LO;
            }

            if (!(s32)SDK_SYS_STACKSIZE)
            {
                return (void *)sysStackLo;
            }
            else if ((s32)SDK_SYS_STACKSIZE < 0)
            {
                sysStackLo -= (s32)SDK_SYS_STACKSIZE;
            }
            else
            {
                sysStackLo = irqStackLo - (s32)SDK_SYS_STACKSIZE;
            }

            return (void *)sysStackLo;
        }
        default:
            return NULL;
    }
}

void* OS_GetInitArenaLo(OSArenaId id) {
    switch (id) {
        case OS_ARENA_MAIN_SUBPRIV:
            return (void *)((u32)0x027fafcc); //todo SDK_SUBPRIV_ARENA_LO
        case OS_ARENA_WRAM_SUB:
        {
            u32 wramSubLo = ((u32)SDK_WRAM_ARENA_LO);
            if ((u32)HW_WRAM_END < (u32)wramSubLo)
            {
                wramSubLo = (u32)HW_WRAM_END;
            }
            return (void *)wramSubLo;
        }
        case OS_ARENA_WRAM_SUBPRIV:
        {
            u32 privWramLo = HW_PRV_WRAM;
            if ((u32)SDK_WRAM_ARENA_LO > (u32)privWramLo)
            {
                privWramLo = (u32)SDK_WRAM_ARENA_LO;
            }
            return (void *)privWramLo;
        }
        default:
            return NULL;
    }
}

void OS_SetArenaHi(OSArenaId id, void* newHi) {
    OSi_GetArenaInfo().hi[id] = newHi;
}

void OS_SetArenaLo(OSArenaId id, void* newLo) {
    OSi_GetArenaInfo().lo[id] = newLo;
}
