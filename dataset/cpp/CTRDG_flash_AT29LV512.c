#include "CTRDG_flash_AT29LV512.h"
#include "CTRDG_backup.h"
#include "CTRDG_flash.h"
#include "MI_exMemory.h"
#include "OS_interrupt.h"
#include "OS_spinLock.h"
#include "code32.h"

#define CTRDG_BACKUP_COM_ADR1   (CTRDG_AGB_FLASH_ADR+0x00005555)
#define CTRDG_BACKUP_COM_ADR2   (CTRDG_AGB_FLASH_ADR+0x00002aaa)

#define FLASH_LOG_SECTOR_COUNT          16
#define FLASH_SECTOR_LOG_TO_PHYS        32
#define FLASH_SECTOR_LOG_TO_PHYS_SHIFT  5

extern u16 ctrdgi_flash_lock_id;
extern BOOL ctrdgi_backup_irq;

extern void CTRDGi_SetTask(CTRDGTaskInfo * pt, CTRDG_TASK_FUNC task, CTRDG_TASK_FUNC callback);

extern u16 CTRDGi_PollingSR512kCOMMON(u16 phase, u8 *adr, u16 lastData);

static const u16 atMaxTime[] = {
        10, 40, 0, 40
};

const CTRDGiFlashTypePlus AT29LV512_lib = {
        .CTRDGi_WriteAgbFlashSector = CTRDGi_WriteFlash4KBAT,
        .CTRDGi_EraseAgbFlashChip = CTRDGi_EraseFlashChipAT,
        .CTRDGi_EraseAgbFlashSector = CTRDGi_EraseFlash4KBAT,
        .CTRDGi_WriteAgbFlashSectorAsync = CTRDGi_WriteFlash4KBAsyncAT,
        .CTRDGi_EraseAgbFlashChipAsync = CTRDGi_EraseFlashChipAsyncAT,
        .CTRDGi_EraseAgbFlashSectorAsync = CTRDGi_EraseFlash4KBAsyncAT,
        .CTRDGi_PollingSR = CTRDGi_PollingSR512kCOMMON,
        .maxtime = atMaxTime,
        .type = {
                .romSize = 0x00010000,
                .sector = {
                    .size = 0x00001000,
                    .shift = 12,
                    .count = 16,
                    .top = 0
                },
                .agbWait = {MI_CTRDG_RAMCYCLE_18, MI_CTRDG_RAMCYCLE_18}, //read cycle and write cycle
                .makerID = 0x1f,
                .deviceID = 0x3d,
        }
};

const CTRDGiFlashTypePlus AT29LV512_org = {
        .CTRDGi_WriteAgbFlashSector = CTRDGi_WriteFlashSectorAT,
        .CTRDGi_EraseAgbFlashChip = CTRDGi_EraseFlashChipAT,
        .CTRDGi_EraseAgbFlashSector = CTRDGi_EraseFlashSectorAT,
        .CTRDGi_WriteAgbFlashSectorAsync = CTRDGi_WriteFlashSectorAsyncAT,
        .CTRDGi_EraseAgbFlashChipAsync = CTRDGi_EraseFlashChipAsyncAT,
        .CTRDGi_EraseAgbFlashSectorAsync = CTRDGi_EraseFlashSectorAsyncAT,
        .CTRDGi_PollingSR = CTRDGi_PollingSR512kCOMMON,
        .maxtime = atMaxTime,
        .type = {
                .romSize = 0x00010000,
                .sector = {
                    .size = 0x00000080,
                    .shift = 7,
                    .count = 512,
                    .top = 0
                },
                .agbWait = {MI_CTRDG_RAMCYCLE_18, MI_CTRDG_RAMCYCLE_18}, //read cycle and write cycle
                .makerID = 0x1f,
                .deviceID = 0x3d,
        }
};

u32 CTRDGi_EraseFlashChipCoreAT(CTRDGTaskInfo *arg)
{
    MICartridgeRamCycle ram_cycle;
    u32 result;
    (void)arg;

    (void)OS_LockCartridge(ctrdgi_flash_lock_id);

    ram_cycle = MI_GetCartridgeRamCycle();
    MI_SetCartridgeRamCycle(AgbFlash->agbWait[0]);

    ctrdgi_backup_irq = OS_DisableIrq();

    *(vu8 *)CTRDG_BACKUP_COM_ADR1 = 0xaa;
    *(vu8 *)CTRDG_BACKUP_COM_ADR2 = 0x55;
    *(vu8 *)CTRDG_BACKUP_COM_ADR1 = 0x80;
    *(vu8 *)CTRDG_BACKUP_COM_ADR1 = 0xaa;
    *(vu8 *)CTRDG_BACKUP_COM_ADR2 = 0x55;
    *(vu8 *)CTRDG_BACKUP_COM_ADR1 = 0x10;

    (void)OS_RestoreIrq(ctrdgi_backup_irq);

    result = CTRDGi_PollingSR(CTRDG_BACKUP_PHASE_CHIP_ERASE, (u8 *)CTRDG_AGB_FLASH_ADR, 0xff);

    MI_SetCartridgeRamCycle(ram_cycle);

    (void)OS_UnlockCartridge(ctrdgi_flash_lock_id);

    return result;
}

u32 CTRDGi_EraseFlashSectorCoreAT(CTRDGTaskInfo *arg)
{
    u32 i;
    u8 *dst;
    BOOL shlet_ime;
    MICartridgeRamCycle ram_cycle;
    u32 result;
    CTRDGTaskInfo p = *arg;
    u16 p_secNo = p.sec_num;

    dst = (u8 *)(CTRDG_AGB_FLASH_ADR + (p_secNo << AT29LV512_org.type.sector.shift));

    (void)OS_LockCartridge(ctrdgi_flash_lock_id);

    ram_cycle = MI_GetCartridgeRamCycle();
    MI_SetCartridgeRamCycle(AgbFlash->agbWait[0]);

    shlet_ime = OS_DisableIrq();

    *(vu8 *)CTRDG_BACKUP_COM_ADR1 = 0xaa;
    *(vu8 *)CTRDG_BACKUP_COM_ADR2 = 0x55;
    *(vu8 *)CTRDG_BACKUP_COM_ADR1 = 0xa0;

    for (i = AT29LV512_org.type.sector.size; i > 0; i--)
    {
        *dst++ = 0xff;
    }
    dst--;

    (void)OS_RestoreIrq(shlet_ime);

    result = CTRDGi_PollingSR(CTRDG_BACKUP_PHASE_PROGRAM, dst, 0xff);

    if (result)
    {
        result = (u16)((result & 0xff00) | CTRDG_BACKUP_PHASE_SECTOR_ERASE);
    }

    MI_SetCartridgeRamCycle(ram_cycle);

    (void)OS_UnlockCartridge(ctrdgi_flash_lock_id);

    return result;
}

u32 CTRDGi_EraseFlash4KBCoreAT(CTRDGTaskInfo *arg)
{
    u32 result;
    u16 i, p_secNo, retry;
    CTRDGTaskInfo p = *arg;
    u16 l_secNo = p.sec_num;

    if (l_secNo >= FLASH_LOG_SECTOR_COUNT)
    {
        return CTRDG_BACKUP_RESULT_ERROR | CTRDG_BACKUP_PHASE_PARAMETER_CHECK;
    }

    p_secNo = (u16)(l_secNo << FLASH_SECTOR_LOG_TO_PHYS_SHIFT);
    for (i = 0; i < FLASH_SECTOR_LOG_TO_PHYS; i++)
    {
        for (retry = 2; retry > 0; retry--)
        {
            result = CTRDGi_EraseFlashSectorAT(p_secNo);
            if (result == 0)
            {
                break;
            }
        }
        p_secNo++;
        if (result)
        {
            break;
        }
    }

    return result;
}

u32 CTRDGi_WriteFlashSectorCoreAT(CTRDGTaskInfo *arg)
{
    u32 i;
    u8 *dst;
    BOOL shlet_ime;
    MICartridgeRamCycle ram_cycle;
    u32 result;
    CTRDGTaskInfo p = *arg;
    u16 p_secNo = p.sec_num;
    u8 *src = p.data;

    (void)OS_LockCartridge(ctrdgi_flash_lock_id);

    ram_cycle = MI_GetCartridgeRamCycle();
    MI_SetCartridgeRamCycle(AgbFlash->agbWait[0]);

    dst = (u8 *)(CTRDG_AGB_FLASH_ADR + (p_secNo << AT29LV512_org.type.sector.shift));

    shlet_ime = OS_DisableIrq();

    *(vu8 *)CTRDG_BACKUP_COM_ADR1 = 0xaa;
    *(vu8 *)CTRDG_BACKUP_COM_ADR2 = 0x55;
    *(vu8 *)CTRDG_BACKUP_COM_ADR1 = 0xa0;

    for (i = AT29LV512_org.type.sector.size; i > 0; i--)
    {
        *dst++ = *src++;
    }
    dst--;
    src--;

    (void)OS_RestoreIrq(shlet_ime);

    result = CTRDGi_PollingSR(CTRDG_BACKUP_PHASE_PROGRAM, dst, *src);

    MI_SetCartridgeRamCycle(ram_cycle);

    (void)OS_UnlockCartridge(ctrdgi_flash_lock_id);

    return result;
}

u32 CTRDGi_WriteFlash4KBCoreAT(CTRDGTaskInfo *arg)
{
    u32 result;
    u16 p_secNo, retry;
    CTRDGTaskInfo p = *arg;
    u16 l_secNo = p.sec_num;
    u8 *src = p.data;

    if (l_secNo >= FLASH_LOG_SECTOR_COUNT)
    {
        return CTRDG_BACKUP_RESULT_ERROR | CTRDG_BACKUP_PHASE_PARAMETER_CHECK;
    }

    p_secNo = (u16)(l_secNo << FLASH_SECTOR_LOG_TO_PHYS_SHIFT);
    ctrdg_flash_remainder = (u16)(AT29LV512_lib.type.sector.size);
    while (ctrdg_flash_remainder)
    {
        for (retry = 2; retry > 0; retry--)
        {
            result = CTRDGi_WriteFlashSectorAT(p_secNo, src);
            if (result == 0)
            {
                break;
            }
        }
        if (result)
        {
            break;
        }
        ctrdg_flash_remainder -= AT29LV512_org.type.sector.size;
        src += AT29LV512_org.type.sector.size;
        p_secNo++;
    }

    return result;
}

u16 CTRDGi_EraseFlashChipAT(void)
{
    CTRDGTaskInfo p;
    return (u16)CTRDGi_EraseFlashChipCoreAT(&p);
}

u16 CTRDGi_EraseFlash4KBAT(u16 l_secNo)
{
    CTRDGTaskInfo p;
    p.sec_num = l_secNo;
    return (u16)CTRDGi_EraseFlash4KBCoreAT(&p);
}

u16 CTRDGi_WriteFlash4KBAT(u16 l_secNo, u8 *src)
{
    CTRDGTaskInfo p;
    p.sec_num = l_secNo;
    p.data = src;
    return (u16)CTRDGi_WriteFlash4KBCoreAT(&p);
}

void CTRDGi_EraseFlashChipAsyncAT(CTRDG_TASK_FUNC callback)
{
    CTRDGTaskInfo p;

    CTRDGi_SetTask(&p, CTRDGi_EraseFlashChipCoreAT, callback);
}

void CTRDGi_EraseFlash4KBAsyncAT(u16 l_secNo, CTRDG_TASK_FUNC callback)
{
    CTRDGTaskInfo p;

    p.sec_num = l_secNo;
    CTRDGi_SetTask(&p, CTRDGi_EraseFlash4KBCoreAT, callback);
}

void CTRDGi_WriteFlash4KBAsyncAT(u16 l_secNo, u8 *src, CTRDG_TASK_FUNC callback)
{
    CTRDGTaskInfo p;

    p.sec_num = l_secNo;
    p.data = src;
    CTRDGi_SetTask(&p, CTRDGi_WriteFlash4KBCoreAT, callback);
}

u16 CTRDGi_EraseFlashSectorAT(u16 p_secNo)
{
    CTRDGTaskInfo p;
    p.sec_num = p_secNo;
    return (u16)CTRDGi_EraseFlashSectorCoreAT(&p);
}

u16 CTRDGi_WriteFlashSectorAT(u16 p_secNo, u8 *src)
{
    CTRDGTaskInfo p;

    p.sec_num = p_secNo;
    p.data = src;
    return (u16)CTRDGi_WriteFlashSectorCoreAT(&p);
}

void CTRDGi_EraseFlashSectorAsyncAT(u16 p_secNo, CTRDG_TASK_FUNC callback)
{
    CTRDGTaskInfo p;

    p.sec_num = p_secNo;
    CTRDGi_SetTask(&p, CTRDGi_EraseFlashSectorCoreAT, callback);
}

void CTRDGi_WriteFlashSectorAsyncAT(u16 p_secNo, u8 *src, CTRDG_TASK_FUNC callback)
{
    CTRDGTaskInfo p;

    p.sec_num = p_secNo;
    p.data = src;
    CTRDGi_SetTask(&p, CTRDGi_WriteFlashSectorCoreAT, callback);
}
