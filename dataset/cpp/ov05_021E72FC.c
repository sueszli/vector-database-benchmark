#include "global.h"
#include "ov05_021E72FC.h"
#include "constants/sndseq.h"
#include "heap.h"
#include "unk_020051F4.h"

extern void FieldSystem_CreateTask(u32 param0, void *func, UnkStruct021E7358 *param2);
extern u32 PlayerAvatar_GetMapObject(u32 param0);
extern u32 sub_02058720(u32 param0);
extern BOOL sub_02054B30(u8 param0);
extern BOOL sub_02054B3C(u8 param0);
extern BOOL sub_02054B48(u8 param0);
extern BOOL sub_02054B54(u8 param0);
extern UnkStruct021E7358 *TaskManager_GetEnvironment(UnkStruct021E7358 *param0);
extern void sub_02058410(u32 param0, u32 param1);
extern BOOL sub_02057254(u32 param0);
extern u32 sub_0205AFDC(u32 param0, u32 param1);
extern void sub_02057260(u32 param0, u32 param1);
extern void sub_02055304(u32 param0, u32 param1);
extern BOOL sub_02056B74(u32 param0, u32 param1, u32 param2);
extern void sub_02058418(u32 param0, u32 param1);

BOOL ov05_021E72FC(u32 param0, u32 param1)
{
    u8 res = (u8)sub_02058720(PlayerAvatar_GetMapObject(param1));
    u32 r2;
    if (sub_02054B30(res) == TRUE)
    {
        r2 = 3;
        goto label;
    }
    else if (sub_02054B3C(res) == TRUE)
    {
        r2 = 2;
        goto label;
    }
    else if (sub_02054B48(res) == TRUE)
    {
        r2 = 0;
        goto label;
    }
    else if (sub_02054B54(res) == TRUE)
    {
        r2 = 1;
        goto label;
    }
    return FALSE;
label:
    ov05_021E7358(param0, param1, r2);
    return TRUE;
}

void ov05_021E7358(u32 param0, u32 param1, u32 param2)
{
    UnkStruct021E7358 *res = ov05_021E74D4(24);
    res->Unk0C = param0;
    res->Unk10 = param1;
    res->Unk00 = param2;
    PlaySE(SEQ_SE_DP_F209);
    FieldSystem_CreateTask(param0, ov05_021E73B4, res);
}

u32 ov05_021E7388(u32 param0)
{
    switch (param0)
    {
        case 0:
            return 2;
        case 2:
            return 1;
        case 1:
            return 3;
        case 3:
            return 0;
    }
    return 0;
}

BOOL ov05_021E73B4(UnkStruct021E7358 *param0)
{
    UnkStruct021E7358 *strct = TaskManager_GetEnvironment(param0);
    u32 res = PlayerAvatar_GetMapObject(strct->Unk10);
    u8 res2 = (u8)sub_02058720(res);
    switch (strct->Unk08)
    {
        case 0:
            sub_02058410(res, 1 << 8);
            strct->Unk08++;
            break;
        case 1:
            if (!sub_02057254(strct->Unk10))
            {
                break;
            }
            sub_02057260(strct->Unk10, sub_0205AFDC(strct->Unk00, 12));
            sub_02055304(strct->Unk10, strct->Unk00);
            strct->Unk08++;
            strct->Unk04 = 7;
            break;
        case 2:
            if (strct->Unk04 == 2 || strct->Unk04 == 4 || strct->Unk04 == 6)
            {
                strct->Unk00 = ov05_021E7388(strct->Unk00);
                sub_02055304(strct->Unk10, strct->Unk00);
            }
            if (--(strct->Unk04))
            {
                break;
            }
            if (sub_02054B30(res2) == TRUE)
            {
                strct->Unk00 = 3;
            }
            else if (sub_02054B3C(res2) == TRUE)
            {
                strct->Unk00 = 2;
            }
            else if (sub_02054B48(res2) == TRUE)
            {
                strct->Unk00 = 0;
            }
            else if (sub_02054B54(res2) == TRUE)
            {
                strct->Unk00 = 1;
            }
            else
            {
                strct->Unk00 = ov05_021E7388(strct->Unk00);
            }
            if (sub_02056B74(strct->Unk10, res, strct->Unk00) == FALSE)
            {
                strct->Unk08 = 1;
                break;
            }
            sub_02058418(res, 0x80);
            sub_02058418(res, 1 << 8);
            sub_02055304(strct->Unk10, strct->Unk00);
            ov05_021E74F8(strct);
            sub_020054F0(1624, 0);
            return TRUE;
    }
    return FALSE;
}

UnkStruct021E7358 *ov05_021E74D4(u32 param0)
{
    UnkStruct021E7358 *res = (UnkStruct021E7358 *)AllocFromHeapAtEnd(HEAP_ID_4, param0);
    if (res == NULL)
    {
        GF_AssertFail();
    }
    memset((void *)res, 0, param0);
    return res;
}

void ov05_021E74F8(UnkStruct021E7358 *param0)
{
    FreeToHeapExplicit(HEAP_ID_4, param0);
}
