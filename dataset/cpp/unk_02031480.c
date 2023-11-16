#include "global.h"
#include "unk_02031480.h"

struct UnkStruct_02031480
{
    u8 unk000[8][2];
    u8 unk010[8];
    u8 unk018[8][70];
    u8 unk248[8];
    u8 unk250;
    u8 unk251;
    u8 unk252;
};

static struct UnkStruct_02031480 *UNK_021C59FC;

void sub_02031480(HeapID heapId)
{
    if (UNK_021C59FC == NULL)
    {
        UNK_021C59FC = AllocFromHeap(heapId, sizeof(struct UnkStruct_02031480));
        MI_CpuFill8(UNK_021C59FC, 0, sizeof(struct UnkStruct_02031480));
    }

    for (int i = 0; i < 8; i++)
    {
        UNK_021C59FC->unk010[i] = 0xff;
    }
    UNK_021C59FC->unk250 = 0xff;
    UNK_021C59FC->unk251 = 0xff;
    UNK_021C59FC->unk252 = 0x00;
}

void sub_020314D0()
{
    FreeToHeap(UNK_021C59FC);
    UNK_021C59FC = NULL;
}

BOOL sub_020314E8()
{
    if (UNK_021C59FC != NULL)
    {
        return TRUE;
    }

    return FALSE;
}

void sub_020314FC(u8 param0, u32 unused, u8 *param2)
{
#pragma unused(unused)
    u8 st0[3];

    st0[0] = param2[0];

    if (sub_02031190() == 0)
    {

        st0[1] = param0;
        st0[2] = st0[0];

        sub_02030C4C(0x12, &st0[1]);

        UNK_021C59FC->unk010[param0] = st0[0];

        for (int i = 0; i < 8; i++)
        {
            if (sub_02030E7C((u16)i) != 0)
            {
                if (st0[0] != UNK_021C59FC->unk010[i])
                {
                    return;
                }
            }
        }

        sub_02030C4C(0x11, &st0[0]);
    }
}

void sub_02031560(u32 unused1, u32 unused2, u8 *param2)
{
#pragma unused(unused1)
#pragma unused(unused2)
    UNK_021C59FC->unk010[param2[0]] = param2[1];
}

void sub_02031574(u32 unused1, u32 unused2, u8 *param2)
{
#pragma unused(unused1)
#pragma unused(unused2)
    UNK_021C59FC->unk250 = *param2;
}

void sub_02031588(u8 param0)
{
    UNK_021C59FC->unk251 = param0;
    UNK_021C59FC->unk252 = 1;
}

void sub_020315A4()
{
    if (UNK_021C59FC != NULL && UNK_021C59FC->unk252 != 0 &&
        sub_020311D0(0x10, &UNK_021C59FC->unk251))
    {
        UNK_021C59FC->unk252 = 0;
    }
}

BOOL sub_020315D8(u8 param0)
{
    if (UNK_021C59FC == NULL)
    {
        return TRUE;
    }

    if (UNK_021C59FC->unk250 == param0)
    {
        return TRUE;
    }

    return FALSE;
}

u8 sub_020315FC(u8 index)
{
    return UNK_021C59FC->unk010[index];
}

void sub_0203160C(u32 param0, u32 unused, u8 *param2)
{
#pragma unused(unused)
    UNK_021C59FC->unk000[param0][0] = param2[0];
    UNK_021C59FC->unk000[param0][1] = param2[1];
}

u32 sub_02031628()
{
    return 2;
}

void sub_0203162C(u8 param0, u8 param1)
{
    u8 st0[2] = { param0, param1 };

    sub_020311D0(0x13, st0);
}

int sub_02031640(u32 param0, u8 param1)
{
    if (UNK_021C59FC == NULL)
    {
        return -1;
    }

    if (UNK_021C59FC->unk000[param0][0] == param1)
    {
        return UNK_021C59FC->unk000[param0][1];
    }

    return -1;
}

void sub_02031668()
{
    for (int i = 0; i < 8; i++)
    {
        MI_CpuFill8(UNK_021C59FC->unk000[i], 0, 2);
    }
}

void sub_0203168C()
{
    for (int i = 0; i < 8; i++)
    {
        UNK_021C59FC->unk248[i] = 0;
    }
}

BOOL sub_020316AC(u32 param0, void *param1)
{
    if (UNK_021C59FC != NULL)
    {
        MI_CpuCopy8(param1, UNK_021C59FC->unk018[param0], 0x46);
        sub_020311D0(0x14, UNK_021C59FC->unk018[param0]);

        return TRUE;
    }

    return FALSE;
}

u8 *sub_020316E0(int param0)
{
    if (UNK_021C59FC->unk248[param0] != 0)
    {
        return UNK_021C59FC->unk018[param0];
    }

    return NULL;
}

void sub_02031704(u32 param0, u32 unused, void *param2)
{
#pragma unused(unused)
    UNK_021C59FC->unk248[param0] = 1;
    MI_CpuCopy8(param2, UNK_021C59FC->unk018[param0], 0x46);
}

u32 sub_02031730()
{
    return 0x46;
}
