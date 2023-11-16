#include "global.h"
#include "scrcmd.h"
#include "bag.h"

const u16 gFossilPokemonMap[7][2] = {
    { ITEM_OLD_AMBER,    SPECIES_AERODACTYL },
    { ITEM_HELIX_FOSSIL, SPECIES_OMANYTE },
    { ITEM_DOME_FOSSIL,  SPECIES_KABUTO },
    { ITEM_ROOT_FOSSIL,  SPECIES_LILEEP },
    { ITEM_CLAW_FOSSIL,  SPECIES_ANORITH },
    { ITEM_ARMOR_FOSSIL, SPECIES_SHIELDON },
    { ITEM_SKULL_FOSSIL, SPECIES_CRANIDOS },
};

BOOL ScrCmd_CountFossils(struct ScriptContext * ctx) //01F1
{
    struct FieldSystem *fieldSystem = ctx->fieldSystem;

    u16 *ret_ptr = ScriptGetVarPointer(ctx);

    u8 i;
    u16 total;
    for (i = 0, total = 0; i < 7; i++)
    {
        total += Bag_GetQuantity(Save_Bag_Get(fieldSystem->saveData), gFossilPokemonMap[i][0], HEAP_ID_4);
    }

    *ret_ptr = total;
    return FALSE;
}

BOOL ScrCmd_GetFossilPokemon(struct ScriptContext * ctx) //01F4
{
    u16 *ret_ptr = ScriptGetVarPointer(ctx);
    u16 fossilId = ScriptGetVar(ctx);

    *ret_ptr = 0;

    for (u16 i = 0; i < 7; i++)
    {
        if (gFossilPokemonMap[i][0] == fossilId)
        {
            *ret_ptr = gFossilPokemonMap[i][1];
            break;
        }
    }

    return FALSE;
}

BOOL ScrCmd_GetFossilMinimumAmount(struct ScriptContext * ctx) //01F5
{
    struct FieldSystem *fieldSystem = ctx->fieldSystem;
    u16 * ret_ptr1 = ScriptGetVarPointer(ctx);
    u16 * ret_ptr2 = ScriptGetVarPointer(ctx);
    u16 needed_amount = ScriptGetVar(ctx);

    *ret_ptr1 = 0;
    *ret_ptr2 = 0;

    u8 i = 0;
    u16 total = 0;
    for (; i < 7; i++)
    {
        total += Bag_GetQuantity(Save_Bag_Get(fieldSystem->saveData), gFossilPokemonMap[i][0], HEAP_ID_4);
        if (total >= needed_amount)
        {
            *ret_ptr1 = gFossilPokemonMap[i][0];
            *ret_ptr2 = i;
            break;
        }
    }

    return FALSE;
}

BOOL ScrCmd_Unk01F2(struct ScriptContext * ctx) //01F2
{
#pragma unused(ctx)
    return FALSE;
}

BOOL ScrCmd_Unk01F3(struct ScriptContext * ctx) //01F3
{
#pragma unused(ctx)
    return FALSE;
}
