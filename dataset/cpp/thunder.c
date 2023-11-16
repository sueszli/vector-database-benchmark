#include "global.h"
#include "battle_anim.h"
#include "rom_8077ABC.h"

extern s16 gBattleAnimArgs[8];
extern u8 gBattleAnimAttacker;
extern u8 gBattleAnimTarget;
extern u16 gBattleTypeFlags;

void sub_80D61C8(struct Sprite *sprite);
static void sub_80D6218(struct Sprite *);

// thunder (positions the lightning bolts)
// Used in Thunder, Thunder Punch, and Tri Attack.

const union AnimCmd gSpriteAnim_83D97B4[] =
{
    ANIMCMD_FRAME(0, 5),
    ANIMCMD_FRAME(16, 5),
    ANIMCMD_FRAME(32, 8),
    ANIMCMD_FRAME(48, 5),
    ANIMCMD_FRAME(64, 5),
    ANIMCMD_END,
};

const union AnimCmd *const gSpriteAnimTable_83D97CC[] =
{
    gSpriteAnim_83D97B4,
};

const struct SpriteTemplate gBattleAnimSpriteTemplate_83D97D0 =
{
    .tileTag = ANIM_TAG_LIGHTNING,
    .paletteTag = ANIM_TAG_LIGHTNING,
    .oam = &gOamData_837DF34,
    .anims = gSpriteAnimTable_83D97CC,
    .images = NULL,
    .affineAnims = gDummySpriteAffineAnimTable,
    .callback = sub_80D61C8,
};

void sub_80D61C8(struct Sprite *sprite)
{
    if (GetBattlerSide(gBattleAnimAttacker) != 0)
    {
        sprite->x -= gBattleAnimArgs[0];
    }
    else
    {
        sprite->x += gBattleAnimArgs[0];
    }

    sprite->y += gBattleAnimArgs[1];
    sprite->callback = sub_80D6218;
}

static void sub_80D6218(struct Sprite *sprite)
{
    if (sprite->animEnded)
    {
        DestroyAnimSprite(sprite);
    }
}
