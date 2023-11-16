#include "global.h"
#include "battle.h"
#include "battle_anim.h"
#include "rom_8077ABC.h"

extern s16 gBattleAnimArgs[8];
extern u8 gBattleAnimAttacker;
extern u8 gBattleAnimTarget;
extern u16 gBattleTypeFlags;

void sub_80D3014(struct Sprite *sprite);

// guard (moves guard rings upwards)
// Used in Safeguard.

const union AffineAnimCmd gSpriteAffineAnim_83D7D4C[] =
{
    AFFINEANIMCMD_FRAME(0x100, 0x100, 0, 0),
    AFFINEANIMCMD_END,
};

const union AffineAnimCmd gSpriteAffineAnim_83D7D5C[] =
{
    AFFINEANIMCMD_FRAME(0x200, 0x100, 0, 0),
    AFFINEANIMCMD_END,
};

const union AffineAnimCmd *const gSpriteAffineAnimTable_83D7D6C[] =
{
    gSpriteAffineAnim_83D7D4C,
    gSpriteAffineAnim_83D7D5C,
};

const struct SpriteTemplate gBattleAnimSpriteTemplate_83D7D74 =
{
    .tileTag = ANIM_TAG_GUARD_RING,
    .paletteTag = ANIM_TAG_GUARD_RING,
    .oam = &gOamData_837E13C,
    .anims = gDummySpriteAnimTable,
    .images = NULL,
    .affineAnims = gSpriteAffineAnimTable_83D7D6C,
    .callback = sub_80D3014,
};

void sub_80D3014(struct Sprite *sprite)
{
    if ((gBattleTypeFlags & BATTLE_TYPE_DOUBLE) && IsAnimBankSpriteVisible(gBattleAnimAttacker ^ 2))
    {
        SetAverageBattlerPositions(gBattleAnimAttacker, 0, &sprite->x, &sprite->y);
        sprite->y += 40;

        StartSpriteAffineAnim(sprite, 1);
    }
    else
    {
        sprite->x = GetBattlerSpriteCoord(gBattleAnimAttacker, 0);
        sprite->y = GetBattlerSpriteCoord(gBattleAnimAttacker, 1) + 40;
    }

    sprite->data[0] = 13;
    sprite->data[2] = sprite->x;
    sprite->data[4] = sprite->y - 72;

    sprite->callback = StartAnimLinearTranslation;
    StoreSpriteCallbackInData(sprite, DestroyAnimSprite);
}
