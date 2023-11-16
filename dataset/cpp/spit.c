#include "global.h"
#include "rom_8077ABC.h"
#include "trig.h"
#include "battle_anim.h"
#include "sound.h"

extern s16 gBattleAnimArgs[];
extern u8 gBattleAnimAttacker;
extern u8 gBattleAnimTarget;

void sub_80D28AC(struct Sprite* sprite);

// spit (hurls sprites outward from the pokemon. Similar to orbit_fast, but takes another argument.)
// Used in Spit Up.

const union AffineAnimCmd gSpriteAffineAnim_83D7B44[] =
{
    AFFINEANIMCMD_FRAME(0x80, 0x80, 0, 0),
    AFFINEANIMCMD_FRAME(0x8, 0x8, 0, 1),
    AFFINEANIMCMD_JUMP(1),
};

const union AffineAnimCmd *const gSpriteAffineAnimTable_83D7B5C[] =
{
    gSpriteAffineAnim_83D7B44,
};

const struct SpriteTemplate gBattleAnimSpriteTemplate_83D7B60 =
{
    .tileTag = ANIM_TAG_RED_ORB_2,
    .paletteTag = ANIM_TAG_RED_ORB_2,
    .oam = &gOamData_837DFE4,
    .anims = gDummySpriteAnimTable,
    .images = NULL,
    .affineAnims = gSpriteAffineAnimTable_83D7B5C,
    .callback = sub_80D28AC,
};

static void sub_80D287C(struct Sprite* sprite)
{
    sprite->x2 += sprite->data[0];
    sprite->y2 += sprite->data[1];
    if (sprite->data[3]++ >= sprite->data[2])
        DestroyAnimSprite(sprite);
}

void sub_80D28AC(struct Sprite* sprite)
{
    sprite->x = GetBattlerSpriteCoord(gBattleAnimAttacker, 2);
    sprite->y = GetBattlerSpriteCoord(gBattleAnimAttacker, 3);
    sprite->data[0] = Sin(gBattleAnimArgs[0], 10);
    sprite->data[1] = Cos(gBattleAnimArgs[0], 7);
    sprite->data[2] = gBattleAnimArgs[1];
    sprite->callback = sub_80D287C;
}
