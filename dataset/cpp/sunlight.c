#include "global.h"
#include "battle_anim.h"
#include "rom_8077ABC.h"

void sub_80D517C(struct Sprite *sprite);

// sunlight (creates sunlight orbs)
// Used in Sunny Day

const union AffineAnimCmd gSpriteAffineAnim_83D9594[] =
{
    AFFINEANIMCMD_FRAME(0x50, 0x50, 0, 0),
    AFFINEANIMCMD_FRAME(0x2, 0x2, 10, 1),
    AFFINEANIMCMD_JUMP(1),
};

const union AffineAnimCmd *const gSpriteAffineAnimTable_83D95AC[] =
{
    gSpriteAffineAnim_83D9594,
};

const struct SpriteTemplate gBattleAnimSpriteTemplate_83D95B0 =
{
    .tileTag = ANIM_TAG_SUNLIGHT,
    .paletteTag = ANIM_TAG_SUNLIGHT,
    .oam = &gOamData_837E0B4,
    .anims = gDummySpriteAnimTable,
    .images = NULL,
    .affineAnims = gSpriteAffineAnimTable_83D95AC,
    .callback = sub_80D517C,
};

void sub_80D517C(struct Sprite *sprite)
{
    sprite->x = 0;
    sprite->y = 0;
    sprite->data[0] = 60;
    sprite->data[2] = 140;
    sprite->data[4] = 80;
    sprite->callback = StartAnimLinearTranslation;
    StoreSpriteCallbackInData(sprite, DestroyAnimSprite);
}
