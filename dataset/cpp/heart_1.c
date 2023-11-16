#include "global.h"
#include "rom_8077ABC.h"
#include "trig.h"
#include "battle_anim.h"
#include "sound.h"

extern s16 gBattleAnimArgs[];
extern u8 gBattleAnimAttacker;
extern u8 gBattleAnimTarget;

void sub_80D1C80(struct Sprite* sprite);

// heart_1 (a floating heart in a wave pattern upward.)
// Used in Charm, Covet, and when a Pokemon is infatuated.

const struct SpriteTemplate gBattleAnimSpriteTemplate_83D7A80 =
{
    .tileTag = ANIM_TAG_MAGENTA_HEART,
    .paletteTag = ANIM_TAG_MAGENTA_HEART,
    .oam = &gOamData_837DF2C,
    .anims = gDummySpriteAnimTable,
    .images = NULL,
    .affineAnims = gDummySpriteAffineAnimTable,
    .callback = sub_80D1C80,
};

void sub_80D1C80(struct Sprite* sprite)
{
    if (++sprite->data[0] == 1)
        InitAnimSpritePos(sprite, 0);

    sprite->x2 = Sin(sprite->data[1], 8);
    sprite->y2 = sprite->data[2] >> 8;
    sprite->data[1] = (sprite->data[1] + 7) & 0xFF;
    sprite->data[2] -= 0x80;
    if (sprite->data[0] == 0x3C)
        DestroyAnimSprite(sprite);
}
