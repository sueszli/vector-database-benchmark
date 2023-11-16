#include "global.h"
#include "rom_8077ABC.h"
#include "trig.h"
#include "battle_anim.h"
#include "sound.h"

extern s16 gBattleAnimArgs[];
extern u8 gBattleAnimAttacker;
extern u8 gBattleAnimTarget;

void sub_80CE36C(struct Sprite* sprite);
static void sub_80CE3B0(struct Sprite* sprite);

// twinkle (a tiny twinkling star appears above the Pokemon and descends toward the Pokemon.)
// Used in Moonlight.

const union AnimCmd gSpriteAnim_83D6FE0[] =
{
    ANIMCMD_FRAME(0, 8),
    ANIMCMD_FRAME(4, 8),
    ANIMCMD_FRAME(8, 8),
    ANIMCMD_FRAME(12, 8),
    ANIMCMD_JUMP(0),
};

const union AnimCmd *const gSpriteAnimTable_83D6FF4[] =
{
    gSpriteAnim_83D6FE0,
};

const struct SpriteTemplate gBattleAnimSpriteTemplate_83D6FF8 =
{
    .tileTag = ANIM_TAG_SPARKLE_5,
    .paletteTag = ANIM_TAG_SPARKLE_5,
    .oam = &gOamData_837DF2C,
    .anims = gSpriteAnimTable_83D6FF4,
    .images = NULL,
    .affineAnims = gDummySpriteAffineAnimTable,
    .callback = sub_80CE36C,
};

void sub_80CE36C(struct Sprite* sprite)
{
    sprite->x = GetBattlerSpriteCoord(gBattleAnimAttacker, 2) + gBattleAnimArgs[0];
    sprite->y = gBattleAnimArgs[1];
    sprite->data[0] = 0;
    sprite->data[1] = 0;
    sprite->data[2] = 0;
    sprite->data[3] = 0;
    sprite->data[4] = 1;
    sprite->callback = sub_80CE3B0;
}

static void sub_80CE3B0(struct Sprite* sprite)
{
    if (++sprite->data[1] > 1)
    {
        sprite->data[1] = 0;
        if (sprite->data[2] <= 0x77)
        {
            sprite->y++;
            sprite->data[2]++;
        }
    }

    if (sprite->data[0])
        DestroyAnimSprite(sprite);
}
