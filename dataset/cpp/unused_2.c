#include "global.h"
#include "rom_8077ABC.h"
#include "trig.h"
#include "battle_anim.h"
#include "sound.h"

extern s16 gBattleAnimArgs[];
extern u8 gBattleAnimAttacker;
extern u8 gBattleAnimTarget;

void sub_80CCC50(struct Sprite* sprite);
static void sub_80CCCB4(struct Sprite* sprite);

// unused_2 (unknown effect with music notes.)
// possibly another unused effect. Unknown usage.

const union AnimCmd gSpriteAnim_83D6B58[] =
{
    ANIMCMD_FRAME(0, 1),
    ANIMCMD_END,
};

const union AnimCmd gSpriteAnim_83D6B60[] =
{
    ANIMCMD_FRAME(4, 1),
    ANIMCMD_END,
};

const union AnimCmd gSpriteAnim_83D6B68[] =
{
    ANIMCMD_FRAME(8, 1),
    ANIMCMD_END,
};

const union AnimCmd gSpriteAnim_83D6B70[] =
{
    ANIMCMD_FRAME(12, 1),
    ANIMCMD_END,
};

const union AnimCmd gSpriteAnim_83D6B78[] =
{
    ANIMCMD_FRAME(16, 1),
    ANIMCMD_END,
};

const union AnimCmd gSpriteAnim_83D6B80[] =
{
    ANIMCMD_FRAME(20, 1),
    ANIMCMD_END,
};

const union AnimCmd gSpriteAnim_83D6B88[] =
{
    ANIMCMD_FRAME(0, 1, .vFlip = TRUE),
    ANIMCMD_END,
};

const union AnimCmd gSpriteAnim_83D6B90[] =
{
    ANIMCMD_FRAME(4, 1, .vFlip = TRUE),
    ANIMCMD_END,
};

const union AnimCmd gSpriteAnim_83D6B98[] =
{
    ANIMCMD_FRAME(8, 1, .vFlip = TRUE),
    ANIMCMD_END,
};

const union AnimCmd gSpriteAnim_83D6BA0[] =
{
    ANIMCMD_FRAME(12, 1, .vFlip = TRUE),
    ANIMCMD_END,
};

const union AnimCmd *const gSpriteAnimTable_83D6BA8[] =
{
    gSpriteAnim_83D6B58,
    gSpriteAnim_83D6B60,
    gSpriteAnim_83D6B68,
    gSpriteAnim_83D6B70,
    gSpriteAnim_83D6B78,
    gSpriteAnim_83D6B80,
    gSpriteAnim_83D6B88,
    gSpriteAnim_83D6B90,
    gSpriteAnim_83D6B98,
    gSpriteAnim_83D6BA0,
};

const struct SpriteTemplate gSpriteTemplate_83D6BD0 =
{
    .tileTag = ANIM_TAG_MUSIC_NOTES,
    .paletteTag = ANIM_TAG_MUSIC_NOTES,
    .oam = &gOamData_837DF2C,
    .anims = gSpriteAnimTable_83D6BA8,
    .images = NULL,
    .affineAnims = gDummySpriteAffineAnimTable,
    .callback = sub_80CCC50,
};

void unref_sub_80CCB6C(struct Sprite* sprite)
{
    if (sprite->data[2] > 1)
    {
        if (sprite->data[3] & 1)
        {
            sprite->invisible = FALSE;
            gSprites[sprite->data[0]].invisible = FALSE;
            gSprites[sprite->data[1]].invisible = FALSE;
        }
        else
        {
            sprite->invisible = TRUE;
            gSprites[sprite->data[0]].invisible = TRUE;
            gSprites[sprite->data[1]].invisible = TRUE;
        }

        sprite->data[2] = 0;
        sprite->data[3]++;
    }
    else
    {
        sprite->data[2]++;
    }

    if (sprite->data[3] == 10)
    {
        DestroySprite(&gSprites[sprite->data[0]]);
        DestroySprite(&gSprites[sprite->data[1]]);
        DestroyAnimSprite(sprite);
    }
}

void sub_80CCC50(struct Sprite* sprite)
{
    sprite->data[0] = gBattleAnimArgs[2];
    if (GetBattlerSide(gBattleAnimAttacker) != 0)
        sprite->x -= gBattleAnimArgs[0];
    else
        sprite->x += gBattleAnimArgs[0];

    StartSpriteAnim(sprite, gBattleAnimArgs[5]);
    sprite->data[1] = -gBattleAnimArgs[3];
    sprite->y += gBattleAnimArgs[1];
    sprite->data[3] = gBattleAnimArgs[4];
    sprite->callback = sub_80CCCB4;
    sub_80CCCB4(sprite);
}

static void sub_80CCCB4(struct Sprite* sprite)
{
    sprite->x2 = Cos(sprite->data[0], 100);
    sprite->y2 = Sin(sprite->data[0], 20);
    if (sprite->data[0] <= 0x7F)
        sprite->subpriority = 0;
    else
        sprite->subpriority = 14;

    sprite->data[0] = (sprite->data[0] + sprite->data[1]) & 0xFF;
    sprite->data[5] += 0x82;
    sprite->y2 += sprite->data[5] >> 8;
    sprite->data[2]++;
    if (sprite->data[2] == sprite->data[3])
        DestroyAnimSprite(sprite);
}
