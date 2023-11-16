#include "global.h"
#include "rom_8077ABC.h"
#include "trig.h"
#include "battle_anim.h"

extern s16 gBattleAnimArgs[];
extern u8 gBattleAnimAttacker;
extern u8 gBattleAnimTarget;

BSS_DATA u32 filler_03000724;
BSS_DATA u16 gUnknown_03000728[4];
BSS_DATA u16 gUnknown_03000730[6];
BSS_DATA u32 filler_0300073c;

void sub_80CB59C(struct Sprite* sprite);
void sub_80CB620(struct Sprite *sprite);
static void sub_80CB710(struct Sprite* sprite);

// roots
// Used by Ingrain and Frenzy Plant.

const union AnimCmd gSpriteAnim_83D6600[] =
{
    ANIMCMD_FRAME(0, 7),
    ANIMCMD_FRAME(16, 7),
    ANIMCMD_FRAME(32, 7),
    ANIMCMD_FRAME(48, 7),
    ANIMCMD_END,
};

const union AnimCmd gSpriteAnim_83D6614[] =
{
    ANIMCMD_FRAME(0, 7, .hFlip = TRUE),
    ANIMCMD_FRAME(16, 7, .hFlip = TRUE),
    ANIMCMD_FRAME(32, 7, .hFlip = TRUE),
    ANIMCMD_FRAME(48, 7, .hFlip = TRUE),
    ANIMCMD_END,
};

const union AnimCmd gSpriteAnim_83D6628[] =
{
    ANIMCMD_FRAME(0, 7),
    ANIMCMD_FRAME(16, 7),
    ANIMCMD_FRAME(32, 7),
    ANIMCMD_END,
};

const union AnimCmd gSpriteAnim_83D6638[] =
{
    ANIMCMD_FRAME(0, 7, .hFlip = TRUE),
    ANIMCMD_FRAME(16, 7, .hFlip = TRUE),
    ANIMCMD_FRAME(32, 7, .hFlip = TRUE),
    ANIMCMD_END,
};

const union AnimCmd *const gSpriteAnimTable_83D6648[] =
{
    gSpriteAnim_83D6600,
    gSpriteAnim_83D6614,
    gSpriteAnim_83D6628,
    gSpriteAnim_83D6638,
};

const struct SpriteTemplate gBattleAnimSpriteTemplate_83D6658 =
{
    .tileTag = ANIM_TAG_ROOTS,
    .paletteTag = ANIM_TAG_ROOTS,
    .oam = &gOamData_837DF34,
    .anims = gSpriteAnimTable_83D6648,
    .images = NULL,
    .affineAnims = gDummySpriteAffineAnimTable,
    .callback = sub_80CB59C,
};

const struct SpriteTemplate gBattleAnimSpriteTemplate_83D6670 =
{
    .tileTag = ANIM_TAG_ROOTS,
    .paletteTag = ANIM_TAG_ROOTS,
    .oam = &gOamData_837DF34,
    .anims = gSpriteAnimTable_83D6648,
    .images = NULL,
    .affineAnims = gDummySpriteAffineAnimTable,
    .callback = sub_80CB620,
};

void sub_80CB59C(struct Sprite* sprite)
{
    if (!sprite->data[0])
    {
        sprite->x = GetBattlerSpriteCoord(gBattleAnimAttacker, 2);
        sprite->y = GetBattlerSpriteCoord(gBattleAnimAttacker, 1);
        sprite->x2 = gBattleAnimArgs[0];
        sprite->y2 = gBattleAnimArgs[1];
        sprite->subpriority = gBattleAnimArgs[2] + 30;
        StartSpriteAnim(sprite, gBattleAnimArgs[3]);
        sprite->data[2] = gBattleAnimArgs[4];
        sprite->data[0]++;
        if ((sprite->y + sprite->y2) > 120)
        {
            sprite->y += -120 + (sprite->y2 + sprite->y);
        }
    }
    sprite->callback = sub_80CB710;
}

void sub_80CB620(struct Sprite *sprite)
{
    s16 p1 = GetBattlerSpriteCoord(gBattleAnimAttacker, 2);
    s16 p2 = GetBattlerSpriteCoord(gBattleAnimAttacker, 3);
    s16 e1 = GetBattlerSpriteCoord(gBattleAnimTarget, 2);
    s16 e2 = GetBattlerSpriteCoord(gBattleAnimTarget, 3);

    e1 -= p1;
    e2 -= p2;
    sprite->x = p1 + e1 * gBattleAnimArgs[0] / 100;
    sprite->y = p2 + e2 * gBattleAnimArgs[0] / 100;
    sprite->x2 = gBattleAnimArgs[1];
    sprite->y2 = gBattleAnimArgs[2];
    sprite->subpriority = gBattleAnimArgs[3] + 30;
    StartSpriteAnim(sprite, gBattleAnimArgs[4]);
    sprite->data[2] = gBattleAnimArgs[5];
    sprite->callback = sub_80CB710;
    gUnknown_03000728[0] = sprite->x;
    gUnknown_03000728[1] = sprite->y;
    gUnknown_03000728[2] = e1;
    gUnknown_03000728[3] = e2;
}

static void sub_80CB710(struct Sprite* sprite)
{
    if (++sprite->data[0] > (sprite->data[2] - 10))
        sprite->invisible = sprite->data[0] % 2;

    if (sprite->data[0] > sprite->data[2])
        DestroyAnimSprite(sprite);
}
