#include "global.h"
#include "rom_8077ABC.h"
#include "trig.h"
#include "battle_anim.h"
#include "sound.h"

extern s16 gBattleAnimArgs[];
extern u8 gBattleAnimAttacker;
extern u8 gBattleAnimTarget;

extern void sub_80CB7EC(struct Sprite* sprite, s16 c);
extern bool8 sub_80CB814(struct Sprite* sprite);
extern void sub_80CB8B8(struct Sprite* sprite);
extern const union AnimCmd *const gSpriteAnimTable_83D66B8[];
extern const union AffineAnimCmd *const gSpriteAffineAnimTable_83D6714[];

void sub_80CBAE8(struct Sprite* sprite);
static void sub_80CBB60(struct Sprite* sprite);

// hop_2
// Used in item steal.

const struct SpriteTemplate gBattleAnimSpriteTemplate_83D677C =
{
    .tileTag = ANIM_TAG_ITEM_BAG,
    .paletteTag = ANIM_TAG_ITEM_BAG,
    .oam = &gOamData_837DF94,
    .anims = gSpriteAnimTable_83D66B8,
    .images = NULL,
    .affineAnims = gSpriteAffineAnimTable_83D6714,
    .callback = sub_80CBAE8,
};

void sub_80CBAE8(struct Sprite* sprite)
{
    s16 p1;
    s16 p2;
    sub_8078764(sprite, FALSE);
    p1 = GetBattlerSpriteCoord(gBattleAnimAttacker, 0);
    p2 = GetBattlerSpriteCoord(gBattleAnimAttacker, 1);
    if ((gBattleAnimTarget ^ 2) == gBattleAnimAttacker)
    {
        sprite->data[6] = p1;
        sprite->data[7] = p2 + 10;
        sub_80CB7EC(sprite, 0x3c);
        sprite->data[3] = 1;
    }
    else
    {
        sprite->data[6] = p1;
        sprite->data[7] = p2 + 10;
        sub_80CB7EC(sprite, 0x3c);
        sprite->data[3] = 3;
    }

    sprite->data[4] = 0x3C;
    sprite->callback = sub_80CBB60;
}

static void sub_80CBB60(struct Sprite* sprite)
{
    int zero;
    sprite->data[0] += ((sprite->data[3] * 128) / sprite->data[4]);
    zero = 0;
    if (sprite->data[0] > 0x7F)
    {
        sprite->data[1]++;
        sprite->data[0] = zero;
    }

    sprite->y2 = Sin(sprite->data[0] + 0x80, 30 - sprite->data[1] * 8);
    if (sprite->y2 == 0)
    {
        PlaySE12WithPanning(0x7D, BattleAnimAdjustPanning(SOUND_PAN_TARGET));
    }

    if (sub_80CB814(sprite))
    {
        sprite->y2 = 0;
        sprite->data[0] = 0;
        sprite->callback = sub_80CB8B8;
        PlaySE12WithPanning(0x7D, BattleAnimAdjustPanning(SOUND_PAN_ATTACKER_NEG));
    }
}
