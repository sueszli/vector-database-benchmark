#include "global.h"
#include "rom_8077ABC.h"
#include "trig.h"
#include "battle_anim.h"

extern s16 gBattleAnimArgs[];
extern u8 gBattleAnimAttacker;

void AnimMovePowderParticle(struct Sprite* sprite);
static void AnimMovePowderParticleStep(struct Sprite* sprite);

const union AnimCmd gSpriteAnim_83D61FC[] =
{
    ANIMCMD_FRAME(0, 5),
    ANIMCMD_FRAME(2, 5),
    ANIMCMD_FRAME(4, 5),
    ANIMCMD_FRAME(6, 5),
    ANIMCMD_FRAME(8, 5),
    ANIMCMD_FRAME(10, 5),
    ANIMCMD_FRAME(12, 5),
    ANIMCMD_FRAME(14, 5),
    ANIMCMD_JUMP(0),
};

const union AnimCmd *const gSpriteAnimTable_83D6220[] =
{
    gSpriteAnim_83D61FC,
};

const struct SpriteTemplate gSleepPowderParticleSpriteTemplate =
{
    .tileTag = ANIM_TAG_SLEEP_POWDER,
    .paletteTag = ANIM_TAG_SLEEP_POWDER,
    .oam = &gOamData_837DF64,
    .anims = gSpriteAnimTable_83D6220,
    .images = NULL,
    .affineAnims = gDummySpriteAffineAnimTable,
    .callback = AnimMovePowderParticle,
};

const struct SpriteTemplate gStunSporeParticleSpriteTemplate =
{
    .tileTag = ANIM_TAG_STUN_SPORE,
    .paletteTag = ANIM_TAG_STUN_SPORE,
    .oam = &gOamData_837DF64,
    .anims = gSpriteAnimTable_83D6220,
    .images = NULL,
    .affineAnims = gDummySpriteAffineAnimTable,
    .callback = AnimMovePowderParticle,
};

const struct SpriteTemplate gPoisonPowderParticleSpriteTemplate =
{
    .tileTag = ANIM_TAG_POISON_POWDER,
    .paletteTag = ANIM_TAG_POISON_POWDER,
    .oam = &gOamData_837DF64,
    .anims = gSpriteAnimTable_83D6220,
    .images = NULL,
    .affineAnims = gDummySpriteAffineAnimTable,
    .callback = AnimMovePowderParticle,
};

// Animates the falling particles that horizontally wave back and forth.
// Used by Sleep Powder, Stun Spore, and Poison Powder.
// arg 0: initial x pixel offset
// arg 1: initial y pixel offset
// arg 2: total duration in frames
// arg 3: vertical movement speed (sub-pixel value)
// arg 4: wave amplitude
// arg 5: wave speed
void AnimMovePowderParticle(struct Sprite* sprite)
{
    sprite->x += gBattleAnimArgs[0];
    sprite->y += gBattleAnimArgs[1];
    sprite->data[0] = gBattleAnimArgs[2];
    sprite->data[1] = gBattleAnimArgs[3];

    if (GetBattlerSide(gBattleAnimAttacker))
    {
        sprite->data[3] = -gBattleAnimArgs[4];
    }
    else
    {
        sprite->data[3] = gBattleAnimArgs[4];
    }

    sprite->data[4] = gBattleAnimArgs[5];
    sprite->callback = AnimMovePowderParticleStep;
}

static void AnimMovePowderParticleStep(struct Sprite* sprite)
{
    if (sprite->data[0] > 0)
    {
        sprite->data[0]--;
        sprite->y2 = sprite->data[2] >> 8;
        sprite->data[2] += sprite->data[1];
        sprite->x2 = Sin(sprite->data[5], sprite->data[3]);
        sprite->data[5] = (sprite->data[5] + sprite->data[4]) & 0xFF;
    }
    else
    {
        DestroyAnimSprite(sprite);
    }
}
