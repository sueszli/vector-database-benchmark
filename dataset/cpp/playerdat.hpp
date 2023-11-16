/**
 * @file playerdat.hpp
 *
 * Interface of all player data.
 */
#pragma once

#include <cstdint>

#include "effects.h"
#include "itemdat.h"
#include "spelldat.h"

namespace devilution {

enum class HeroClass : uint8_t {
	Warrior,
	Rogue,
	Sorcerer,
	Monk,
	Bard,
	Barbarian,

	LAST = Barbarian
};

struct PlayerData {
	/* Class Name */
	const char *className;
	/* Class Skill */
	SpellID skill = SpellID::Null;
};

struct ClassAttributes {
	/* Class Starting Strength Stat */
	uint8_t baseStr;
	/* Class Starting Magic Stat */
	uint8_t baseMag;
	/* Class Starting Dexterity Stat */
	uint8_t baseDex;
	/* Class Starting Vitality Stat */
	uint8_t baseVit;
	/* Class Maximum Strength Stat */
	uint8_t maxStr;
	/* Class Maximum Magic Stat */
	uint8_t maxMag;
	/* Class Maximum Dexterity Stat */
	uint8_t maxDex;
	/* Class Maximum Vitality Stat */
	uint8_t maxVit;
	/* Class Life Adjustment */
	int16_t adjLife;
	/* Class Mana Adjustment */
	int16_t adjMana;
	/* Life gained on level up */
	int16_t lvlLife;
	/* Mana gained on level up */
	int16_t lvlMana;
	/* Life from base Vitality */
	int16_t chrLife;
	/* Mana from base Magic */
	int16_t chrMana;
	/* Life from item bonus Vitality */
	int16_t itmLife;
	/* Mana from item bonus Magic */
	int16_t itmMana;
};

const ClassAttributes &GetClassAttributes(HeroClass playerClass);

struct PlayerCombatData {
	/* Class starting chance to Block (used as a %) */
	uint8_t baseToBlock;
	/* Class starting chance to hit when using melee attacks (used as a %) */
	uint8_t baseMeleeToHit;
	/* Class starting chance to hit when using ranged weapons (used as a %) */
	uint8_t baseRangedToHit;
	/* Class starting chance to hit when using spells (used as a %) */
	uint8_t baseMagicToHit;
};

/**
 * @brief Data used to set known skills and provide initial equipment when starting a new game
 *
 * Items will be created in order starting with item 1, 2, etc. If the item can be equipped it
 * will be placed in the first available slot, otherwise if it fits on the belt it will be
 * placed in the first free space, finally being placed in the first free inventory position.
 *
 * The active game mode at the time we're creating a new character controls the choice of item
 * type. ItemType.hellfire is used if we're in Hellfire mode, ItemType.diablo otherwise.
 */
struct PlayerStartingLoadoutData {
	/* Class Skill */
	SpellID skill;
	/* Starting Spell (if any) */
	SpellID spell;
	/* Initial level of the starting spell */
	uint8_t spellLevel;

	struct ItemType {
		_item_indexes diablo;
		_item_indexes hellfire;
	};

	std::array<ItemType, 5> items;

	/* Initial gold amount, up to a single stack (5000 gold) */
	uint16_t gold;
};

struct PlayerSpriteData {
	/* Class Directory Path */
	const char *classPath;
	/* Sprite width: Stand */
	uint8_t stand;
	/* Sprite width: Walk */
	uint8_t walk;
	/* Sprite width: Attack */
	uint8_t attack;
	/* Sprite width: Attack (Bow) */
	uint8_t bow;
	/* Sprite width: Hit Recovery */
	uint8_t swHit;
	/* Sprite width: Block */
	uint8_t block;
	/* Sprite width: Cast Lightning Spell */
	uint8_t lightning;
	/* Sprite width: Cast Fire Spell */
	uint8_t fire;
	/* Sprite width: Cast Magic Spell */
	uint8_t magic;
	/* Sprite width: Death */
	uint8_t death;
};

struct PlayerAnimData {
	/* Unarmed frame count */
	int8_t unarmedFrames;
	/* Unarmed action frame number */
	int8_t unarmedActionFrame;
	/* UnarmedShield frame count */
	int8_t unarmedShieldFrames;
	/* UnarmedShield action frame number */
	int8_t unarmedShieldActionFrame;
	/* Sword frame count */
	int8_t swordFrames;
	/* Sword action frame number */
	int8_t swordActionFrame;
	/* SwordShield frame count */
	int8_t swordShieldFrames;
	/* SwordShield action frame number */
	int8_t swordShieldActionFrame;
	/* Bow frame count */
	int8_t bowFrames;
	/* Bow action frame number */
	int8_t bowActionFrame;
	/* Axe frame count */
	int8_t axeFrames;
	/* Axe action frame number */
	int8_t axeActionFrame;
	/* Mace frame count */
	int8_t maceFrames;
	/* Mace action frame */
	int8_t maceActionFrame;
	/* MaceShield frame count */
	int8_t maceShieldFrames;
	/* MaceShield action frame number */
	int8_t maceShieldActionFrame;
	/* Staff frame count */
	int8_t staffFrames;
	/* Staff action frame number */
	int8_t staffActionFrame;
	/* Nothing (Idle) frame count */
	int8_t idleFrames;
	/* Walking frame count */
	int8_t walkingFrames;
	/* Blocking frame count */
	int8_t blockingFrames;
	/* Death frame count */
	int8_t deathFrames;
	/* Spellcasting frame count */
	int8_t castingFrames;
	/* Hit Recovery frame count */
	int8_t recoveryFrames;
	/* Town Nothing (Idle) frame count */
	int8_t townIdleFrames;
	/* Town Walking frame count */
	int8_t townWalkingFrames;
	/* Spellcasting action frame number */
	int8_t castingActionFrame;
};

/**
 * @brief Attempts to load data values from external files, currently only Experience.tsv is supported.
 */
void LoadPlayerDataFiles();

extern const SfxID herosounds[enum_size<HeroClass>::value][enum_size<HeroSpeech>::value];
uint32_t GetNextExperienceThresholdForLevel(unsigned level);
uint8_t GetMaximumCharacterLevel();
const PlayerData &GetPlayerDataForClass(HeroClass clazz);
const PlayerCombatData &GetPlayerCombatDataForClass(HeroClass clazz);
const PlayerStartingLoadoutData &GetPlayerStartingLoadoutForClass(HeroClass clazz);
extern const PlayerSpriteData PlayersSpriteData[];
extern const PlayerAnimData PlayersAnimData[];

} // namespace devilution
