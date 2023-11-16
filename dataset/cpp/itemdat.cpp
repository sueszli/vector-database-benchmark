/**
 * @file itemdat.cpp
 *
 * Implementation of all item data.
 */

#include "itemdat.h"

#include <string_view>
#include <vector>

#include <expected.hpp>

#include "data/file.hpp"
#include "data/iterators.hpp"
#include "data/record_reader.hpp"
#include "spelldat.h"
#include "utils/str_cat.hpp"

namespace devilution {

/** Contains the data related to each item ID. */
std::vector<ItemData> AllItemsList;

/** Contains the data related to each unique item ID. */
std::vector<UniqueItem> UniqueItems;

/** Contains the data related to each item prefix. */
std::vector<PLStruct> ItemPrefixes;

/** Contains the data related to each item suffix. */
std::vector<PLStruct> ItemSuffixes;

namespace {

tl::expected<item_drop_rate, std::string> ParseItemDropRate(std::string_view value)
{
	if (value == "Never") return IDROP_NEVER;
	if (value == "Regular") return IDROP_REGULAR;
	if (value == "Double") return IDROP_DOUBLE;
	return tl::make_unexpected("Unknown enum value");
}

tl::expected<item_class, std::string> ParseItemClass(std::string_view value)
{
	if (value == "None") return ICLASS_NONE;
	if (value == "Weapon") return ICLASS_WEAPON;
	if (value == "Armor") return ICLASS_ARMOR;
	if (value == "Misc") return ICLASS_MISC;
	if (value == "Gold") return ICLASS_GOLD;
	if (value == "Quest") return ICLASS_QUEST;
	return tl::make_unexpected("Unknown enum value");
}

tl::expected<item_equip_type, std::string> ParseItemEquipType(std::string_view value)
{
	if (value == "None") return ILOC_NONE;
	if (value == "One-handed") return ILOC_ONEHAND;
	if (value == "Two-handed") return ILOC_TWOHAND;
	if (value == "Armor") return ILOC_ARMOR;
	if (value == "Helm") return ILOC_HELM;
	if (value == "Ring") return ILOC_RING;
	if (value == "Amulet") return ILOC_AMULET;
	if (value == "Unequippable") return ILOC_UNEQUIPABLE;
	if (value == "Belt") return ILOC_BELT;
	return tl::make_unexpected("Unknown enum value");
}

tl::expected<item_cursor_graphic, std::string> ParseItemCursorGraphic(std::string_view value)
{
	if (value == "POTION_OF_FULL_MANA") return ICURS_POTION_OF_FULL_MANA;
	if (value == "SCROLL_OF") return ICURS_SCROLL_OF;
	if (value == "GOLD_SMALL") return ICURS_GOLD_SMALL;
	if (value == "GOLD_MEDIUM") return ICURS_GOLD_MEDIUM;
	if (value == "GOLD_LARGE") return ICURS_GOLD_LARGE;
	if (value == "RING_OF_TRUTH") return ICURS_RING_OF_TRUTH;
	if (value == "RING") return ICURS_RING;
	if (value == "SPECTRAL_ELIXIR") return ICURS_SPECTRAL_ELIXIR;
	if (value == "ARENA_POTION") return ICURS_ARENA_POTION;
	if (value == "GOLDEN_ELIXIR") return ICURS_GOLDEN_ELIXIR;
	if (value == "EMPYREAN_BAND") return ICURS_EMPYREAN_BAND;
	if (value == "EAR_SORCERER") return ICURS_EAR_SORCERER;
	if (value == "EAR_WARRIOR") return ICURS_EAR_WARRIOR;
	if (value == "EAR_ROGUE") return ICURS_EAR_ROGUE;
	if (value == "BLOOD_STONE") return ICURS_BLOOD_STONE;
	if (value == "OIL") return ICURS_OIL;
	if (value == "ELIXIR_OF_VITALITY") return ICURS_ELIXIR_OF_VITALITY;
	if (value == "POTION_OF_HEALING") return ICURS_POTION_OF_HEALING;
	if (value == "POTION_OF_FULL_REJUVENATION") return ICURS_POTION_OF_FULL_REJUVENATION;
	if (value == "ELIXIR_OF_MAGIC") return ICURS_ELIXIR_OF_MAGIC;
	if (value == "POTION_OF_FULL_HEALING") return ICURS_POTION_OF_FULL_HEALING;
	if (value == "ELIXIR_OF_DEXTERITY") return ICURS_ELIXIR_OF_DEXTERITY;
	if (value == "POTION_OF_REJUVENATION") return ICURS_POTION_OF_REJUVENATION;
	if (value == "ELIXIR_OF_STRENGTH") return ICURS_ELIXIR_OF_STRENGTH;
	if (value == "POTION_OF_MANA") return ICURS_POTION_OF_MANA;
	if (value == "BRAIN") return ICURS_BRAIN;
	if (value == "OPTIC_AMULET") return ICURS_OPTIC_AMULET;
	if (value == "AMULET") return ICURS_AMULET;
	if (value == "DAGGER") return ICURS_DAGGER;
	if (value == "BLADE") return ICURS_BLADE;
	if (value == "BASTARD_SWORD") return ICURS_BASTARD_SWORD;
	if (value == "MACE") return ICURS_MACE;
	if (value == "LONG_SWORD") return ICURS_LONG_SWORD;
	if (value == "BROAD_SWORD") return ICURS_BROAD_SWORD;
	if (value == "FALCHION") return ICURS_FALCHION;
	if (value == "MORNING_STAR") return ICURS_MORNING_STAR;
	if (value == "SHORT_SWORD") return ICURS_SHORT_SWORD;
	if (value == "CLAYMORE") return ICURS_CLAYMORE;
	if (value == "CLUB") return ICURS_CLUB;
	if (value == "SABRE") return ICURS_SABRE;
	if (value == "SPIKED_CLUB") return ICURS_SPIKED_CLUB;
	if (value == "SCIMITAR") return ICURS_SCIMITAR;
	if (value == "FULL_HELM") return ICURS_FULL_HELM;
	if (value == "MAGIC_ROCK") return ICURS_MAGIC_ROCK;
	if (value == "THE_UNDEAD_CROWN") return ICURS_THE_UNDEAD_CROWN;
	if (value == "HELM") return ICURS_HELM;
	if (value == "BUCKLER") return ICURS_BUCKLER;
	if (value == "VIEL_OF_STEEL") return ICURS_VIEL_OF_STEEL;
	if (value == "BOOK_GREY") return ICURS_BOOK_GREY;
	if (value == "BOOK_RED") return ICURS_BOOK_RED;
	if (value == "BOOK_BLUE") return ICURS_BOOK_BLUE;
	if (value == "BLACK_MUSHROOM") return ICURS_BLACK_MUSHROOM;
	if (value == "SKULL_CAP") return ICURS_SKULL_CAP;
	if (value == "CAP") return ICURS_CAP;
	if (value == "HARLEQUIN_CREST") return ICURS_HARLEQUIN_CREST;
	if (value == "CROWN") return ICURS_CROWN;
	if (value == "MAP_OF_THE_STARS") return ICURS_MAP_OF_THE_STARS;
	if (value == "FUNGAL_TOME") return ICURS_FUNGAL_TOME;
	if (value == "GREAT_HELM") return ICURS_GREAT_HELM;
	if (value == "BATTLE_AXE") return ICURS_BATTLE_AXE;
	if (value == "HUNTERS_BOW") return ICURS_HUNTERS_BOW;
	if (value == "FIELD_PLATE") return ICURS_FIELD_PLATE;
	if (value == "SMALL_SHIELD") return ICURS_SMALL_SHIELD;
	if (value == "CLEAVER") return ICURS_CLEAVER;
	if (value == "STUDDED_LEATHER_ARMOR") return ICURS_STUDDED_LEATHER_ARMOR;
	if (value == "SHORT_STAFF") return ICURS_SHORT_STAFF;
	if (value == "TWO_HANDED_SWORD") return ICURS_TWO_HANDED_SWORD;
	if (value == "CHAIN_MAIL") return ICURS_CHAIN_MAIL;
	if (value == "SMALL_AXE") return ICURS_SMALL_AXE;
	if (value == "KITE_SHIELD") return ICURS_KITE_SHIELD;
	if (value == "SCALE_MAIL") return ICURS_SCALE_MAIL;
	if (value == "SHORT_BOW") return ICURS_SHORT_BOW;
	if (value == "LONG_BATTLE_BOW") return ICURS_LONG_BATTLE_BOW;
	if (value == "LONG_WAR_BOW") return ICURS_LONG_WAR_BOW;
	if (value == "WAR_HAMMER") return ICURS_WAR_HAMMER;
	if (value == "MAUL") return ICURS_MAUL;
	if (value == "LONG_STAFF") return ICURS_LONG_STAFF;
	if (value == "WAR_STAFF") return ICURS_WAR_STAFF;
	if (value == "TAVERN_SIGN") return ICURS_TAVERN_SIGN;
	if (value == "HARD_LEATHER_ARMOR") return ICURS_HARD_LEATHER_ARMOR;
	if (value == "RAGS") return ICURS_RAGS;
	if (value == "QUILTED_ARMOR") return ICURS_QUILTED_ARMOR;
	if (value == "FLAIL") return ICURS_FLAIL;
	if (value == "TOWER_SHIELD") return ICURS_TOWER_SHIELD;
	if (value == "COMPOSITE_BOW") return ICURS_COMPOSITE_BOW;
	if (value == "GREAT_SWORD") return ICURS_GREAT_SWORD;
	if (value == "LEATHER_ARMOR") return ICURS_LEATHER_ARMOR;
	if (value == "SPLINT_MAIL") return ICURS_SPLINT_MAIL;
	if (value == "ROBE") return ICURS_ROBE;
	if (value == "ANVIL_OF_FURY") return ICURS_ANVIL_OF_FURY;
	if (value == "BROAD_AXE") return ICURS_BROAD_AXE;
	if (value == "LARGE_AXE") return ICURS_LARGE_AXE;
	if (value == "GREAT_AXE") return ICURS_GREAT_AXE;
	if (value == "AXE") return ICURS_AXE;
	if (value == "LARGE_SHIELD") return ICURS_LARGE_SHIELD;
	if (value == "GOTHIC_SHIELD") return ICURS_GOTHIC_SHIELD;
	if (value == "CLOAK") return ICURS_CLOAK;
	if (value == "CAPE") return ICURS_CAPE;
	if (value == "FULL_PLATE_MAIL") return ICURS_FULL_PLATE_MAIL;
	if (value == "GOTHIC_PLATE") return ICURS_GOTHIC_PLATE;
	if (value == "BREAST_PLATE") return ICURS_BREAST_PLATE;
	if (value == "RING_MAIL") return ICURS_RING_MAIL;
	if (value == "STAFF_OF_LAZARUS") return ICURS_STAFF_OF_LAZARUS;
	if (value == "ARKAINES_VALOR") return ICURS_ARKAINES_VALOR;
	if (value == "SHORT_WAR_BOW") return ICURS_SHORT_WAR_BOW;
	if (value == "COMPOSITE_STAFF") return ICURS_COMPOSITE_STAFF;
	if (value == "SHORT_BATTLE_BOW") return ICURS_SHORT_BATTLE_BOW;
	if (value == "GOLD") return ICURS_GOLD;
	if (value == "AURIC_AMULET") return ICURS_AURIC_AMULET;
	if (value == "RUNE_BOMB") return ICURS_RUNE_BOMB;
	if (value == "THEODORE") return ICURS_THEODORE;
	if (value == "TORN_NOTE_1") return ICURS_TORN_NOTE_1;
	if (value == "TORN_NOTE_2") return ICURS_TORN_NOTE_2;
	if (value == "TORN_NOTE_3") return ICURS_TORN_NOTE_3;
	if (value == "RECONSTRUCTED_NOTE") return ICURS_RECONSTRUCTED_NOTE;
	if (value == "RUNE_OF_FIRE") return ICURS_RUNE_OF_FIRE;
	if (value == "GREATER_RUNE_OF_FIRE") return ICURS_GREATER_RUNE_OF_FIRE;
	if (value == "RUNE_OF_LIGHTNING") return ICURS_RUNE_OF_LIGHTNING;
	if (value == "GREATER_RUNE_OF_LIGHTNING") return ICURS_GREATER_RUNE_OF_LIGHTNING;
	if (value == "RUNE_OF_STONE") return ICURS_RUNE_OF_STONE;
	if (value == "GREY_SUIT") return ICURS_GREY_SUIT;
	if (value == "BROWN_SUIT") return ICURS_BROWN_SUIT;
	if (value == "BOVINE") return ICURS_BOVINE;
	return tl::make_unexpected("Unknown enum value");
}

tl::expected<ItemType, std::string> ParseItemType(std::string_view value)
{
	if (value == "Misc") return ItemType::Misc;
	if (value == "Sword") return ItemType::Sword;
	if (value == "Axe") return ItemType::Axe;
	if (value == "Bow") return ItemType::Bow;
	if (value == "Mace") return ItemType::Mace;
	if (value == "Shield") return ItemType::Shield;
	if (value == "LightArmor") return ItemType::LightArmor;
	if (value == "Helm") return ItemType::Helm;
	if (value == "MediumArmor") return ItemType::MediumArmor;
	if (value == "HeavyArmor") return ItemType::HeavyArmor;
	if (value == "Staff") return ItemType::Staff;
	if (value == "Gold") return ItemType::Gold;
	if (value == "Ring") return ItemType::Ring;
	if (value == "Amulet") return ItemType::Amulet;
	if (value == "None") return ItemType::None;
	return tl::make_unexpected("Unknown enum value");
}

tl::expected<unique_base_item, std::string> ParseUniqueBaseItem(std::string_view value)
{
	if (value == "NONE") return UITYPE_NONE;
	if (value == "SHORTBOW") return UITYPE_SHORTBOW;
	if (value == "LONGBOW") return UITYPE_LONGBOW;
	if (value == "HUNTBOW") return UITYPE_HUNTBOW;
	if (value == "COMPBOW") return UITYPE_COMPBOW;
	if (value == "WARBOW") return UITYPE_WARBOW;
	if (value == "BATTLEBOW") return UITYPE_BATTLEBOW;
	if (value == "DAGGER") return UITYPE_DAGGER;
	if (value == "FALCHION") return UITYPE_FALCHION;
	if (value == "CLAYMORE") return UITYPE_CLAYMORE;
	if (value == "BROADSWR") return UITYPE_BROADSWR;
	if (value == "SABRE") return UITYPE_SABRE;
	if (value == "SCIMITAR") return UITYPE_SCIMITAR;
	if (value == "LONGSWR") return UITYPE_LONGSWR;
	if (value == "BASTARDSWR") return UITYPE_BASTARDSWR;
	if (value == "TWOHANDSWR") return UITYPE_TWOHANDSWR;
	if (value == "GREATSWR") return UITYPE_GREATSWR;
	if (value == "CLEAVER") return UITYPE_CLEAVER;
	if (value == "LARGEAXE") return UITYPE_LARGEAXE;
	if (value == "BROADAXE") return UITYPE_BROADAXE;
	if (value == "SMALLAXE") return UITYPE_SMALLAXE;
	if (value == "BATTLEAXE") return UITYPE_BATTLEAXE;
	if (value == "GREATAXE") return UITYPE_GREATAXE;
	if (value == "MACE") return UITYPE_MACE;
	if (value == "MORNSTAR") return UITYPE_MORNSTAR;
	if (value == "SPIKCLUB") return UITYPE_SPIKCLUB;
	if (value == "MAUL") return UITYPE_MAUL;
	if (value == "WARHAMMER") return UITYPE_WARHAMMER;
	if (value == "FLAIL") return UITYPE_FLAIL;
	if (value == "LONGSTAFF") return UITYPE_LONGSTAFF;
	if (value == "SHORTSTAFF") return UITYPE_SHORTSTAFF;
	if (value == "COMPSTAFF") return UITYPE_COMPSTAFF;
	if (value == "QUARSTAFF") return UITYPE_QUARSTAFF;
	if (value == "WARSTAFF") return UITYPE_WARSTAFF;
	if (value == "SKULLCAP") return UITYPE_SKULLCAP;
	if (value == "HELM") return UITYPE_HELM;
	if (value == "GREATHELM") return UITYPE_GREATHELM;
	if (value == "CROWN") return UITYPE_CROWN;
	if (value == "RAGS") return UITYPE_RAGS;
	if (value == "STUDARMOR") return UITYPE_STUDARMOR;
	if (value == "CLOAK") return UITYPE_CLOAK;
	if (value == "ROBE") return UITYPE_ROBE;
	if (value == "CHAINMAIL") return UITYPE_CHAINMAIL;
	if (value == "LEATHARMOR") return UITYPE_LEATHARMOR;
	if (value == "BREASTPLATE") return UITYPE_BREASTPLATE;
	if (value == "CAPE") return UITYPE_CAPE;
	if (value == "PLATEMAIL") return UITYPE_PLATEMAIL;
	if (value == "FULLPLATE") return UITYPE_FULLPLATE;
	if (value == "BUCKLER") return UITYPE_BUCKLER;
	if (value == "SMALLSHIELD") return UITYPE_SMALLSHIELD;
	if (value == "LARGESHIELD") return UITYPE_LARGESHIELD;
	if (value == "KITESHIELD") return UITYPE_KITESHIELD;
	if (value == "GOTHSHIELD") return UITYPE_GOTHSHIELD;
	if (value == "RING") return UITYPE_RING;
	if (value == "AMULET") return UITYPE_AMULET;
	if (value == "SKCROWN") return UITYPE_SKCROWN;
	if (value == "INFRARING") return UITYPE_INFRARING;
	if (value == "OPTAMULET") return UITYPE_OPTAMULET;
	if (value == "TRING") return UITYPE_TRING;
	if (value == "HARCREST") return UITYPE_HARCREST;
	if (value == "MAPOFDOOM") return UITYPE_MAPOFDOOM;
	if (value == "ELIXIR") return UITYPE_ELIXIR;
	if (value == "ARMOFVAL") return UITYPE_ARMOFVAL;
	if (value == "STEELVEIL") return UITYPE_STEELVEIL;
	if (value == "GRISWOLD") return UITYPE_GRISWOLD;
	if (value == "LGTFORGE") return UITYPE_LGTFORGE;
	if (value == "LAZSTAFF") return UITYPE_LAZSTAFF;
	if (value == "BOVINE") return UITYPE_BOVINE;
	if (value == "INVALID") return UITYPE_INVALID;
	return tl::make_unexpected("Unknown enum value");
}

tl::expected<ItemSpecialEffect, std::string> ParseItemSpecialEffect(std::string_view value)
{
	if (value == "RandomStealLife") return ItemSpecialEffect::RandomStealLife;
	if (value == "RandomArrowVelocity") return ItemSpecialEffect::RandomArrowVelocity;
	if (value == "FireArrows") return ItemSpecialEffect::FireArrows;
	if (value == "FireDamage") return ItemSpecialEffect::FireDamage;
	if (value == "LightningDamage") return ItemSpecialEffect::LightningDamage;
	if (value == "DrainLife") return ItemSpecialEffect::DrainLife;
	if (value == "MultipleArrows") return ItemSpecialEffect::MultipleArrows;
	if (value == "Knockback") return ItemSpecialEffect::Knockback;
	if (value == "StealMana3") return ItemSpecialEffect::StealMana3;
	if (value == "StealMana5") return ItemSpecialEffect::StealMana5;
	if (value == "StealLife3") return ItemSpecialEffect::StealLife3;
	if (value == "StealLife5") return ItemSpecialEffect::StealLife5;
	if (value == "QuickAttack") return ItemSpecialEffect::QuickAttack;
	if (value == "FastAttack") return ItemSpecialEffect::FastAttack;
	if (value == "FasterAttack") return ItemSpecialEffect::FasterAttack;
	if (value == "FastestAttack") return ItemSpecialEffect::FastestAttack;
	if (value == "FastHitRecovery") return ItemSpecialEffect::FastHitRecovery;
	if (value == "FasterHitRecovery") return ItemSpecialEffect::FasterHitRecovery;
	if (value == "FastestHitRecovery") return ItemSpecialEffect::FastestHitRecovery;
	if (value == "FastBlock") return ItemSpecialEffect::FastBlock;
	if (value == "LightningArrows") return ItemSpecialEffect::LightningArrows;
	if (value == "Thorns") return ItemSpecialEffect::Thorns;
	if (value == "NoMana") return ItemSpecialEffect::NoMana;
	if (value == "HalfTrapDamage") return ItemSpecialEffect::HalfTrapDamage;
	if (value == "TripleDemonDamage") return ItemSpecialEffect::TripleDemonDamage;
	if (value == "ZeroResistance") return ItemSpecialEffect::ZeroResistance;
	return tl::make_unexpected("Unknown enum value");
}

tl::expected<item_misc_id, std::string> ParseItemMiscId(std::string_view value)
{
	if (value == "NONE") return IMISC_NONE;
	if (value == "USEFIRST") return IMISC_USEFIRST;
	if (value == "FULLHEAL") return IMISC_FULLHEAL;
	if (value == "HEAL") return IMISC_HEAL;
	if (value == "MANA") return IMISC_MANA;
	if (value == "FULLMANA") return IMISC_FULLMANA;
	if (value == "ELIXSTR") return IMISC_ELIXSTR;
	if (value == "ELIXMAG") return IMISC_ELIXMAG;
	if (value == "ELIXDEX") return IMISC_ELIXDEX;
	if (value == "ELIXVIT") return IMISC_ELIXVIT;
	if (value == "REJUV") return IMISC_REJUV;
	if (value == "FULLREJUV") return IMISC_FULLREJUV;
	if (value == "USELAST") return IMISC_USELAST;
	if (value == "SCROLL") return IMISC_SCROLL;
	if (value == "SCROLLT") return IMISC_SCROLLT;
	if (value == "STAFF") return IMISC_STAFF;
	if (value == "BOOK") return IMISC_BOOK;
	if (value == "RING") return IMISC_RING;
	if (value == "AMULET") return IMISC_AMULET;
	if (value == "UNIQUE") return IMISC_UNIQUE;
	if (value == "OILFIRST") return IMISC_OILFIRST;
	if (value == "OILOF") return IMISC_OILOF;
	if (value == "OILACC") return IMISC_OILACC;
	if (value == "OILMAST") return IMISC_OILMAST;
	if (value == "OILSHARP") return IMISC_OILSHARP;
	if (value == "OILDEATH") return IMISC_OILDEATH;
	if (value == "OILSKILL") return IMISC_OILSKILL;
	if (value == "OILBSMTH") return IMISC_OILBSMTH;
	if (value == "OILFORT") return IMISC_OILFORT;
	if (value == "OILPERM") return IMISC_OILPERM;
	if (value == "OILHARD") return IMISC_OILHARD;
	if (value == "OILIMP") return IMISC_OILIMP;
	if (value == "OILLAST") return IMISC_OILLAST;
	if (value == "MAPOFDOOM") return IMISC_MAPOFDOOM;
	if (value == "EAR") return IMISC_EAR;
	if (value == "SPECELIX") return IMISC_SPECELIX;
	if (value == "RUNEFIRST") return IMISC_RUNEFIRST;
	if (value == "RUNEF") return IMISC_RUNEF;
	if (value == "RUNEL") return IMISC_RUNEL;
	if (value == "GR_RUNEL") return IMISC_GR_RUNEL;
	if (value == "GR_RUNEF") return IMISC_GR_RUNEF;
	if (value == "RUNES") return IMISC_RUNES;
	if (value == "RUNELAST") return IMISC_RUNELAST;
	if (value == "AURIC") return IMISC_AURIC;
	if (value == "NOTE") return IMISC_NOTE;
	if (value == "ARENAPOT") return IMISC_ARENAPOT;
	return tl::make_unexpected("Unknown enum value");
}

tl::expected<item_effect_type, std::string> ParseItemEffectType(std::string_view value)
{
	if (value == "TOHIT") return IPL_TOHIT;
	if (value == "TOHIT_CURSE") return IPL_TOHIT_CURSE;
	if (value == "DAMP") return IPL_DAMP;
	if (value == "DAMP_CURSE") return IPL_DAMP_CURSE;
	if (value == "TOHIT_DAMP") return IPL_TOHIT_DAMP;
	if (value == "TOHIT_DAMP_CURSE") return IPL_TOHIT_DAMP_CURSE;
	if (value == "ACP") return IPL_ACP;
	if (value == "ACP_CURSE") return IPL_ACP_CURSE;
	if (value == "FIRERES") return IPL_FIRERES;
	if (value == "LIGHTRES") return IPL_LIGHTRES;
	if (value == "MAGICRES") return IPL_MAGICRES;
	if (value == "ALLRES") return IPL_ALLRES;
	if (value == "SPLLVLADD") return IPL_SPLLVLADD;
	if (value == "CHARGES") return IPL_CHARGES;
	if (value == "FIREDAM") return IPL_FIREDAM;
	if (value == "LIGHTDAM") return IPL_LIGHTDAM;
	if (value == "STR") return IPL_STR;
	if (value == "STR_CURSE") return IPL_STR_CURSE;
	if (value == "MAG") return IPL_MAG;
	if (value == "MAG_CURSE") return IPL_MAG_CURSE;
	if (value == "DEX") return IPL_DEX;
	if (value == "DEX_CURSE") return IPL_DEX_CURSE;
	if (value == "VIT") return IPL_VIT;
	if (value == "VIT_CURSE") return IPL_VIT_CURSE;
	if (value == "ATTRIBS") return IPL_ATTRIBS;
	if (value == "ATTRIBS_CURSE") return IPL_ATTRIBS_CURSE;
	if (value == "GETHIT_CURSE") return IPL_GETHIT_CURSE;
	if (value == "GETHIT") return IPL_GETHIT;
	if (value == "LIFE") return IPL_LIFE;
	if (value == "LIFE_CURSE") return IPL_LIFE_CURSE;
	if (value == "MANA") return IPL_MANA;
	if (value == "MANA_CURSE") return IPL_MANA_CURSE;
	if (value == "DUR") return IPL_DUR;
	if (value == "DUR_CURSE") return IPL_DUR_CURSE;
	if (value == "INDESTRUCTIBLE") return IPL_INDESTRUCTIBLE;
	if (value == "LIGHT") return IPL_LIGHT;
	if (value == "LIGHT_CURSE") return IPL_LIGHT_CURSE;
	if (value == "MULT_ARROWS") return IPL_MULT_ARROWS;
	if (value == "FIRE_ARROWS") return IPL_FIRE_ARROWS;
	if (value == "LIGHT_ARROWS") return IPL_LIGHT_ARROWS;
	if (value == "INVCURS") return IPL_INVCURS;
	if (value == "THORNS") return IPL_THORNS;
	if (value == "NOMANA") return IPL_NOMANA;
	if (value == "FIREBALL") return IPL_FIREBALL;
	if (value == "ABSHALFTRAP") return IPL_ABSHALFTRAP;
	if (value == "KNOCKBACK") return IPL_KNOCKBACK;
	if (value == "STEALMANA") return IPL_STEALMANA;
	if (value == "STEALLIFE") return IPL_STEALLIFE;
	if (value == "TARGAC") return IPL_TARGAC;
	if (value == "FASTATTACK") return IPL_FASTATTACK;
	if (value == "FASTRECOVER") return IPL_FASTRECOVER;
	if (value == "FASTBLOCK") return IPL_FASTBLOCK;
	if (value == "DAMMOD") return IPL_DAMMOD;
	if (value == "RNDARROWVEL") return IPL_RNDARROWVEL;
	if (value == "SETDAM") return IPL_SETDAM;
	if (value == "SETDUR") return IPL_SETDUR;
	if (value == "NOMINSTR") return IPL_NOMINSTR;
	if (value == "SPELL") return IPL_SPELL;
	if (value == "ONEHAND") return IPL_ONEHAND;
	if (value == "3XDAMVDEM") return IPL_3XDAMVDEM;
	if (value == "ALLRESZERO") return IPL_ALLRESZERO;
	if (value == "DRAINLIFE") return IPL_DRAINLIFE;
	if (value == "RNDSTEALLIFE") return IPL_RNDSTEALLIFE;
	if (value == "SETAC") return IPL_SETAC;
	if (value == "ADDACLIFE") return IPL_ADDACLIFE;
	if (value == "ADDMANAAC") return IPL_ADDMANAAC;
	if (value == "AC_CURSE") return IPL_AC_CURSE;
	if (value == "LASTDIABLO") return IPL_LASTDIABLO;
	if (value == "FIRERES_CURSE") return IPL_FIRERES_CURSE;
	if (value == "LIGHTRES_CURSE") return IPL_LIGHTRES_CURSE;
	if (value == "MAGICRES_CURSE") return IPL_MAGICRES_CURSE;
	if (value == "DEVASTATION") return IPL_DEVASTATION;
	if (value == "DECAY") return IPL_DECAY;
	if (value == "PERIL") return IPL_PERIL;
	if (value == "JESTERS") return IPL_JESTERS;
	if (value == "CRYSTALLINE") return IPL_CRYSTALLINE;
	if (value == "DOPPELGANGER") return IPL_DOPPELGANGER;
	if (value == "ACDEMON") return IPL_ACDEMON;
	if (value == "ACUNDEAD") return IPL_ACUNDEAD;
	if (value == "MANATOLIFE") return IPL_MANATOLIFE;
	if (value == "LIFETOMANA") return IPL_LIFETOMANA;
	return tl::make_unexpected("Unknown enum value");
}

tl::expected<AffixItemType, std::string> ParseAffixItemType(std::string_view value)
{
	if (value == "Misc") return AffixItemType::Misc;
	if (value == "Bow") return AffixItemType::Bow;
	if (value == "Staff") return AffixItemType::Staff;
	if (value == "Weapon") return AffixItemType::Weapon;
	if (value == "Shield") return AffixItemType::Shield;
	if (value == "Armor") return AffixItemType::Armor;
	return tl::make_unexpected("Unknown enum value");
}

tl::expected<goodorevil, std::string> ParseAffixAlignment(std::string_view value)
{
	if (value == "Any") return GOE_ANY;
	if (value == "Evil") return GOE_EVIL;
	if (value == "Good") return GOE_GOOD;
	return tl::make_unexpected("Unknown enum value");
}

void LoadItemDat()
{
	const std::string_view filename = "txtdata\\items\\itemdat.tsv";
	tl::expected<DataFile, DataFile::Error> dataFileResult = DataFile::load(filename);
	if (!dataFileResult.has_value()) {
		DataFile::reportFatalError(dataFileResult.error(), filename);
	}

	DataFile &dataFile = dataFileResult.value();
	if (tl::expected<void, DataFile::Error> result = dataFile.skipHeader();
	    !result.has_value()) {
		DataFile::reportFatalError(result.error(), filename);
	}

	AllItemsList.clear();
	AllItemsList.reserve(dataFile.numRecords());
	for (DataFileRecord record : dataFile) {
		RecordReader reader { record, filename };
		ItemData &item = AllItemsList.emplace_back();
		reader.advance(); // Skip the first column (item ID).
		reader.read("dropRate", item.iRnd, ParseItemDropRate);
		reader.read("class", item.iClass, ParseItemClass);
		reader.read("equipType", item.iLoc, ParseItemEquipType);
		reader.read("cursorGraphic", item.iCurs, ParseItemCursorGraphic);
		reader.read("itemType", item.itype, ParseItemType);
		reader.read("uniqueBaseItem", item.iItemId, ParseUniqueBaseItem);
		reader.readString("name", item.iName);
		reader.readString("shortName", item.iSName);
		reader.readInt("minMonsterLevel", item.iMinMLvl);
		reader.readInt("durability", item.iDurability);
		reader.readInt("minDamage", item.iMinDam);
		reader.readInt("maxDamage", item.iMaxDam);
		reader.readInt("minArmor", item.iMinAC);
		reader.readInt("maxArmor", item.iMaxAC);
		reader.readInt("minStrength", item.iMinStr);
		reader.readInt("minMagic", item.iMinMag);
		reader.readInt("minDexterity", item.iMinDex);
		reader.readEnumList("specialEffects", item.iFlags, ParseItemSpecialEffect);
		reader.read("miscId", item.iMiscId, ParseItemMiscId);
		reader.read("spell", item.iSpell, ParseSpellId);
		reader.readBool("usable", item.iUsable);
		reader.readInt("value", item.iValue);
	}
	AllItemsList.shrink_to_fit();
}

void ReadItemPower(RecordReader &reader, std::string_view fieldName, ItemPower &power)
{
	reader.read(fieldName, power.type, ParseItemEffectType);
	reader.readOptionalInt(StrCat(fieldName, ".value1"), power.param1);
	reader.readOptionalInt(StrCat(fieldName, ".value2"), power.param2);
}

void LoadUniqueItemDat()
{
	const std::string_view filename = "txtdata\\items\\unique_itemdat.tsv";
	tl::expected<DataFile, DataFile::Error> dataFileResult = DataFile::load(filename);
	if (!dataFileResult.has_value()) {
		DataFile::reportFatalError(dataFileResult.error(), filename);
	}

	DataFile &dataFile = dataFileResult.value();
	if (tl::expected<void, DataFile::Error> result = dataFile.skipHeader();
	    !result.has_value()) {
		DataFile::reportFatalError(result.error(), filename);
	}

	UniqueItems.clear();
	UniqueItems.reserve(dataFile.numRecords());
	for (DataFileRecord record : dataFile) {
		RecordReader reader { record, filename };
		UniqueItem &item = UniqueItems.emplace_back();
		reader.readString("name", item.UIName);
		reader.read("uniqueBaseItem", item.UIItemId, ParseUniqueBaseItem);
		reader.readInt("minLevel", item.UIMinLvl);
		reader.readInt("value", item.UIValue);

		// powers (up to 6)
		item.UINumPL = 0;
		for (size_t i = 0; i < 6; ++i) {
			if (reader.value().empty())
				break;
			ReadItemPower(reader, StrCat("power", i), item.powers[item.UINumPL++]);
		}
	}
	UniqueItems.shrink_to_fit();
}

void LoadItemAffixesDat(std::string_view filename, std::vector<PLStruct> &out)
{
	tl::expected<DataFile, DataFile::Error> dataFileResult = DataFile::load(filename);
	if (!dataFileResult.has_value()) {
		DataFile::reportFatalError(dataFileResult.error(), filename);
	}

	DataFile &dataFile = dataFileResult.value();
	if (tl::expected<void, DataFile::Error> result = dataFile.skipHeader();
	    !result.has_value()) {
		DataFile::reportFatalError(result.error(), filename);
	}

	out.clear();
	out.reserve(dataFile.numRecords());
	for (DataFileRecord record : dataFile) {
		RecordReader reader { record, filename };
		PLStruct &item = out.emplace_back();
		reader.readString("name", item.PLName);
		ReadItemPower(reader, "power", item.power);
		reader.readInt("minLevel", item.PLMinLvl);
		reader.readEnumList("itemTypes", item.PLIType, ParseAffixItemType);
		reader.read("alignment", item.PLGOE, ParseAffixAlignment);
		reader.readBool("doubleChance", item.PLDouble);
		reader.readBool("useful", item.PLOk);
		reader.readInt("minVal", item.minVal);
		reader.readInt("maxVal", item.maxVal);
		reader.readInt("multVal", item.multVal);
	}
	out.shrink_to_fit();
}

} // namespace

void LoadItemData()
{
	LoadItemDat();
	LoadUniqueItemDat();
	LoadItemAffixesDat("txtdata\\items\\item_prefixes.tsv", ItemPrefixes);
	LoadItemAffixesDat("txtdata\\items\\item_suffixes.tsv", ItemSuffixes);
}

std::string_view ItemTypeToString(ItemType itemType)
{
	switch (itemType) {
	case ItemType::Misc: return "Misc";
	case ItemType::Sword: return "Sword";
	case ItemType::Axe: return "Axe";
	case ItemType::Bow: return "Bow";
	case ItemType::Mace: return "Mace";
	case ItemType::Shield: return "Shield";
	case ItemType::LightArmor: return "LightArmor";
	case ItemType::Helm: return "Helm";
	case ItemType::MediumArmor: return "MediumArmor";
	case ItemType::HeavyArmor: return "HeavyArmor";
	case ItemType::Staff: return "Staff";
	case ItemType::Gold: return "Gold";
	case ItemType::Ring: return "Ring";
	case ItemType::Amulet: return "Amulet";
	case ItemType::None: return "None";
	}
	return "";
}

} // namespace devilution
