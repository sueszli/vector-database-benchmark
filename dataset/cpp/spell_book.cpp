#include "panels/spell_book.hpp"

#include <cstdint>
#include <optional>

#include <fmt/format.h>

#include "control.h"
#include "engine/backbuffer_state.hpp"
#include "engine/clx_sprite.hpp"
#include "engine/load_cel.hpp"
#include "engine/load_clx.hpp"
#include "engine/rectangle.hpp"
#include "engine/render/clx_render.hpp"
#include "engine/render/text_render.hpp"
#include "init.h"
#include "missiles.h"
#include "panels/spell_icons.hpp"
#include "panels/ui_panels.hpp"
#include "player.h"
#include "spelldat.h"
#include "utils/language.h"

namespace devilution {

namespace {

OptionalOwnedClxSpriteList spellBookButtons;
OptionalOwnedClxSpriteList spellBookBackground;

const size_t SpellBookPages = 6;
const size_t SpellBookPageEntries = 7;

constexpr uint16_t SpellBookButtonWidthDiablo = 76;
constexpr uint16_t SpellBookButtonWidthHellfire = 61;

uint16_t SpellBookButtonWidth()
{
	return gbIsHellfire ? SpellBookButtonWidthHellfire : SpellBookButtonWidthDiablo;
}

/** Maps from spellbook page number and position to SpellID. */
const SpellID SpellPages[SpellBookPages][SpellBookPageEntries] = {
	{ SpellID::Null, SpellID::Firebolt, SpellID::ChargedBolt, SpellID::HolyBolt, SpellID::Healing, SpellID::HealOther, SpellID::Inferno },
	{ SpellID::Resurrect, SpellID::FireWall, SpellID::Telekinesis, SpellID::Lightning, SpellID::TownPortal, SpellID::Flash, SpellID::StoneCurse },
	{ SpellID::Phasing, SpellID::ManaShield, SpellID::Elemental, SpellID::Fireball, SpellID::FlameWave, SpellID::ChainLightning, SpellID::Guardian },
	{ SpellID::Nova, SpellID::Golem, SpellID::Teleport, SpellID::Apocalypse, SpellID::BoneSpirit, SpellID::BloodStar, SpellID::Etherealize },
	{ SpellID::LightningWall, SpellID::Immolation, SpellID::Warp, SpellID::Reflect, SpellID::Berserk, SpellID::RingOfFire, SpellID::Search },
	{ SpellID::Invalid, SpellID::Invalid, SpellID::Invalid, SpellID::Invalid, SpellID::Invalid, SpellID::Invalid, SpellID::Invalid }
};

SpellID GetSpellFromSpellPage(size_t page, size_t entry)
{
	assert(page <= SpellBookPages && entry <= SpellBookPageEntries);
	if (page == 0 && entry == 0) {
		switch (InspectPlayer->_pClass) {
		case HeroClass::Warrior:
			return SpellID::ItemRepair;
		case HeroClass::Rogue:
			return SpellID::TrapDisarm;
		case HeroClass::Sorcerer:
			return SpellID::StaffRecharge;
		case HeroClass::Monk:
			return SpellID::Search;
		case HeroClass::Bard:
			return SpellID::Identify;
		case HeroClass::Barbarian:
			return SpellID::Rage;
		}
	}
	return SpellPages[page][entry];
}

constexpr Size SpellBookDescription { 250, 43 };
constexpr int SpellBookDescriptionPaddingHorizontal = 2;

void PrintSBookStr(const Surface &out, Point position, std::string_view text, UiFlags flags = UiFlags::None)
{
	DrawString(out, text,
	    Rectangle(GetPanelPosition(UiPanels::Spell, position + Displacement { SPLICONLENGTH, 0 }),
	        SpellBookDescription)
	        .inset({ SpellBookDescriptionPaddingHorizontal, 0 }),
	    { .flags = UiFlags::ColorWhite | flags });
}

SpellType GetSBookTrans(SpellID ii, bool townok)
{
	Player &player = *InspectPlayer;
	if ((player._pClass == HeroClass::Monk) && (ii == SpellID::Search))
		return SpellType::Skill;
	SpellType st = SpellType::Spell;
	if ((player._pISpells & GetSpellBitmask(ii)) != 0) {
		st = SpellType::Charges;
	}
	if ((player._pAblSpells & GetSpellBitmask(ii)) != 0) {
		st = SpellType::Skill;
	}
	if (st == SpellType::Spell) {
		if (CheckSpell(*InspectPlayer, ii, st, true) != SpellCheckResult::Success) {
			st = SpellType::Invalid;
		}
		if (player.GetSpellLevel(ii) == 0) {
			st = SpellType::Invalid;
		}
	}
	if (townok && leveltype == DTYPE_TOWN && st != SpellType::Invalid && !GetSpellData(ii).isAllowedInTown()) {
		st = SpellType::Invalid;
	}

	return st;
}

StringOrView GetSpellPowerText(SpellID spell, int spellLevel)
{
	if (spellLevel == 0) {
		return _("Unusable");
	}
	if (spell == SpellID::BoneSpirit) {
		return _(/* TRANSLATORS: UI constraints, keep short please.*/ "Dmg: 1/3 target hp");
	}
	const auto [min, max] = GetDamageAmt(spell, spellLevel);
	if (min == -1) {
		return StringOrView {};
	}
	if (spell == SpellID::Healing || spell == SpellID::HealOther) {
		return fmt::format(fmt::runtime(_(/* TRANSLATORS: UI constraints, keep short please.*/ "Heals: {:d} - {:d}")), min, max);
	}
	return fmt::format(fmt::runtime(_(/* TRANSLATORS: UI constraints, keep short please.*/ "Damage: {:d} - {:d}")), min, max);
}

} // namespace

void InitSpellBook()
{
	spellBookBackground = LoadCel("data\\spellbk", static_cast<uint16_t>(SidePanelSize.width));
	spellBookButtons = LoadCel("data\\spellbkb", SpellBookButtonWidth());
	LoadSmallSpellIcons();
}

void FreeSpellBook()
{
	FreeSmallSpellIcons();
	spellBookButtons = std::nullopt;
	spellBookBackground = std::nullopt;
}

void DrawSpellBook(const Surface &out)
{
	constexpr int SpellBookButtonX = 7;
	constexpr int SpellBookButtonY = 348;
	ClxDraw(out, GetPanelPosition(UiPanels::Spell, { 0, 351 }), (*spellBookBackground)[0]);
	const int buttonX = gbIsHellfire && sbooktab < 5
	    ? SpellBookButtonWidthHellfire * sbooktab
	    : SpellBookButtonWidthDiablo * sbooktab
	        // BUGFIX: rendering of page 3 and page 4 buttons are both off-by-one pixel (fixed).
	        + (sbooktab == 2 || sbooktab == 3 ? 1 : 0);

	ClxDraw(out, GetPanelPosition(UiPanels::Spell, { SpellBookButtonX + buttonX, SpellBookButtonY }), (*spellBookButtons)[sbooktab]);
	Player &player = *InspectPlayer;
	uint64_t spl = player._pMemSpells | player._pISpells | player._pAblSpells;

	const int lineHeight = 18;

	int yp = 12;
	const int textPaddingTop = 7;
	for (size_t pageEntry = 0; pageEntry < SpellBookPageEntries; pageEntry++) {
		SpellID sn = GetSpellFromSpellPage(sbooktab, pageEntry);
		if (IsValidSpell(sn) && (spl & GetSpellBitmask(sn)) != 0) {
			SpellType st = GetSBookTrans(sn, true);
			SetSpellTrans(st);
			const Point spellCellPosition = GetPanelPosition(UiPanels::Spell, { 11, yp + SpellBookDescription.height });
			DrawSmallSpellIcon(out, spellCellPosition, sn);
			if (sn == player._pRSpell && st == player._pRSplType && !IsInspectingPlayer()) {
				SetSpellTrans(SpellType::Skill);
				DrawSmallSpellIconBorder(out, spellCellPosition);
			}

			const Point line0 { 0, yp + textPaddingTop };
			const Point line1 { 0, yp + textPaddingTop + lineHeight };
			PrintSBookStr(out, line0, pgettext("spell", GetSpellData(sn).sNameText));
			switch (GetSBookTrans(sn, false)) {
			case SpellType::Skill:
				PrintSBookStr(out, line1, _("Skill"));
				break;
			case SpellType::Charges: {
				const int charges = player.InvBody[INVLOC_HAND_LEFT]._iCharges;
				PrintSBookStr(out, line1, fmt::format(fmt::runtime(ngettext("Staff ({:d} charge)", "Staff ({:d} charges)", charges)), charges));
			} break;
			default: {
				const int mana = GetManaAmount(player, sn) >> 6;
				const int lvl = player.GetSpellLevel(sn);
				PrintSBookStr(out, line0, fmt::format(fmt::runtime(pgettext(/* TRANSLATORS: UI constraints, keep short please.*/ "spellbook", "Level {:d}")), lvl), UiFlags::AlignRight);
				if (const StringOrView text = GetSpellPowerText(sn, lvl); !text.empty()) {
					PrintSBookStr(out, line1, text, UiFlags::AlignRight);
				}
				PrintSBookStr(out, line1, fmt::format(fmt::runtime(pgettext(/* TRANSLATORS: UI constraints, keep short please.*/ "spellbook", "Mana: {:d}")), mana));
			} break;
			}
		}
		yp += SpellBookDescription.height;
	}
}

void CheckSBook()
{
	// Icons are drawn in a column near the left side of the panel and aligned with the spell book description entries
	// Spell icons/buttons are 37x38 pixels, laid out from 11,18 with a 5 pixel margin between each icon. This is close
	// enough to the height of the space given to spell descriptions that we can reuse that value and subtract the
	// padding from the end of the area.
	Rectangle iconArea = { GetPanelPosition(UiPanels::Spell, { 11, 18 }), Size { 37, SpellBookDescription.height * 7 - 5 } };
	if (iconArea.contains(MousePosition) && !IsInspectingPlayer()) {
		SpellID sn = GetSpellFromSpellPage(sbooktab, (MousePosition.y - iconArea.position.y) / SpellBookDescription.height);
		Player &player = *InspectPlayer;
		uint64_t spl = player._pMemSpells | player._pISpells | player._pAblSpells;
		if (IsValidSpell(sn) && (spl & GetSpellBitmask(sn)) != 0) {
			SpellType st = SpellType::Spell;
			if ((player._pISpells & GetSpellBitmask(sn)) != 0) {
				st = SpellType::Charges;
			}
			if ((player._pAblSpells & GetSpellBitmask(sn)) != 0) {
				st = SpellType::Skill;
			}
			player._pRSpell = sn;
			player._pRSplType = st;
			RedrawEverything();
		}
		return;
	}

	// The width of the panel excluding the border is 305 pixels. This does not cleanly divide by 4 meaning Diablo tabs
	// end up with an extra pixel somewhere around the buttons. Vanilla Diablo had the buttons left-aligned, devilutionX
	// instead justifies the buttons and puts the gap between buttons 2/3. See DrawSpellBook
	const int buttonWidth = SpellBookButtonWidth();
	// Tabs are drawn in a row near the bottom of the panel
	Rectangle tabArea = { GetPanelPosition(UiPanels::Spell, { 7, 320 }), Size { 305, 29 } };
	if (tabArea.contains(MousePosition)) {
		int hitColumn = MousePosition.x - tabArea.position.x;
		// Clicking on the gutter currently activates tab 3. Could make it do nothing by checking for == here and return early.
		if (!gbIsHellfire && hitColumn > buttonWidth * 2) {
			// Subtract 1 pixel to account for the gutter between buttons 2/3
			hitColumn--;
		}
		sbooktab = hitColumn / buttonWidth;
	}
}

} // namespace devilution
