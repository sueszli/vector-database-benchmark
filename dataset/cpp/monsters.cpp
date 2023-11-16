#ifdef _DEBUG
#include "lua/modules/dev/monsters.hpp"

#include <optional>
#include <string>

#include <sol/sol.hpp>

#include "levels/gendung.h"
#include "lighting.h"
#include "lua/metadoc.hpp"
#include "monstdat.h"
#include "monster.h"
#include "player.h"
#include "utils/str_case.hpp"
#include "utils/str_cat.hpp"

namespace devilution {

namespace {

std::string DebugCmdSpawnUniqueMonster(std::string name, std::optional<unsigned> countOpt)
{
	if (leveltype == DTYPE_TOWN) return "Can't spawn monsters in town";
	if (name.empty()) return "name is required";
	const unsigned count = countOpt.value_or(1);
	if (count < 1) return "count must be positive";

	AsciiStrToLower(name);

	int mtype = -1;
	UniqueMonsterType uniqueIndex = UniqueMonsterType::None;
	for (size_t i = 0; UniqueMonstersData.size(); ++i) {
		auto mondata = UniqueMonstersData[i];
		const std::string monsterName = AsciiStrToLower(std::string_view(mondata.mName));
		if (monsterName.find(name) == std::string::npos)
			continue;
		mtype = mondata.mtype;
		uniqueIndex = static_cast<UniqueMonsterType>(i);
		if (monsterName == name) // to support partial name matching but always choose the correct monster if full name is given
			break;
	}

	if (mtype == -1) return "Monster not found";

	size_t id = MaxLvlMTypes - 1;
	bool found = false;

	for (size_t i = 0; i < LevelMonsterTypeCount; i++) {
		if (LevelMonsterTypes[i].type == mtype) {
			id = i;
			found = true;
			break;
		}
	}

	if (!found) {
		CMonster &monsterType = LevelMonsterTypes[id];
		monsterType.type = static_cast<_monster_id>(mtype);
		InitMonsterGFX(monsterType);
		InitMonsterSND(monsterType);
		monsterType.placeFlags |= PLACE_SCATTER;
		monsterType.corpseId = 1;
	}

	Player &myPlayer = *MyPlayer;

	unsigned spawnedMonster = 0;

	auto ret = Crawl(0, MaxCrawlRadius, [&](Displacement displacement) -> std::optional<std::string> {
		Point pos = myPlayer.position.tile + displacement;
		if (dPlayer[pos.x][pos.y] != 0 || dMonster[pos.x][pos.y] != 0)
			return {};
		if (!IsTileWalkable(pos))
			return {};

		Monster *monster = AddMonster(pos, myPlayer._pdir, id, true);
		if (monster == nullptr)
			return StrCat("Spawned ", spawnedMonster, " monsters. (Unable to spawn more)");
		PrepareUniqueMonst(*monster, uniqueIndex, 0, 0, UniqueMonstersData[static_cast<size_t>(uniqueIndex)]);
		monster->corpseId = 1;
		spawnedMonster += 1;

		if (spawnedMonster >= count)
			return StrCat("Spawned ", spawnedMonster, " monsters.");

		return {};
	});

	if (!ret.has_value())
		ret = StrCat("Spawned ", spawnedMonster, " monsters. (Unable to spawn more)");
	return *ret;
}

std::string DebugCmdSpawnMonster(std::string name, std::optional<unsigned> countOpt)
{
	if (leveltype == DTYPE_TOWN) return "Can't spawn monsters in town";
	if (name.empty()) return "name is required";
	const unsigned count = countOpt.value_or(1);
	if (count < 1) return "count must be positive";

	AsciiStrToLower(name);

	int mtype = -1;

	for (int i = 0; i < NUM_MTYPES; i++) {
		auto mondata = MonstersData[i];
		const std::string monsterName = AsciiStrToLower(std::string_view(mondata.name));
		if (monsterName.find(name) == std::string::npos)
			continue;
		mtype = i;
		if (monsterName == name) // to support partial name matching but always choose the correct monster if full name is given
			break;
	}

	if (mtype == -1) return "Monster not found";

	size_t id = MaxLvlMTypes - 1;
	bool found = false;

	for (size_t i = 0; i < LevelMonsterTypeCount; i++) {
		if (LevelMonsterTypes[i].type == mtype) {
			id = i;
			found = true;
			break;
		}
	}

	if (!found) {
		CMonster &monsterType = LevelMonsterTypes[id];
		monsterType.type = static_cast<_monster_id>(mtype);
		InitMonsterGFX(monsterType);
		InitMonsterSND(monsterType);
		monsterType.placeFlags |= PLACE_SCATTER;
		monsterType.corpseId = 1;
	}

	Player &myPlayer = *MyPlayer;

	unsigned spawnedMonster = 0;

	auto ret = Crawl(0, MaxCrawlRadius, [&](Displacement displacement) -> std::optional<std::string> {
		Point pos = myPlayer.position.tile + displacement;
		if (dPlayer[pos.x][pos.y] != 0 || dMonster[pos.x][pos.y] != 0)
			return {};
		if (!IsTileWalkable(pos))
			return {};

		if (AddMonster(pos, myPlayer._pdir, id, true) == nullptr)
			return StrCat("Spawned ", spawnedMonster, " monsters. (Unable to spawn more)");
		spawnedMonster += 1;

		if (spawnedMonster >= count)
			return StrCat("Spawned ", spawnedMonster, " monsters.");

		return {};
	});

	if (!ret.has_value())
		return StrCat("Spawned ", spawnedMonster, " monsters. (Unable to spawn more)");
	return *ret;
}

} // namespace

sol::table LuaDevMonstersModule(sol::state_view &lua)
{
	sol::table table = lua.create_table();
	SetDocumented(table, "spawn", "(name: string, count: number = 1)", "Spawn monster(s)", &DebugCmdSpawnMonster);
	SetDocumented(table, "spawnUnique", "(name: string, count: number = 1)", "Spawn unique monster(s)", &DebugCmdSpawnUniqueMonster);
	return table;
}

} // namespace devilution
#endif // _DEBUG
