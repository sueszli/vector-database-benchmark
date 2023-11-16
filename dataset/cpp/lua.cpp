#include "lua/lua.hpp"

#include <optional>
#include <string_view>
#include <unordered_map>

#include <sol/sol.hpp>

#include <config.h>

#include "appfat.h"
#include "engine/assets.hpp"
#include "lua/modules/audio.hpp"
#include "lua/modules/log.hpp"
#include "lua/modules/render.hpp"
#include "plrmsg.h"
#include "utils/console.h"
#include "utils/log.hpp"
#include "utils/str_cat.hpp"

#ifdef _DEBUG
#include "lua/modules/dev.hpp"
#include "lua/repl.hpp"
#endif

namespace devilution {

namespace {

struct LuaState {
	sol::state sol = {};
	sol::table commonPackages = {};
	std::unordered_map<std::string, sol::bytecode> compiledScripts = {};
	sol::environment sandbox = {};
	sol::table events = {};
};

std::optional<LuaState> CurrentLuaState;

// A Lua function that we use to generate a `require` implementation.
constexpr std::string_view RequireGenSrc = R"lua(
function requireGen(env, loaded, loadFn)
  return function(packageName)
      local p = loaded[packageName]
      if p == nil then
          local loader = loadFn(packageName)
          setEnvironment(loader, env)
          if type(loader) == "string" then
            error(loader)
          end
          p = loader(packageName)
          loaded[packageName] = p
      end
      return p
  end
end
)lua";

sol::object LuaLoadScriptFromAssets(std::string_view packageName)
{
	LuaState &luaState = *CurrentLuaState;
	constexpr std::string_view PathPrefix = "lua\\";
	constexpr std::string_view PathSuffix = ".lua";
	std::string path;
	path.reserve(PathPrefix.size() + packageName.size() + PathSuffix.size());
	StrAppend(path, PathPrefix, packageName, PathSuffix);
	std::replace(path.begin() + PathPrefix.size(), path.end() - PathSuffix.size(), '.', '\\');

	auto iter = luaState.compiledScripts.find(path);
	if (iter != luaState.compiledScripts.end()) {
		return luaState.sol.load(iter->second.as_string_view(), path, sol::load_mode::binary);
	}

	tl::expected<AssetData, std::string> assetData = LoadAsset(path);
	if (!assetData.has_value()) {
		sol::stack::push(luaState.sol.lua_state(), assetData.error());
		return sol::stack_object(luaState.sol.lua_state(), -1);
	}
	sol::load_result result = luaState.sol.load(std::string_view(*assetData), path, sol::load_mode::text);
	if (!result.valid()) {
		sol::stack::push(luaState.sol.lua_state(),
		    StrCat("Lua error when loading ", path, ": ", result.get<std::string>()));
		return sol::stack_object(luaState.sol.lua_state(), -1);
	}
	const sol::function fn = result;
	luaState.compiledScripts[path] = fn.dump();
	return result;
}

int LuaPrint(lua_State *state)
{
	const int n = lua_gettop(state);
	for (int i = 1; i <= n; i++) {
		size_t l;
		const char *s = luaL_tolstring(state, i, &l);
		if (i > 1)
			printInConsole("\t");
		printInConsole(std::string_view(s, l));
		lua_pop(state, 1);
	}
	printNewlineInConsole();
	return 0;
}

void LuaWarn(void *userData, const char *message, int continued)
{
	static std::string warnBuffer;
	warnBuffer.append(message);
	if (continued != 0)
		return;
	LogWarn("{}", warnBuffer);
	warnBuffer.clear();
}

sol::object RunScript(std::optional<sol::environment> env, std::string_view packageName, bool optional)
{
	sol::object result = LuaLoadScriptFromAssets(packageName);
	// We return a string on error:
	if (result.get_type() == sol::type::string) {
		if (!optional)
			app_fatal(result.as<std::string>());
		LogError("{}", result.as<std::string>());
		return sol::lua_nil;
	}
	auto fn = result.as<sol::protected_function>();
	if (env.has_value()) {
		sol::set_environment(*env, fn);
	}
	return SafeCallResult(fn(), optional);
}

void LuaPanic(sol::optional<std::string> message)
{
	LogError("Lua is in a panic state and will now abort() the application:\n{}",
	    message.value_or("unknown error"));
}

} // namespace

void Sol2DebugPrintStack(lua_State *state)
{
	LogDebug("{}", sol::detail::debug::dump_types(state));
}

void Sol2DebugPrintSection(const std::string &message, lua_State *state)
{
	LogDebug("-- {} -- [ {} ]", message, sol::detail::debug::dump_types(state));
}

sol::environment CreateLuaSandbox()
{
	sol::state &lua = CurrentLuaState->sol;
	sol::environment sandbox(CurrentLuaState->sol, sol::create);

	// Registering globals
	sandbox.set(
	    "print", LuaPrint,
	    "_DEBUG",
#ifdef _DEBUG
	    true,
#else
	    false,
#endif
	    "_VERSION", LUA_VERSION);

	// Register safe built-in globals.
	for (const std::string_view global : {
	         // Built-ins:
	         "assert", "warn", "error", "ipairs", "next", "pairs", "pcall",
	         "select", "tonumber", "tostring", "type", "xpcall",
	         "rawequal", "rawget", "rawset", "setmetatable",
	// Built-in packages:
#ifdef _DEBUG
	         "debug",
#endif
	         "base", "coroutine", "table", "string", "math", "utf8" }) {
		const sol::object obj = lua[global];
		if (obj.get_type() == sol::type::lua_nil) {
			app_fatal(StrCat("Missing Lua global [", global, "]"));
		}
		sandbox[global] = obj;
	}

	// We only allow datetime-related functions from `os`:
	const sol::table os = lua["os"];
	sandbox.create_named("os",
	    "date", os["date"],
	    "difftime", os["difftime"],
	    "time", os["time"]);

	sandbox["require"] = lua["requireGen"](sandbox, CurrentLuaState->commonPackages, LuaLoadScriptFromAssets);

	return sandbox;
}

void LuaInitialize()
{
	CurrentLuaState.emplace(LuaState { .sol = { sol::c_call<decltype(&LuaPanic), &LuaPanic> } });
	sol::state &lua = CurrentLuaState->sol;
	lua_setwarnf(lua.lua_state(), LuaWarn, /*ud=*/nullptr);
	lua.open_libraries(
	    sol::lib::base,
	    sol::lib::coroutine,
	    sol::lib::debug,
	    sol::lib::math,
	    sol::lib::os,
	    sol::lib::package,
	    sol::lib::string,
	    sol::lib::table,
	    sol::lib::utf8);

	// Registering devilutionx object table
	SafeCallResult(lua.safe_script(RequireGenSrc), /*optional=*/false);

	// Loaded without a sandbox.
	CurrentLuaState->events = RunScript(/*env=*/std::nullopt, "devilutionx.events", /*optional=*/false);

	CurrentLuaState->commonPackages = lua.create_table_with(
#ifdef _DEBUG
	    "devilutionx.dev", LuaDevModule(lua),
#endif
	    "devilutionx.version", PROJECT_VERSION,
	    "devilutionx.log", LuaLogModule(lua),
	    "devilutionx.audio", LuaAudioModule(lua),
	    "devilutionx.render", LuaRenderModule(lua),
	    "devilutionx.message", [](std::string_view text) { EventPlrMsg(text, UiFlags::ColorRed); },
	    // These packages are loaded without a sandbox:
	    "devilutionx.events", CurrentLuaState->events,
	    "inspect", RunScript(/*env=*/std::nullopt, "inspect", /*optional=*/false));

	// Used by the custom require implementation.
	lua["setEnvironment"] = [](const sol::environment &env, const sol::function &fn) { sol::set_environment(env, fn); };

	RunScript(CreateLuaSandbox(), "user", /*optional=*/true);

	LuaEvent("GameBoot");
}

void LuaShutdown()
{
#ifdef _DEBUG
	LuaReplShutdown();
#endif
	CurrentLuaState = std::nullopt;
}

void LuaEvent(std::string_view name)
{
	const auto trigger = CurrentLuaState->events.traverse_get<std::optional<sol::object>>(name, "trigger");
	if (!trigger.has_value() || !trigger->is<sol::protected_function>()) {
		LogError("events.{}.trigger is not a function", name);
		return;
	}
	const sol::protected_function fn = trigger->as<sol::protected_function>();
	SafeCallResult(fn(), /*optional=*/true);
}

sol::state &GetLuaState()
{
	return CurrentLuaState->sol;
}

sol::object SafeCallResult(sol::protected_function_result result, bool optional)
{
	const bool valid = result.valid();
	if (!valid) {
		const std::string error = result.get_type() == sol::type::string
		    ? StrCat("Lua error: ", result.get<std::string>())
		    : "Unknown Lua error";
		if (!optional)
			app_fatal(error);
		LogError(error);
	}
	return result;
}

} // namespace devilution
