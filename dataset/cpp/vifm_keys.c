/* vifm
 * Copyright (C) 2021 xaizek.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
 */

#include "vifm_keys.h"

#include <stdlib.h> /* free() */
#include <wchar.h>

#include "../engine/keys.h"
#include "../modes/modes.h"
#include "../ui/statusbar.h"
#include "../ui/ui.h"
#include "../utils/macros.h"
#include "../utils/str.h"
#include "../bracket_notation.h"
#include "../flist_pos.h"
#include "../status.h"
#include "lua/lauxlib.h"
#include "lua/lua.h"
#include "api.h"
#include "common.h"
#include "vlua_state.h"

static int VLUA_API(keys_add)(lua_State *lua);

VLUA_DECLARE_SAFE(keys_add);

static void parse_modes(vlua_t *vlua, char modes[MODES_COUNT]);
static void lua_key_handler(key_info_t key_info, keys_info_t *keys_info);
static void build_handler_args(lua_State *lua, key_info_t key_info,
		const keys_info_t *keys_info);

/* Functions of `vifm.keys` table. */
static const luaL_Reg vifm_keys_methods[] = {
	{ "add", VLUA_REF(keys_add) },
	{ NULL,  NULL               }
};

void
vifm_keys_init(lua_State *lua)
{
	luaL_newlib(lua, vifm_keys_methods);
}

/* Member of `vifm.keys` that registers a new key or raises an error.  Returns
 * boolean, which is true on success. */
static int
VLUA_API(keys_add)(lua_State *lua)
{
	vlua_t *vlua = vlua_state_get(lua);

	luaL_checktype(lua, 1, LUA_TTABLE);

	wchar_t lhs[16];
	const size_t max_lhs_len = ARRAY_LEN(lhs) - 1U;
	vlua_cmn_check_field(lua, 1, "shortcut", LUA_TSTRING);
	wchar_t *tmp_lhs = substitute_specs(lua_tostring(lua, -1));
	if(wcslen(tmp_lhs) == 0 || wcslen(tmp_lhs) > max_lhs_len)
	{
		free(tmp_lhs);
		return luaL_error(lua, "Shortcut can't be empty or longer than %d",
				(int)max_lhs_len);
	}

	wcscpy(lhs, tmp_lhs);
	free(tmp_lhs);

	const char *descr = "";
	if(vlua_cmn_check_opt_field(lua, 1, "description", LUA_TSTRING))
	{
		descr = vlua_state_store_string(vlua, lua_tostring(lua, -1));
	}

	int is_selector = 0;
	if(vlua_cmn_check_opt_field(lua, 1, "isselector", LUA_TBOOLEAN))
	{
		is_selector = lua_toboolean(lua, -1);
	}

	FollowedBy followed_by = FOLLOWED_BY_NONE;
	if(vlua_cmn_check_opt_field(lua, 1, "followedby", LUA_TSTRING))
	{
		const char *value = lua_tostring(lua, -1);
		if(strcmp(value, "none") == 0)
		{
			followed_by = FOLLOWED_BY_NONE;
		}
		else if(strcmp(value, "selector") == 0)
		{
			followed_by = FOLLOWED_BY_SELECTOR;
		}
		else if(strcmp(value, "keyarg") == 0)
		{
			followed_by = FOLLOWED_BY_MULTIKEY;
		}
		else
		{
			return luaL_error(lua, "Unrecognized value for `followedby`: %s", value);
		}
	}

	char modes[MODES_COUNT] = { };
	vlua_cmn_check_field(lua, 1, "modes", LUA_TTABLE);
	parse_modes(vlua, modes);

	lua_createtable(lua, /*narr=*/0, /*nrec=*/2);
	vlua_cmn_check_field(lua, 1, "handler", LUA_TFUNCTION);
	lua_setfield(lua, -2, "handler");
	lua_pushboolean(lua, is_selector);
	lua_setfield(lua, -2, "isselector");
	void *handler = vlua_cmn_to_pointer(lua);

	key_conf_t key = {
		.data.handler = &lua_key_handler,
		.descr = descr,
		.followed = followed_by,
	};

	key.user_data = vlua_state_store_pointer(vlua, handler);
	if(key.user_data == NULL)
	{
		return luaL_error(lua, "%s", "Failed to store handler data");
	}

	int success = 1;

	int mode;
	for(mode = 0; mode < MODES_COUNT; ++mode)
	{
		if(modes[mode])
		{
			success &= (vle_keys_foreign_add(lhs, &key, is_selector, mode) == 0);
		}
	}

	lua_pushboolean(lua, success);
	return 1;
}

/* Parses table at the top of the Lua stack into a set of flags indicating
 * which modes were mentioned in the table. */
static void
parse_modes(vlua_t *vlua, char modes[MODES_COUNT])
{
	lua_pushnil(vlua->lua);
	while(lua_next(vlua->lua, -2) != 0)
	{
		const char *name = lua_tostring(vlua->lua, -1);

		if(strcmp(name, "cmdline") == 0)
		{
			modes[CMDLINE_MODE] = 1;
		}
		else if(strcmp(name, "nav") == 0)
		{
			modes[NAV_MODE] = 1;
		}
		else if(strcmp(name, "normal") == 0)
		{
			modes[NORMAL_MODE] = 1;
		}
		else if(strcmp(name, "visual") == 0)
		{
			modes[VISUAL_MODE] = 1;
		}
		else if(strcmp(name, "menus") == 0)
		{
			modes[MENU_MODE] = 1;
		}
		else if(strcmp(name, "view") == 0)
		{
			modes[VIEW_MODE] = 1;
		}
		else if(strcmp(name, "dialogs") == 0)
		{
			modes[SORT_MODE] = 1;
			modes[ATTR_MODE] = 1;
			modes[CHANGE_MODE] = 1;
			modes[FILE_INFO_MODE] = 1;
		}

		lua_pop(vlua->lua, 1);
	}
}

/* Handler of all foreign mappings registered from Lua. */
static void
lua_key_handler(key_info_t key_info, keys_info_t *keys_info)
{
	state_ptr_t *p = key_info.user_data;
	lua_State *lua = p->vlua->lua;

	vlua_cmn_from_pointer(lua, p->ptr);
	lua_getfield(lua, -1, "isselector");
	int is_selector = lua_toboolean(lua, -1);
	lua_getfield(lua, -2, "handler");

	build_handler_args(lua, key_info, keys_info);

	/* After we've built handler arguments, this list isn't needed anymore. */
	free(keys_info->indexes);
	keys_info->indexes = NULL;
	keys_info->count = 0;

	curr_stats.save_msg = 0;

	int sm_cookie;
	if(is_selector)
	{
		sm_cookie = vlua_state_safe_mode_on(lua);
	}

	int result = lua_pcall(lua, 1, 1, 0);

	if(is_selector)
	{
		vlua_state_safe_mode_off(lua, sm_cookie);
	}

	if(result != LUA_OK)
	{
		ui_sb_err(lua_tostring(lua, -1));
		lua_pop(lua, 3);
		curr_stats.save_msg = 1;
		return;
	}

	if(is_selector && lua_istable(lua, -1))
	{
		if(vlua_cmn_extract_indexes(lua, curr_view, &keys_info->count,
					&keys_info->indexes) == 0)
		{
			if(keys_info->count == 0)
			{
				free(keys_info->indexes);
				keys_info->indexes = NULL;
			}
		}

		if(lua_getfield(lua, -1, "cursorpos") == LUA_TNUMBER)
		{
			const int pos = lua_tointeger(lua, -1) - 1;
			if(pos >= 0)
			{
				fpos_set_pos(curr_view, pos);
			}
		}
		lua_pop(lua, 1);
	}

	lua_pop(lua, 3);
}

/* Builds table passed to key handler, leaves it at the top of the stack. */
static void
build_handler_args(lua_State *lua, key_info_t key_info,
		const keys_info_t *keys_info)
{
	lua_createtable(lua, /*narr=*/0, /*nrec=*/3);

	if(key_info.count == NO_COUNT_GIVEN)
	{
		lua_pushnil(lua);
	}
	else
	{
		lua_pushinteger(lua, key_info.count);
	}
	lua_setfield(lua, -2, "count");

	if(key_info.reg == NO_REG_GIVEN)
	{
		lua_pushnil(lua);
	}
	else
	{
		const char reg_name[] = { key_info.reg, '\0' };
		lua_pushstring(lua, reg_name);
	}
	lua_setfield(lua, -2, "register");

	if(keys_info->selector)
	{
		int i;
		lua_createtable(lua, keys_info->count, /*nrec=*/0);
		for(i = 0; i < keys_info->count; ++i)
		{
			lua_pushinteger(lua, keys_info->indexes[i] + 1);
			lua_seti(lua, -2, i + 1);
		}
		lua_setfield(lua, -2, "indexes");
	}

	if(key_info.multi != L'\0')
	{
		lua_pushfstring(lua, "%U", key_info.multi);
		lua_setfield(lua, -2, "keyarg");
	}
}

/* vim: set tabstop=2 softtabstop=2 shiftwidth=2 noexpandtab cinoptions-=(0 : */
/* vim: set cinoptions+=t0 : */
