/* vifm
 * Copyright (C) 2015 xaizek.
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

#include "cabbrevs_menu.h"

#include <stddef.h> /* wchar_t */
#include <string.h> /* strdup() */
#include <wchar.h> /* wcslen() */

#include "../engine/abbrevs.h"
#include "../engine/text_buffer.h"
#include "../modes/menu.h"
#include "../ui/ui.h"
#include "../utils/string_array.h"
#include "../utils/str.h"
#include "../bracket_notation.h"
#include "menus.h"

static KHandlerResponse commands_khandler(view_t *view, menu_data_t *m,
		const wchar_t keys[]);

int
show_cabbrevs_menu(view_t *view)
{
	void *state;
	const wchar_t *lhs, *rhs;
	const char *descr;
	int no_remap;

	static menu_data_t m;
	menus_init_data(&m, view,
			strdup("Abbreviation -- N -- Expansion/Description"),
			strdup("No abbreviations set"));
	m.key_handler = &commands_khandler;

	state = NULL;
	while(vle_abbr_iter(&lhs, &rhs, &descr, &no_remap, &state))
	{
		char *const line = describe_abbrev(lhs, rhs, descr, no_remap, /*offset=*/2);
		m.len = put_into_string_array(&m.items, m.len, line);
	}

	return menus_enter(&m, view);
}

/* Menu-specific shortcut handler.  Returns code that specifies both taken
 * actions and what should be done next. */
static KHandlerResponse
commands_khandler(view_t *view, menu_data_t *m, const wchar_t keys[])
{
	if(wcscmp(keys, L"dd") == 0) /* Remove element. */
	{
		char cmd_buf[512];

		break_at(m->items[m->pos], ' ');
		snprintf(cmd_buf, sizeof(cmd_buf), "cunabbrev %s", m->items[m->pos]);
		modmenu_run_command(cmd_buf);

		menus_remove_current(m->state);
		return KHR_REFRESH_WINDOW;
	}
	return KHR_UNHANDLED;
}

char *
describe_abbrev(const wchar_t lhs[], const wchar_t rhs[], const char descr[],
		int no_remap, int offset)
{
	enum { LHS_MIN_WIDTH = 13 };
	const char map_mark = no_remap ? '*' : ' ';

	char *rhs_descr = vle_abbr_describe(rhs, descr);
	char *const line = format_str("%-*ls %3c    %s", offset + LHS_MIN_WIDTH, lhs,
			map_mark, rhs_descr);
	free(rhs_descr);

	return line;
}

/* vim: set tabstop=2 softtabstop=2 shiftwidth=2 noexpandtab cinoptions-=(0 : */
/* vim: set cinoptions+=t0 filetype=c : */
