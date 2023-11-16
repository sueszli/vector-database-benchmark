/*
 * xmlwin.c
 * vim: expandtab:ts=4:sts=4:sw=4
 *
 * Copyright (C) 2012 - 2019 James Booth <boothj5@gmail.com>
 *
 * This file is part of Profanity.
 *
 * Profanity is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Profanity is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Profanity.  If not, see <https://www.gnu.org/licenses/>.
 *
 * In addition, as a special exception, the copyright holders give permission to
 * link the code of portions of this program with the OpenSSL library under
 * certain conditions as described in each individual source file, and
 * distribute linked combinations including the two.
 *
 * You must obey the GNU General Public License in all respects for all of the
 * code used other than OpenSSL. If you modify file(s) with this exception, you
 * may extend this exception to your version of the file(s), but you are not
 * obligated to do so. If you do not wish to do so, delete this exception
 * statement from your version. If you delete this exception statement from all
 * source files in the program, then also delete it here.
 *
 */

#include "config.h"

#include <assert.h>
#include <string.h>

#include "ui/win_types.h"
#include "ui/window_list.h"

void
xmlwin_show(ProfXMLWin* xmlwin, const char* const msg)
{
    assert(xmlwin != NULL);

    ProfWin* window = (ProfWin*)xmlwin;
    if (g_str_has_prefix(msg, "SENT:")) {
        win_println(window, THEME_DEFAULT, "-", "SENT:");
        win_println(window, THEME_ONLINE, "-", "%s", &msg[6]);
        win_println(window, THEME_ONLINE, "-", "");
    } else if (g_str_has_prefix(msg, "RECV:")) {
        win_println(window, THEME_DEFAULT, "-", "RECV:");
        win_println(window, THEME_AWAY, "-", "%s", &msg[6]);
        win_println(window, THEME_AWAY, "-", "");
    }
}

gchar*
xmlwin_get_string(ProfXMLWin* xmlwin)
{
    assert(xmlwin != NULL);

    return g_strdup("XML console");
}
