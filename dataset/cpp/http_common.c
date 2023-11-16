/*
 * http_common.c
 * vim: expandtab:ts=4:sts=4:sw=4
 *
 * Copyright (C) 2020 William Wennerström <william@wstrm.dev>
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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <gio/gio.h>

#include "tools/http_common.h"

#define FALLBACK_MSG ""

void
http_print_transfer_update(ProfWin* window, char* id, const char* fmt, ...)
{
    va_list args;

    va_start(args, fmt);
    GString* msg = g_string_new(FALLBACK_MSG);
    g_string_vprintf(msg, fmt, args);
    va_end(args);

    if (window->type != WIN_CONSOLE) {
        win_update_entry_message(window, id, msg->str);
    } else {
        cons_show("%s", msg->str);
    }

    g_string_free(msg, TRUE);
}

void
http_print_transfer(ProfWin* window, char* id, const char* fmt, ...)
{
    va_list args;

    va_start(args, fmt);
    GString* msg = g_string_new(FALLBACK_MSG);
    g_string_vprintf(msg, fmt, args);
    va_end(args);

    win_print_http_transfer(window, msg->str, id);

    g_string_free(msg, TRUE);
}
