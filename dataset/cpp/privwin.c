/*
 * privwin.c
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
#include <glib.h>
#include <stdlib.h>

#include "log.h"
#include "config/preferences.h"
#include "ui/win_types.h"
#include "ui/window.h"
#include "ui/titlebar.h"
#include "ui/window_list.h"

void
privwin_incoming_msg(ProfPrivateWin* privatewin, ProfMessage* message)
{
    assert(privatewin != NULL);

    if (message->plain == NULL) {
        log_error("privwin_incoming_msg: Message with no plain field from: %s", message->from_jid);
        return;
    }

    ProfWin* window = (ProfWin*)privatewin;
    int num = wins_get_num(window);

    auto_jid Jid* jidp = jid_create(privatewin->fulljid);
    if (jidp == NULL) {
        return;
    }

    gboolean is_current = wins_is_current(window);
    gboolean notify = prefs_do_chat_notify(is_current);

    // currently viewing chat window with sender
    if (wins_is_current(window)) {
        win_print_incoming(window, jidp->resourcepart, message);
        title_bar_set_typing(FALSE);
        status_bar_active(num, WIN_PRIVATE, privatewin->fulljid);

        // not currently viewing chat window with sender
    } else {
        status_bar_new(num, WIN_PRIVATE, privatewin->fulljid);
        cons_show_incoming_private_message(jidp->resourcepart, jidp->barejid, num, privatewin->unread, window);
        win_insert_last_read_position_marker((ProfWin*)privatewin, privatewin->fulljid);
        win_print_incoming(window, jidp->resourcepart, message);

        privatewin->unread++;

        if (prefs_get_boolean(PREF_FLASH)) {
            flash();
        }
    }

    wins_add_urls_ac(window, message, FALSE);
    wins_add_quotes_ac(window, message->plain, TRUE);

    if (prefs_get_boolean(PREF_BEEP)) {
        beep();
    }

    if (notify) {
        notify_message(jidp->resourcepart, num, message->plain);
    }
}

void
privwin_outgoing_msg(ProfPrivateWin* privwin, const char* const message)
{
    assert(privwin != NULL);

    ProfWin* window = (ProfWin*)privwin;
    wins_add_quotes_ac(window, message, FALSE);
    win_print_outgoing((ProfWin*)privwin, "-", NULL, NULL, message);
}

void
privwin_message_occupant_offline(ProfPrivateWin* privwin)
{
    assert(privwin != NULL);

    win_println((ProfWin*)privwin, THEME_ERROR, "-", "Unable to send message, occupant no longer present in room.");
}

void
privwin_message_left_room(ProfPrivateWin* privwin)
{
    assert(privwin != NULL);

    win_println((ProfWin*)privwin, THEME_ERROR, "-", "Unable to send message, you are no longer present in room.");
}

void
privwin_occupant_offline(ProfPrivateWin* privwin)
{
    assert(privwin != NULL);

    privwin->occupant_offline = TRUE;
    auto_jid Jid* jidp = jid_create(privwin->fulljid);
    win_println((ProfWin*)privwin, THEME_OFFLINE, "-", "<- %s has left the room.", jidp->resourcepart);
}

void
privwin_occupant_kicked(ProfPrivateWin* privwin, const char* const actor, const char* const reason)
{
    assert(privwin != NULL);

    privwin->occupant_offline = TRUE;
    auto_jid Jid* jidp = jid_create(privwin->fulljid);
    GString* message = g_string_new(jidp->resourcepart);
    g_string_append(message, " has been kicked from the room");
    if (actor) {
        g_string_append(message, " by ");
        g_string_append(message, actor);
    }
    if (reason) {
        g_string_append(message, ", reason: ");
        g_string_append(message, reason);
    }

    win_println((ProfWin*)privwin, THEME_OFFLINE, "!", "<- %s", message->str);
    g_string_free(message, TRUE);
}

void
privwin_occupant_banned(ProfPrivateWin* privwin, const char* const actor, const char* const reason)
{
    assert(privwin != NULL);

    privwin->occupant_offline = TRUE;
    auto_jid Jid* jidp = jid_create(privwin->fulljid);
    GString* message = g_string_new(jidp->resourcepart);
    g_string_append(message, " has been banned from the room");
    if (actor) {
        g_string_append(message, " by ");
        g_string_append(message, actor);
    }
    if (reason) {
        g_string_append(message, ", reason: ");
        g_string_append(message, reason);
    }

    win_println((ProfWin*)privwin, THEME_OFFLINE, "!", "<- %s", message->str);
    g_string_free(message, TRUE);
}

void
privwin_occupant_online(ProfPrivateWin* privwin)
{
    assert(privwin != NULL);

    privwin->occupant_offline = FALSE;
    auto_jid Jid* jidp = jid_create(privwin->fulljid);
    win_println((ProfWin*)privwin, THEME_ONLINE, "-", "-- %s has joined the room.", jidp->resourcepart);
}

void
privwin_room_destroyed(ProfPrivateWin* privwin)
{
    assert(privwin != NULL);

    privwin->room_left = TRUE;
    auto_jid Jid* jidp = jid_create(privwin->fulljid);
    win_println((ProfWin*)privwin, THEME_OFFLINE, "!", "-- %s has been destroyed.", jidp->barejid);
}

void
privwin_room_joined(ProfPrivateWin* privwin)
{
    assert(privwin != NULL);

    privwin->room_left = FALSE;
    auto_jid Jid* jidp = jid_create(privwin->fulljid);
    win_println((ProfWin*)privwin, THEME_OFFLINE, "!", "-- You have joined %s.", jidp->barejid);
}

void
privwin_room_left(ProfPrivateWin* privwin)
{
    assert(privwin != NULL);

    privwin->room_left = TRUE;
    auto_jid Jid* jidp = jid_create(privwin->fulljid);
    win_println((ProfWin*)privwin, THEME_OFFLINE, "!", "-- You have left %s.", jidp->barejid);
}

void
privwin_room_kicked(ProfPrivateWin* privwin, const char* const actor, const char* const reason)
{
    assert(privwin != NULL);

    privwin->room_left = TRUE;
    GString* message = g_string_new("Kicked from ");
    auto_jid Jid* jidp = jid_create(privwin->fulljid);
    g_string_append(message, jidp->barejid);
    if (actor) {
        g_string_append(message, " by ");
        g_string_append(message, actor);
    }
    if (reason) {
        g_string_append(message, ", reason: ");
        g_string_append(message, reason);
    }

    win_println((ProfWin*)privwin, THEME_OFFLINE, "!", "<- %s", message->str);
    g_string_free(message, TRUE);
}

void
privwin_room_banned(ProfPrivateWin* privwin, const char* const actor, const char* const reason)
{
    assert(privwin != NULL);

    privwin->room_left = TRUE;
    GString* message = g_string_new("Banned from ");
    auto_jid Jid* jidp = jid_create(privwin->fulljid);
    g_string_append(message, jidp->barejid);
    if (actor) {
        g_string_append(message, " by ");
        g_string_append(message, actor);
    }
    if (reason) {
        g_string_append(message, ", reason: ");
        g_string_append(message, reason);
    }

    win_println((ProfWin*)privwin, THEME_OFFLINE, "!", "<- %s", message->str);
    g_string_free(message, TRUE);
}

gchar*
privwin_get_string(ProfPrivateWin* privwin)
{
    assert(privwin != NULL);

    GString* res = g_string_new("Private ");
    g_string_append(res, privwin->fulljid);

    if (privwin->unread > 0) {
        g_string_append_printf(res, ", %d unread", privwin->unread);
    }

    return g_string_free(res, FALSE);
}
