/*
 * iq.c
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

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <glib.h>

#include <strophe.h>

#include "profanity.h"
#include "log.h"
#include "config/preferences.h"
#include "event/server_events.h"
#include "plugins/plugins.h"
#include "tools/http_upload.h"
#include "ui/ui.h"
#include "ui/window_list.h"
#include "xmpp/xmpp.h"
#include "xmpp/connection.h"
#include "xmpp/session.h"
#include "xmpp/iq.h"
#include "xmpp/capabilities.h"
#include "xmpp/blocking.h"
#include "xmpp/session.h"
#include "xmpp/stanza.h"
#include "xmpp/form.h"
#include "xmpp/roster_list.h"
#include "xmpp/roster.h"
#include "xmpp/muc.h"
#include "src/database.h"
#include "ui/window.h"

#ifdef HAVE_OMEMO
#include "omemo/omemo.h"
#endif

typedef struct p_room_info_data_t
{
    char* room;
    gboolean display;
} ProfRoomInfoData;

typedef struct p_iq_handle_t
{
    ProfIqCallback func;
    ProfIqFreeCallback free_func;
    void* userdata;
} ProfIqHandler;

typedef struct privilege_set_t
{
    char* item;
    char* privilege;
} ProfPrivilegeSet;

typedef struct affiliation_list_t
{
    char* affiliation;
    bool show_ui_message;
} ProfAffiliationList;

typedef struct command_config_data_t
{
    char* sessionid;
    char* command;
} CommandConfigData;

typedef struct mam_rsm_userdata
{
    char* barejid;
    char* start_datestr;
    char* end_datestr;
    gboolean fetch_next;
    ProfChatWin* win;
} MamRsmUserdata;

typedef struct late_delivery_userdata
{
    ProfChatWin* win;
    GDateTime* enddate;
    GDateTime* startdate;
} LateDeliveryUserdata;

static int _iq_handler(xmpp_conn_t* const conn, xmpp_stanza_t* const stanza, void* const userdata);

static void _error_handler(xmpp_stanza_t* const stanza);
static void _disco_info_get_handler(xmpp_stanza_t* const stanza);
static void _disco_items_get_handler(xmpp_stanza_t* const stanza);
static void _disco_items_result_handler(xmpp_stanza_t* const stanza);
static void _last_activity_get_handler(xmpp_stanza_t* const stanza);
static void _version_get_handler(xmpp_stanza_t* const stanza);
static void _ping_get_handler(xmpp_stanza_t* const stanza);

static int _version_result_id_handler(xmpp_stanza_t* const stanza, void* const userdata);
static int _disco_info_response_id_handler(xmpp_stanza_t* const stanza, void* const userdata);
static int _disco_info_response_id_handler_onconnect(xmpp_stanza_t* const stanza, void* const userdata);
static int _http_upload_response_id_handler(xmpp_stanza_t* const stanza, void* const upload_ctx);
static int _last_activity_response_id_handler(xmpp_stanza_t* const stanza, void* const userdata);
static int _room_info_response_id_handler(xmpp_stanza_t* const stanza, void* const userdata);
static int _destroy_room_result_id_handler(xmpp_stanza_t* const stanza, void* const userdata);
static int _room_config_id_handler(xmpp_stanza_t* const stanza, void* const userdata);
static int _room_config_submit_id_handler(xmpp_stanza_t* const stanza, void* const userdata);
static int _room_affiliation_list_result_id_handler(xmpp_stanza_t* const stanza, void* const userdata);
static int _room_affiliation_set_result_id_handler(xmpp_stanza_t* const stanza, void* const userdata);
static int _room_role_set_result_id_handler(xmpp_stanza_t* const stanza, void* const userdata);
static int _room_role_list_result_id_handler(xmpp_stanza_t* const stanza, void* const userdata);
static int _room_kick_result_id_handler(xmpp_stanza_t* const stanza, void* const userdata);
static int _enable_carbons_id_handler(xmpp_stanza_t* const stanza, void* const userdata);
static int _disable_carbons_id_handler(xmpp_stanza_t* const stanza, void* const userdata);
static int _manual_pong_id_handler(xmpp_stanza_t* const stanza, void* const userdata);
static int _caps_response_id_handler(xmpp_stanza_t* const stanza, void* const userdata);
static int _caps_response_for_jid_id_handler(xmpp_stanza_t* const stanza, void* const userdata);
static int _caps_response_legacy_id_handler(xmpp_stanza_t* const stanza, void* const userdata);
static int _auto_pong_id_handler(xmpp_stanza_t* const stanza, void* const userdata);
static int _room_list_id_handler(xmpp_stanza_t* const stanza, void* const userdata);
static int _command_list_result_handler(xmpp_stanza_t* const stanza, void* const userdata);
static int _command_exec_response_handler(xmpp_stanza_t* const stanza, void* const userdata);
static int _mam_rsm_id_handler(xmpp_stanza_t* const stanza, void* const userdata);
static int _register_change_password_result_id_handler(xmpp_stanza_t* const stanza, void* const userdata);

static void _iq_mam_request(ProfChatWin* win, GDateTime* startdate, GDateTime* enddate);
static void _iq_free_room_data(ProfRoomInfoData* roominfo);
static void _iq_free_affiliation_set(ProfPrivilegeSet* affiliation_set);
static void _iq_free_affiliation_list(ProfAffiliationList* affiliation_list);
static void _iq_id_handler_free(ProfIqHandler* handler);

// scheduled
static int _autoping_timed_send(xmpp_conn_t* const conn, void* const userdata);

static void _identity_destroy(DiscoIdentity* identity);
static void _item_destroy(DiscoItem* item);

static gboolean autoping_wait = FALSE;
static GTimer* autoping_time = NULL;
static GHashTable* id_handlers;
static GHashTable* rooms_cache = NULL;
static GSList* late_delivery_windows = NULL;
static gboolean received_disco_items = FALSE;

static int
_iq_handler(xmpp_conn_t* const conn, xmpp_stanza_t* const stanza, void* const userdata)
{
    log_debug("iq stanza handler fired");
    autoping_timer_extend();

    char* text;
    size_t text_size;
    xmpp_stanza_to_text(stanza, &text, &text_size);
    gboolean cont = plugins_on_iq_stanza_receive(text);
    xmpp_free(connection_get_ctx(), text);
    if (!cont) {
        return 1;
    }

    const char* type = xmpp_stanza_get_type(stanza);

    if (g_strcmp0(type, STANZA_TYPE_ERROR) == 0) {
        _error_handler(stanza);
    }

    xmpp_stanza_t* discoinfo = xmpp_stanza_get_child_by_ns(stanza, XMPP_NS_DISCO_INFO);
    if (discoinfo && (g_strcmp0(type, STANZA_TYPE_GET) == 0)) {
        _disco_info_get_handler(stanza);
    }

    xmpp_stanza_t* discoitems = xmpp_stanza_get_child_by_ns(stanza, XMPP_NS_DISCO_ITEMS);
    if (discoitems && (g_strcmp0(type, STANZA_TYPE_GET) == 0)) {
        _disco_items_get_handler(stanza);
    }
    if (discoitems && (g_strcmp0(type, STANZA_TYPE_RESULT) == 0)) {
        _disco_items_result_handler(stanza);
    }

    xmpp_stanza_t* lastactivity = xmpp_stanza_get_child_by_ns(stanza, STANZA_NS_LASTACTIVITY);
    if (lastactivity && (g_strcmp0(type, STANZA_TYPE_GET) == 0)) {
        _last_activity_get_handler(stanza);
    }

    xmpp_stanza_t* version = xmpp_stanza_get_child_by_ns(stanza, STANZA_NS_VERSION);
    if (version && (g_strcmp0(type, STANZA_TYPE_GET) == 0)) {
        _version_get_handler(stanza);
    }

    xmpp_stanza_t* ping = xmpp_stanza_get_child_by_ns(stanza, STANZA_NS_PING);
    if (ping && (g_strcmp0(type, STANZA_TYPE_GET) == 0)) {
        _ping_get_handler(stanza);
    }

    xmpp_stanza_t* roster = xmpp_stanza_get_child_by_ns(stanza, XMPP_NS_ROSTER);
    if (roster && (g_strcmp0(type, STANZA_TYPE_SET) == 0)) {
        roster_set_handler(stanza);
    }
    if (roster && (g_strcmp0(type, STANZA_TYPE_RESULT) == 0)) {
        roster_result_handler(stanza);
    }

    xmpp_stanza_t* blocking = xmpp_stanza_get_child_by_ns(stanza, STANZA_NS_BLOCKING);
    if (blocking && (g_strcmp0(type, STANZA_TYPE_SET) == 0)) {
        blocked_set_handler(stanza);
    }

    const char* id = xmpp_stanza_get_id(stanza);
    if (id) {
        ProfIqHandler* handler = g_hash_table_lookup(id_handlers, id);
        if (handler) {
            int keep = handler->func(stanza, handler->userdata);
            if (!keep) {
                g_hash_table_remove(id_handlers, id);
            }
        }
    }

    return 1;
}

void
iq_handlers_init(void)
{
    xmpp_conn_t* const conn = connection_get_conn();
    xmpp_ctx_t* const ctx = connection_get_ctx();
    xmpp_handler_add(conn, _iq_handler, NULL, STANZA_NAME_IQ, NULL, ctx);

    if (prefs_get_autoping() != 0) {
        int millis = prefs_get_autoping() * 1000;
        xmpp_timed_handler_add(conn, _autoping_timed_send, millis, ctx);
    }

    iq_rooms_cache_clear();
    iq_handlers_clear();

    id_handlers = g_hash_table_new_full(g_str_hash, g_str_equal, free, (GDestroyNotify)_iq_id_handler_free);
    rooms_cache = g_hash_table_new_full(g_str_hash, g_str_equal, free, (GDestroyNotify)xmpp_stanza_release);
}

void
iq_handlers_clear()
{
    if (id_handlers) {
        g_hash_table_remove_all(id_handlers);
        g_hash_table_destroy(id_handlers);
        id_handlers = NULL;
    }
}

static void
_iq_id_handler_free(ProfIqHandler* handler)
{
    if (handler == NULL) {
        return;
    }
    if (handler->free_func && handler->userdata) {
        handler->free_func(handler->userdata);
    }
    free(handler);
}

void
iq_id_handler_add(const char* const id, ProfIqCallback func, ProfIqFreeCallback free_func, void* userdata)
{
    ProfIqHandler* handler = malloc(sizeof(ProfIqHandler));
    if (handler) {
        handler->func = func;
        handler->free_func = free_func;
        handler->userdata = userdata;

        g_hash_table_insert(id_handlers, strdup(id), handler);
    }
}

void
iq_autoping_timer_cancel(void)
{
    autoping_wait = FALSE;
    if (autoping_time) {
        g_timer_destroy(autoping_time);
        autoping_time = NULL;
    }
}

void
iq_autoping_check(void)
{
    if (connection_get_status() != JABBER_CONNECTED) {
        return;
    }

    if (autoping_wait == FALSE) {
        return;
    }

    if (autoping_time == NULL) {
        return;
    }

    gdouble elapsed = g_timer_elapsed(autoping_time, NULL);
    unsigned long seconds_elapsed = elapsed * 1.0;
    gint timeout = prefs_get_autoping_timeout();
    if (timeout > 0 && seconds_elapsed >= timeout) {
        cons_show("Autoping response timed out after %u seconds.", timeout);
        log_debug("Autoping check: timed out after %u seconds, disconnecting", timeout);
        iq_autoping_timer_cancel();
        session_autoping_fail();
    }
}

void
iq_set_autoping(const int seconds)
{
    if (connection_get_status() != JABBER_CONNECTED) {
        return;
    }

    xmpp_conn_t* const conn = connection_get_conn();
    xmpp_timed_handler_delete(conn, _autoping_timed_send);

    if (seconds == 0) {
        return;
    }

    int millis = seconds * 1000;
    xmpp_ctx_t* const ctx = connection_get_ctx();
    xmpp_timed_handler_add(conn, _autoping_timed_send, millis, ctx);
}

void
iq_rooms_cache_clear(void)
{
    if (rooms_cache) {
        g_hash_table_remove_all(rooms_cache);
        g_hash_table_destroy(rooms_cache);
        rooms_cache = NULL;
    }
}

void
iq_room_list_request(gchar* conferencejid, gchar* filter)
{
    if (g_hash_table_contains(rooms_cache, conferencejid)) {
        log_debug("Rooms request cached for: %s", conferencejid);
        _room_list_id_handler(g_hash_table_lookup(rooms_cache, conferencejid), filter);
        return;
    }

    log_debug("Rooms request not cached for: %s", conferencejid);

    xmpp_ctx_t* const ctx = connection_get_ctx();
    char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = stanza_create_disco_items_iq(ctx, id, conferencejid, NULL);

    iq_id_handler_add(id, _room_list_id_handler, NULL, filter);

    iq_send_stanza(iq);
    xmpp_stanza_release(iq);
}

void
iq_enable_carbons(void)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    xmpp_stanza_t* iq = stanza_enable_carbons(ctx);
    const char* id = xmpp_stanza_get_id(iq);

    iq_id_handler_add(id, _enable_carbons_id_handler, NULL, NULL);

    iq_send_stanza(iq);
    xmpp_stanza_release(iq);
}

void
iq_disable_carbons(void)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    xmpp_stanza_t* iq = stanza_disable_carbons(ctx);
    const char* id = xmpp_stanza_get_id(iq);

    iq_id_handler_add(id, _disable_carbons_id_handler, NULL, NULL);

    iq_send_stanza(iq);
    xmpp_stanza_release(iq);
}

void
iq_http_upload_request(HTTPUpload* upload)
{
    char* jid = connection_jid_for_feature(STANZA_NS_HTTP_UPLOAD);
    if (jid == NULL) {
        cons_show_error("XEP-0363 HTTP File Upload is not supported by the server");
        return;
    }

    xmpp_ctx_t* const ctx = connection_get_ctx();
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = stanza_create_http_upload_request(ctx, id, jid, upload);
    iq_id_handler_add(id, _http_upload_response_id_handler, NULL, upload);

    iq_send_stanza(iq);
    xmpp_stanza_release(iq);

    return;
}

void
iq_disco_info_request(gchar* jid)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = stanza_create_disco_info_iq(ctx, id, jid, NULL);

    iq_id_handler_add(id, _disco_info_response_id_handler, NULL, NULL);

    iq_send_stanza(iq);
    xmpp_stanza_release(iq);
}

void
iq_disco_info_request_onconnect(gchar* jid)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = stanza_create_disco_info_iq(ctx, id, jid, NULL);

    iq_id_handler_add(id, _disco_info_response_id_handler_onconnect, NULL, NULL);

    iq_send_stanza(iq);
    xmpp_stanza_release(iq);
}

void
iq_last_activity_request(gchar* jid)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = stanza_create_last_activity_iq(ctx, id, jid);

    iq_id_handler_add(id, _last_activity_response_id_handler, NULL, NULL);

    iq_send_stanza(iq);
    xmpp_stanza_release(iq);
}

void
iq_room_info_request(const char* const room, gboolean display_result)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = stanza_create_disco_info_iq(ctx, id, room, NULL);

    ProfRoomInfoData* cb_data = malloc(sizeof(ProfRoomInfoData));
    if (cb_data) {
        cb_data->room = strdup(room);
        cb_data->display = display_result;

        iq_id_handler_add(id, _room_info_response_id_handler, (ProfIqFreeCallback)_iq_free_room_data, cb_data);

        iq_send_stanza(iq);
        xmpp_stanza_release(iq);
    }
}

void
iq_send_caps_request_for_jid(const char* const to, const char* const id,
                             const char* const node, const char* const ver)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();

    if (!node) {
        log_error("Could not create caps request, no node");
        return;
    }
    if (!ver) {
        log_error("Could not create caps request, no ver");
        return;
    }

    GString* node_str = g_string_new("");
    g_string_printf(node_str, "%s#%s", node, ver);
    xmpp_stanza_t* iq = stanza_create_disco_info_iq(ctx, id, to, node_str->str);
    g_string_free(node_str, TRUE);

    iq_id_handler_add(id, _caps_response_for_jid_id_handler, free, strdup(to));

    iq_send_stanza(iq);
    xmpp_stanza_release(iq);
}

void
iq_send_caps_request(const char* const to, const char* const id,
                     const char* const node, const char* const ver)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();

    if (!node) {
        log_error("Could not create caps request, no node");
        return;
    }
    if (!ver) {
        log_error("Could not create caps request, no ver");
        return;
    }

    GString* node_str = g_string_new("");
    g_string_printf(node_str, "%s#%s", node, ver);
    xmpp_stanza_t* iq = stanza_create_disco_info_iq(ctx, id, to, node_str->str);
    g_string_free(node_str, TRUE);

    iq_id_handler_add(id, _caps_response_id_handler, NULL, NULL);

    iq_send_stanza(iq);
    xmpp_stanza_release(iq);
}

void
iq_send_caps_request_legacy(const char* const to, const char* const id,
                            const char* const node, const char* const ver)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();

    if (!node) {
        log_error("Could not create caps request, no node");
        return;
    }
    if (!ver) {
        log_error("Could not create caps request, no ver");
        return;
    }

    gchar* node_str = g_strdup_printf("%s#%s", node, ver);
    xmpp_stanza_t* iq = stanza_create_disco_info_iq(ctx, id, to, node_str);

    iq_id_handler_add(id, _caps_response_legacy_id_handler, g_free, node_str);

    iq_send_stanza(iq);
    xmpp_stanza_release(iq);
}

void
iq_disco_items_request(gchar* jid)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    xmpp_stanza_t* iq = stanza_create_disco_items_iq(ctx, "discoitemsreq", jid, NULL);
    iq_send_stanza(iq);
    xmpp_stanza_release(iq);
}

void
iq_disco_items_request_onconnect(gchar* jid)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    xmpp_stanza_t* iq = stanza_create_disco_items_iq(ctx, "discoitemsreq_onconnect", jid, NULL);
    iq_send_stanza(iq);
    xmpp_stanza_release(iq);
}

void
iq_send_software_version(const char* const fulljid)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    xmpp_stanza_t* iq = stanza_create_software_version_iq(ctx, fulljid);

    const char* id = xmpp_stanza_get_id(iq);
    iq_id_handler_add(id, _version_result_id_handler, free, strdup(fulljid));

    iq_send_stanza(iq);
    xmpp_stanza_release(iq);
}

void
iq_confirm_instant_room(const char* const room_jid)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    xmpp_stanza_t* iq = stanza_create_instant_room_request_iq(ctx, room_jid);
    iq_send_stanza(iq);
    xmpp_stanza_release(iq);
}

void
iq_destroy_room(const char* const room_jid)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    xmpp_stanza_t* iq = stanza_create_instant_room_destroy_iq(ctx, room_jid);

    const char* id = xmpp_stanza_get_id(iq);
    iq_id_handler_add(id, _destroy_room_result_id_handler, NULL, NULL);

    iq_send_stanza(iq);
    xmpp_stanza_release(iq);
}

void
iq_request_room_config_form(const char* const room_jid)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    xmpp_stanza_t* iq = stanza_create_room_config_request_iq(ctx, room_jid);

    const char* id = xmpp_stanza_get_id(iq);
    iq_id_handler_add(id, _room_config_id_handler, NULL, NULL);

    iq_send_stanza(iq);
    xmpp_stanza_release(iq);
}

void
iq_submit_room_config(ProfConfWin* confwin)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    xmpp_stanza_t* iq = stanza_create_room_config_submit_iq(ctx, confwin->roomjid, confwin->form);

    const char* id = xmpp_stanza_get_id(iq);
    iq_id_handler_add(id, _room_config_submit_id_handler, NULL, NULL);

    iq_send_stanza(iq);
    xmpp_stanza_release(iq);
}

void
iq_room_config_cancel(ProfConfWin* confwin)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    xmpp_stanza_t* iq = stanza_create_room_config_cancel_iq(ctx, confwin->roomjid);
    iq_send_stanza(iq);
    xmpp_stanza_release(iq);
}

void
iq_room_affiliation_list(const char* const room, char* affiliation, bool show_ui_message)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    xmpp_stanza_t* iq = stanza_create_room_affiliation_list_iq(ctx, room, affiliation);

    const char* id = xmpp_stanza_get_id(iq);

    ProfAffiliationList* affiliation_list = malloc(sizeof(ProfAffiliationList));
    if (affiliation_list) {
        affiliation_list->affiliation = strdup(affiliation);
        affiliation_list->show_ui_message = show_ui_message;

        iq_id_handler_add(id, _room_affiliation_list_result_id_handler, (ProfIqFreeCallback)_iq_free_affiliation_list, affiliation_list);

        iq_send_stanza(iq);
        xmpp_stanza_release(iq);
    }
}

void
iq_room_kick_occupant(const char* const room, const char* const nick, const char* const reason)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    xmpp_stanza_t* iq = stanza_create_room_kick_iq(ctx, room, nick, reason);

    const char* id = xmpp_stanza_get_id(iq);
    iq_id_handler_add(id, _room_kick_result_id_handler, free, strdup(nick));

    iq_send_stanza(iq);
    xmpp_stanza_release(iq);
}

void
iq_room_affiliation_set(const char* const room, const char* const jid, char* affiliation,
                        const char* const reason)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    xmpp_stanza_t* iq = stanza_create_room_affiliation_set_iq(ctx, room, jid, affiliation, reason);

    const char* id = xmpp_stanza_get_id(iq);

    ProfPrivilegeSet* affiliation_set = malloc(sizeof(struct privilege_set_t));
    if (affiliation_set) {
        affiliation_set->item = strdup(jid);
        affiliation_set->privilege = strdup(affiliation);

        iq_id_handler_add(id, _room_affiliation_set_result_id_handler, (ProfIqFreeCallback)_iq_free_affiliation_set, affiliation_set);

        iq_send_stanza(iq);
        xmpp_stanza_release(iq);
    }
}

void
iq_room_role_set(const char* const room, const char* const nick, char* role,
                 const char* const reason)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    xmpp_stanza_t* iq = stanza_create_room_role_set_iq(ctx, room, nick, role, reason);

    const char* id = xmpp_stanza_get_id(iq);

    struct privilege_set_t* role_set = malloc(sizeof(ProfPrivilegeSet));
    if (role_set) {
        role_set->item = strdup(nick);
        role_set->privilege = strdup(role);

        iq_id_handler_add(id, _room_role_set_result_id_handler, (ProfIqFreeCallback)_iq_free_affiliation_set, role_set);

        iq_send_stanza(iq);
        xmpp_stanza_release(iq);
    }
}

void
iq_room_role_list(const char* const room, char* role)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    xmpp_stanza_t* iq = stanza_create_room_role_list_iq(ctx, room, role);

    const char* id = xmpp_stanza_get_id(iq);
    iq_id_handler_add(id, _room_role_list_result_id_handler, free, strdup(role));

    iq_send_stanza(iq);
    xmpp_stanza_release(iq);
}

void
iq_send_ping(const char* const target)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    xmpp_stanza_t* iq = stanza_create_ping_iq(ctx, target);
    const char* id = xmpp_stanza_get_id(iq);

    GDateTime* now = g_date_time_new_now_local();
    iq_id_handler_add(id, _manual_pong_id_handler, (ProfIqFreeCallback)g_date_time_unref, now);

    iq_send_stanza(iq);
    xmpp_stanza_release(iq);
}

void
iq_command_list(const char* const target)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    const char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = stanza_create_disco_items_iq(ctx, id, target, STANZA_NS_COMMAND);

    iq_id_handler_add(id, _command_list_result_handler, NULL, NULL);

    iq_send_stanza(iq);
    xmpp_stanza_release(iq);
}

void
iq_command_exec(const char* const target, const char* const command)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    xmpp_stanza_t* iq = stanza_create_command_exec_iq(ctx, target, command);
    const char* id = xmpp_stanza_get_id(iq);

    iq_id_handler_add(id, _command_exec_response_handler, free, strdup(command));

    iq_send_stanza(iq);
    xmpp_stanza_release(iq);
}

void
iq_submit_command_config(ProfConfWin* confwin)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    CommandConfigData* data = (CommandConfigData*)confwin->userdata;
    xmpp_stanza_t* iq = stanza_create_command_config_submit_iq(ctx, confwin->roomjid, data->command, data->sessionid, confwin->form);

    const char* id = xmpp_stanza_get_id(iq);
    iq_id_handler_add(id, _command_exec_response_handler, free, strdup(data->command));

    iq_send_stanza(iq);
    xmpp_stanza_release(iq);
    free(data->sessionid);
    free(data->command);
    free(data);
}

void
iq_cancel_command_config(ProfConfWin* confwin)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    CommandConfigData* data = (CommandConfigData*)confwin->userdata;
    xmpp_stanza_t* iq = stanza_create_room_config_cancel_iq(ctx, confwin->roomjid);
    iq_send_stanza(iq);
    xmpp_stanza_release(iq);
    free(data->sessionid);
    free(data->command);
    free(data);
}

static void
_error_handler(xmpp_stanza_t* const stanza)
{
    const char* id = xmpp_stanza_get_id(stanza);
    auto_char char* error_msg = stanza_get_error_message(stanza);

    if (id) {
        log_debug("IQ error handler fired, id: %s, error: %s", id, error_msg);
        log_error("IQ error received, id: %s, error: %s", id, error_msg);
    } else {
        log_debug("IQ error handler fired, error: %s", error_msg);
        log_error("IQ error received, error: %s", error_msg);
    }
}

static int
_caps_response_id_handler(xmpp_stanza_t* const stanza, void* const userdata)
{
    const char* id = xmpp_stanza_get_id(stanza);
    xmpp_stanza_t* query = xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_QUERY);

    const char* type = xmpp_stanza_get_type(stanza);
    // ignore non result
    if ((g_strcmp0(type, "get") == 0) || (g_strcmp0(type, "set") == 0)) {
        return 1;
    }

    if (id) {
        log_debug("Capabilities response handler fired for id %s", id);
    } else {
        log_debug("Capabilities response handler fired");
    }

    const char* from = xmpp_stanza_get_from(stanza);
    if (!from) {
        log_info("_caps_response_id_handler(): No from attribute");
        return 0;
    }

    // handle error responses
    if (g_strcmp0(type, STANZA_TYPE_ERROR) == 0) {
        auto_char char* error_message = stanza_get_error_message(stanza);
        log_warning("Error received for capabilities response from %s: ", from, error_message);
        return 0;
    }

    if (query == NULL) {
        log_info("_caps_response_id_handler(): No query element found.");
        return 0;
    }

    const char* node = xmpp_stanza_get_attribute(query, STANZA_ATTR_NODE);
    if (node == NULL) {
        log_info("_caps_response_id_handler(): No node attribute found");
        return 0;
    }

    // validate sha1
    auto_gcharv gchar** split = g_strsplit(node, "#", -1);
    char* given_sha1 = split[1];
    auto_gchar gchar* generated_sha1 = stanza_create_caps_sha1_from_query(query);

    if (g_strcmp0(given_sha1, generated_sha1) != 0) {
        log_warning("Generated sha-1 does not match given:");
        log_warning("Generated : %s", generated_sha1);
        log_warning("Given     : %s", given_sha1);
    } else {
        log_debug("Valid SHA-1 hash found: %s", given_sha1);

        if (caps_cache_contains(given_sha1)) {
            log_debug("Capabilities already cached: %s", given_sha1);
        } else {
            log_debug("Capabilities not cached: %s, storing", given_sha1);
            EntityCapabilities* capabilities = stanza_create_caps_from_query_element(query);

            caps_add_by_ver(given_sha1, capabilities);
            caps_destroy(capabilities);
        }

        caps_map_jid_to_ver(from, given_sha1);
    }

    return 0;
}

static int
_caps_response_for_jid_id_handler(xmpp_stanza_t* const stanza, void* const userdata)
{
    char* jid = (char*)userdata;
    const char* id = xmpp_stanza_get_id(stanza);
    xmpp_stanza_t* query = xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_QUERY);

    const char* type = xmpp_stanza_get_type(stanza);
    // ignore non result
    if ((g_strcmp0(type, "get") == 0) || (g_strcmp0(type, "set") == 0)) {
        return 1;
    }

    if (id) {
        log_debug("Capabilities response handler fired for id %s", id);
    } else {
        log_debug("Capabilities response handler fired");
    }

    const char* from = xmpp_stanza_get_from(stanza);
    if (!from) {
        if (jid) {
            log_info("_caps_response_for_jid_id_handler(): No from attribute for %s", jid);
        } else {
            log_info("_caps_response_for_jid_id_handler(): No from attribute");
        }
        return 0;
    }

    // handle error responses
    if (g_strcmp0(type, STANZA_TYPE_ERROR) == 0) {
        auto_char char* error_message = stanza_get_error_message(stanza);
        log_warning("Error received for capabilities response from %s: ", from, error_message);
        return 0;
    }

    if (query == NULL) {
        if (jid) {
            log_info("_caps_response_for_jid_id_handler(): No query element found for %s.", jid);
        } else {
            log_info("_caps_response_for_jid_id_handler(): No query element found.");
        }
        return 0;
    }

    const char* node = xmpp_stanza_get_attribute(query, STANZA_ATTR_NODE);
    if (node == NULL) {
        if (jid) {
            log_info("_caps_response_for_jid_id_handler(): No node attribute found for %s", jid);
        } else {
            log_info("_caps_response_for_jid_id_handler(): No node attribute found");
        }
        return 0;
    }

    log_debug("Associating capabilities with: %s", jid);
    EntityCapabilities* capabilities = stanza_create_caps_from_query_element(query);
    caps_add_by_jid(jid, capabilities);

    return 0;
}

static int
_caps_response_legacy_id_handler(xmpp_stanza_t* const stanza, void* const userdata)
{
    const char* id = xmpp_stanza_get_id(stanza);
    xmpp_stanza_t* query = xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_QUERY);
    char* expected_node = (char*)userdata;

    const char* type = xmpp_stanza_get_type(stanza);
    // ignore non result
    if ((g_strcmp0(type, "get") == 0) || (g_strcmp0(type, "set") == 0)) {
        return 1;
    }

    if (id) {
        log_debug("Capabilities response handler fired for id %s", id);
    } else {
        log_debug("Capabilities response handler fired");
    }

    const char* from = xmpp_stanza_get_from(stanza);
    if (!from) {
        log_info("_caps_response_legacy_id_handler(): No from attribute");
        return 0;
    }

    // handle error responses
    if (g_strcmp0(type, STANZA_TYPE_ERROR) == 0) {
        auto_char char* error_message = stanza_get_error_message(stanza);
        log_warning("Error received for capabilities response from %s: ", from, error_message);
        return 0;
    }

    if (query == NULL) {
        log_info("_caps_response_legacy_id_handler(): No query element found.");
        return 0;
    }

    const char* node = xmpp_stanza_get_attribute(query, STANZA_ATTR_NODE);
    if (node == NULL) {
        log_info("_caps_response_legacy_id_handler(): No node attribute found");
        return 0;
    }

    // nodes match
    if (g_strcmp0(expected_node, node) == 0) {
        log_debug("Legacy capabilities, nodes match %s", node);
        if (caps_cache_contains(node)) {
            log_debug("Capabilities already cached: %s", node);
        } else {
            log_debug("Capabilities not cached: %s, storing", node);
            EntityCapabilities* capabilities = stanza_create_caps_from_query_element(query);
            caps_add_by_ver(node, capabilities);
            caps_destroy(capabilities);
        }

        caps_map_jid_to_ver(from, node);

        // node match fail
    } else {
        log_info("Legacy Capabilities nodes do not match, expected %s, given %s.", expected_node, node);
    }

    return 0;
}

static int
_room_list_id_handler(xmpp_stanza_t* const stanza, void* const userdata)
{
    gchar* filter = (gchar*)userdata;
    const char* id = xmpp_stanza_get_id(stanza);
    const char* from = xmpp_stanza_get_from(stanza);

    if (prefs_get_boolean(PREF_ROOM_LIST_CACHE) && !g_hash_table_contains(rooms_cache, from)) {
        g_hash_table_insert(rooms_cache, strdup(from), xmpp_stanza_copy(stanza));
    }

    log_debug("Response to query: %s", id);

    xmpp_stanza_t* query = xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_QUERY);
    if (query == NULL) {
        return 0;
    }

    cons_show("");
    if (filter) {
        cons_show("Rooms list response received: %s, filter: %s", from, filter);
    } else {
        cons_show("Rooms list response received: %s", from);
    }
    xmpp_stanza_t* child = xmpp_stanza_get_children(query);
    if (child == NULL) {
        cons_show("  No rooms found.");
        return 0;
    }

    GPatternSpec* glob = NULL;
    if (filter != NULL) {
        gchar* filter_lower = g_utf8_strdown(filter, -1);
        GString* glob_str = g_string_new("*");
        g_string_append(glob_str, filter_lower);
        g_string_append(glob_str, "*");
        glob = g_pattern_spec_new(glob_str->str);
        g_string_free(glob_str, TRUE);
    }

    gboolean matched = FALSE;
    while (child) {
        const char* stanza_name = xmpp_stanza_get_name(child);
        if (stanza_name && (g_strcmp0(stanza_name, STANZA_NAME_ITEM) == 0)) {
            const char* item_jid = xmpp_stanza_get_attribute(child, STANZA_ATTR_JID);
            auto_gchar gchar* item_jid_lower = NULL;
            if (item_jid) {
                auto_jid Jid* jidp = jid_create(item_jid);
                if (jidp && jidp->localpart) {
                    item_jid_lower = g_utf8_strdown(jidp->localpart, -1);
                }
            }
            const char* item_name = xmpp_stanza_get_attribute(child, STANZA_ATTR_NAME);
            auto_gchar gchar* item_name_lower = NULL;
            if (item_name) {
                item_name_lower = g_utf8_strdown(item_name, -1);
            }
            if ((item_jid_lower) && ((glob == NULL) || ((g_pattern_match(glob, strlen(item_jid_lower), item_jid_lower, NULL)) || (item_name_lower && g_pattern_match(glob, strlen(item_name_lower), item_name_lower, NULL))))) {

                if (glob) {
                    matched = TRUE;
                }
                GString* item = g_string_new(item_jid);
                if (item_name) {
                    g_string_append(item, " (");
                    g_string_append(item, item_name);
                    g_string_append(item, ")");
                }
                cons_show("  %s", item->str);
                g_string_free(item, TRUE);
            }
        }
        child = xmpp_stanza_get_next(child);
    }

    if (glob && matched == FALSE) {
        cons_show("  No rooms found matching filter: %s", filter);
    }

    if (glob) {
        g_pattern_spec_free(glob);
    }

    return 0;
}

static int
_command_list_result_handler(xmpp_stanza_t* const stanza, void* const userdata)
{
    const char* id = xmpp_stanza_get_id(stanza);
    const char* type = xmpp_stanza_get_type(stanza);
    auto_char char* from = strdup(xmpp_stanza_get_from(stanza));

    if (id) {
        log_debug("IQ command list result handler fired, id: %s.", id);
    } else {
        log_debug("IQ command list result handler fired.");
    }

    // handle error responses
    if (g_strcmp0(type, STANZA_TYPE_ERROR) == 0) {
        auto_char char* error_message = stanza_get_error_message(stanza);
        log_debug("Error retrieving command list for %s: %s", from, error_message);
        ProfWin* win = wins_get_by_string(from);
        if (win) {
            win_command_list_error(win, error_message);
        }
        return 0;
    }

    GSList* cmds = NULL;

    xmpp_stanza_t* query = xmpp_stanza_get_child_by_ns(stanza, XMPP_NS_DISCO_ITEMS);
    if (query) {
        xmpp_stanza_t* child = xmpp_stanza_get_children(query);
        while (child) {
            const char* name = xmpp_stanza_get_name(child);
            if (g_strcmp0(name, "item") == 0) {
                const char* node = xmpp_stanza_get_attribute(child, STANZA_ATTR_NODE);
                if (node) {
                    cmds = g_slist_insert_sorted(cmds, (gpointer)node, (GCompareFunc)g_strcmp0);
                }
            }
            child = xmpp_stanza_get_next(child);
        }
    }

    ProfWin* win = wins_get_by_string(from);
    if (win == NULL) {
        win = wins_get_console();
    }

    win_handle_command_list(win, cmds);
    g_slist_free(cmds);

    return 0;
}

static int
_command_exec_response_handler(xmpp_stanza_t* const stanza, void* const userdata)
{
    const char* id = xmpp_stanza_get_id(stanza);
    const char* type = xmpp_stanza_get_type(stanza);
    const char* from = xmpp_stanza_get_from(stanza);
    char* command = userdata;

    if (id) {
        log_debug("IQ command exec response handler fired, id: %s.", id);
    } else {
        log_debug("IQ command exec response handler fired.");
    }

    ProfWin* win = wins_get_by_string(from);
    if (win == NULL) {
        /* No more window associated with this command.
         * Fallback to console. */
        win = wins_get_console();
    }

    // handle error responses
    if (g_strcmp0(type, STANZA_TYPE_ERROR) == 0) {
        auto_char char* error_message = stanza_get_error_message(stanza);
        log_debug("Error executing command %s for %s: %s", command, from, error_message);
        win_command_exec_error(win, command, error_message);
        return 0;
    }

    xmpp_stanza_t* cmd = xmpp_stanza_get_child_by_ns(stanza, STANZA_NS_COMMAND);
    if (!cmd) {
        log_error("No command element for command response");
        win_command_exec_error(win, command, "Malformed command response");
        return 0;
    }

    const char* status = xmpp_stanza_get_attribute(cmd, STANZA_ATTR_STATUS);
    if (g_strcmp0(status, "completed") == 0) {
        win_handle_command_exec_status(win, command, "completed");
        xmpp_stanza_t* note = xmpp_stanza_get_child_by_name(cmd, "note");
        if (note) {
            const char* type = xmpp_stanza_get_attribute(note, "type");
            const char* value = xmpp_stanza_get_text(note);
            win_handle_command_exec_result_note(win, type, value);
        }
        xmpp_stanza_t* x = xmpp_stanza_get_child_by_ns(cmd, STANZA_NS_DATA);
        if (x) {
            xmpp_stanza_t* roster = xmpp_stanza_get_child_by_ns(x, XMPP_NS_ROSTER);
            if (roster) {
                /* Special handling of xep-0133 roster in response */
                GSList* list = NULL;
                xmpp_stanza_t* child = xmpp_stanza_get_children(roster);
                while (child) {
                    const char* barejid = xmpp_stanza_get_attribute(child, STANZA_ATTR_JID);
                    gchar* barejid_lower = g_utf8_strdown(barejid, -1);
                    const char* name = xmpp_stanza_get_attribute(child, STANZA_ATTR_NAME);
                    const char* sub = xmpp_stanza_get_attribute(child, STANZA_ATTR_SUBSCRIPTION);
                    const char* ask = xmpp_stanza_get_attribute(child, STANZA_ATTR_ASK);

                    GSList* groups = NULL;
                    groups = roster_get_groups_from_item(child);

                    gboolean pending_out = FALSE;
                    if (ask && (strcmp(ask, "subscribe") == 0)) {
                        pending_out = TRUE;
                    }

                    PContact contact = p_contact_new(barejid_lower, name, groups, sub, NULL, pending_out);
                    list = g_slist_insert_sorted(list, contact, (GCompareFunc)roster_compare_name);
                    child = xmpp_stanza_get_next(child);
                }

                cons_show_roster(list);
                g_slist_free(list);
            } else {
                DataForm* form = form_create(x);
                ProfConfWin* confwin = (ProfConfWin*)wins_new_config(from, form, NULL, NULL, NULL);
                confwin_handle_configuration(confwin, form);
            }
        }
    } else if (g_strcmp0(status, "executing") == 0) {
        win_handle_command_exec_status(win, command, "executing");

        /* Looking for a jabber:x:data type form */
        xmpp_stanza_t* x = xmpp_stanza_get_child_by_ns(cmd, STANZA_NS_DATA);
        if (x == NULL) {
            return 0;
        }

        const char* form_type = xmpp_stanza_get_type(x);
        if (g_strcmp0(form_type, "form") != 0) {
            log_error("Unsupported payload in command response");
            win_command_exec_error(win, command, "Unsupported command response");
            return 0;
        }
        const char* sessionid = xmpp_stanza_get_attribute(cmd, "sessionid");

        DataForm* form = form_create(x);
        CommandConfigData* data = malloc(sizeof(CommandConfigData));
        if (sessionid == NULL) {
            data->sessionid = NULL;
        } else {
            data->sessionid = strdup(sessionid);
        }
        data->command = strdup(command);
        ProfConfWin* confwin = (ProfConfWin*)wins_new_config(from, form, iq_submit_command_config, iq_cancel_command_config, data);
        confwin_handle_configuration(confwin, form);
    } else if (g_strcmp0(status, "canceled") == 0) {
        win_handle_command_exec_status(win, command, "canceled");
        xmpp_stanza_t* note = xmpp_stanza_get_child_by_name(cmd, "note");
        if (note) {
            const char* type = xmpp_stanza_get_attribute(note, "type");
            const char* value = xmpp_stanza_get_text(note);
            win_handle_command_exec_result_note(win, type, value);
        }
    } else {
        log_error("Unsupported command status %s", status);
        win_command_exec_error(win, command, "Malformed command response");
    }

    return 0;
}

static int
_enable_carbons_id_handler(xmpp_stanza_t* const stanza, void* const userdata)
{
    const char* type = xmpp_stanza_get_type(stanza);
    if (g_strcmp0(type, "error") == 0) {
        auto_char char* error_message = stanza_get_error_message(stanza);
        cons_show_error("Server error enabling message carbons: %s", error_message);
        log_debug("Error enabling carbons: %s", error_message);
    } else {
        log_debug("Message carbons enabled.");
    }

    return 0;
}

static int
_disable_carbons_id_handler(xmpp_stanza_t* const stanza, void* const userdata)
{
    const char* type = xmpp_stanza_get_type(stanza);
    if (g_strcmp0(type, "error") == 0) {
        auto_char char* error_message = stanza_get_error_message(stanza);
        cons_show_error("Server error disabling message carbons: %s", error_message);
        log_debug("Error disabling carbons: %s", error_message);
    } else {
        log_debug("Message carbons disabled.");
    }

    return 0;
}

static int
_manual_pong_id_handler(xmpp_stanza_t* const stanza, void* const userdata)
{
    const char* from = xmpp_stanza_get_from(stanza);
    const char* type = xmpp_stanza_get_type(stanza);
    GDateTime* sent = (GDateTime*)userdata;

    // handle error responses
    if (g_strcmp0(type, STANZA_TYPE_ERROR) == 0) {
        auto_char char* error_message = stanza_get_error_message(stanza);
        if (!error_message) {
            cons_show_error("Error returned from pinging %s.", from);
        } else {
            cons_show_error("Error returned from pinging %s: %s.", from, error_message);
        }

        return 0;
    }

    GDateTime* now = g_date_time_new_now_local();

    GTimeSpan elapsed = g_date_time_difference(now, sent);
    int elapsed_millis = elapsed / 1000;

    g_date_time_unref(now);

    if (from == NULL) {
        cons_show("Ping response from server: %dms.", elapsed_millis);
    } else {
        cons_show("Ping response from %s: %dms.", from, elapsed_millis);
    }

    return 0;
}

static int
_autoping_timed_send(xmpp_conn_t* const conn, void* const userdata)
{
    if (connection_get_status() != JABBER_CONNECTED) {
        return 1;
    }

    if (connection_supports(XMPP_FEATURE_PING) == FALSE) {
        log_warning("Server doesn't advertise %s feature, disabling autoping.", XMPP_FEATURE_PING);
        prefs_set_autoping(0);
        cons_show_error("Server ping not supported (%s), autoping disabled.", XMPP_FEATURE_PING);
        return 0;
    }

    if (autoping_wait) {
        log_debug("Autoping: Existing ping already in progress, aborting");
        return 1;
    }

    xmpp_ctx_t* ctx = (xmpp_ctx_t*)userdata;
    xmpp_stanza_t* iq = stanza_create_ping_iq(ctx, NULL);
    const char* id = xmpp_stanza_get_id(iq);
    log_debug("Autoping: Sending ping request: %s", id);

    // add pong handler
    iq_id_handler_add(id, _auto_pong_id_handler, NULL, NULL);

    iq_send_stanza(iq);
    xmpp_stanza_release(iq);
    autoping_wait = TRUE;
    if (autoping_time) {
        g_timer_destroy(autoping_time);
    }
    autoping_time = g_timer_new();

    return 1;
}

void
autoping_timer_extend(void)
{
    if (autoping_time)
        g_timer_start(autoping_time);
}

static int
_auto_pong_id_handler(xmpp_stanza_t* const stanza, void* const userdata)
{
    iq_autoping_timer_cancel();

    const char* id = xmpp_stanza_get_id(stanza);
    if (id == NULL) {
        log_debug("Autoping: Pong handler fired.");
        return 0;
    }

    log_debug("Autoping: Pong handler fired: %s.", id);

    const char* type = xmpp_stanza_get_type(stanza);
    if (type == NULL) {
        return 0;
    }
    if (g_strcmp0(type, STANZA_TYPE_ERROR) != 0) {
        return 0;
    }

    // show warning if error
    auto_char char* error_msg = stanza_get_error_message(stanza);
    log_warning("Server ping (id=%s) responded with error: %s", id, error_msg);

    // turn off autoping if error type is 'cancel'
    xmpp_stanza_t* error = xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_ERROR);
    if (error == NULL) {
        return 0;
    }
    const char* errtype = xmpp_stanza_get_type(error);
    if (errtype == NULL) {
        return 0;
    }
    if (g_strcmp0(errtype, STANZA_TYPE_CANCEL) == 0) {
        log_warning("Server ping (id=%s) error type 'cancel', disabling autoping.", id);
        prefs_set_autoping(0);
        cons_show_error("Server ping not supported, autoping disabled.");
        xmpp_timed_handler_delete(connection_get_conn(), _autoping_timed_send);
    }

    return 0;
}

static int
_version_result_id_handler(xmpp_stanza_t* const stanza, void* const userdata)
{
    const char* id = xmpp_stanza_get_id(stanza);

    if (id) {
        log_debug("IQ version result handler fired, id: %s.", id);
    } else {
        log_debug("IQ version result handler fired.");
    }

    const char* type = xmpp_stanza_get_type(stanza);
    const char* from = xmpp_stanza_get_from(stanza);

    if (g_strcmp0(type, STANZA_TYPE_RESULT) != 0) {
        if (g_strcmp0(type, STANZA_TYPE_ERROR) == 0) {
            auto_char char* error_message = stanza_get_error_message(stanza);
            ui_handle_software_version_error(from, error_message);
        } else {
            ui_handle_software_version_error(from, "unknown error");
            log_error("Software version result with unrecognised type attribute.");
        }

        return 0;
    }

    const char* jid = xmpp_stanza_get_from(stanza);

    xmpp_stanza_t* query = xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_QUERY);
    if (query == NULL) {
        log_error("Software version result received with no query element.");
        return 0;
    }

    const char* ns = xmpp_stanza_get_ns(query);
    if (g_strcmp0(ns, STANZA_NS_VERSION) != 0) {
        log_error("Software version result received without namespace.");
        return 0;
    }

    char* name_str = NULL;
    char* version_str = NULL;
    char* os_str = NULL;
    xmpp_stanza_t* name = xmpp_stanza_get_child_by_name(query, "name");
    xmpp_stanza_t* version = xmpp_stanza_get_child_by_name(query, "version");
    xmpp_stanza_t* os = xmpp_stanza_get_child_by_name(query, "os");

    if (name) {
        name_str = xmpp_stanza_get_text(name);
    }
    if (version) {
        version_str = xmpp_stanza_get_text(version);
    }
    if (os) {
        os_str = xmpp_stanza_get_text(os);
    }

    if (g_strcmp0(jid, (char*)userdata) != 0) {
        log_warning("From attribute specified different JID, using original JID.");
    }

    xmpp_ctx_t* ctx = connection_get_ctx();

    auto_jid Jid* jidp = jid_create((char*)userdata);

    const char* presence = NULL;

    // if it has a fulljid it is a regular user (not server or component)
    if (jidp->fulljid) {
        if (muc_active(jidp->barejid)) {
            Occupant* occupant = muc_roster_item(jidp->barejid, jidp->resourcepart);
            presence = string_from_resource_presence(occupant->presence);
        } else {
            PContact contact = roster_get_contact(jidp->barejid);
            if (contact) {
                Resource* resource = p_contact_get_resource(contact, jidp->resourcepart);
                if (!resource) {
                    ui_handle_software_version_error(jidp->fulljid, "Unknown resource");
                    if (name_str)
                        xmpp_free(ctx, name_str);
                    if (version_str)
                        xmpp_free(ctx, version_str);
                    if (os_str)
                        xmpp_free(ctx, os_str);
                    return 0;
                }
                presence = string_from_resource_presence(resource->presence);
            } else {
                presence = "offline";
            }
        }
    }

    if (jidp->fulljid) {
        // regular user
        ui_show_software_version(jidp->fulljid, presence, name_str, version_str, os_str);
    } else {
        // server or component
        ui_show_software_version(jidp->barejid, "online", name_str, version_str, os_str);
    }

    if (name_str)
        xmpp_free(ctx, name_str);
    if (version_str)
        xmpp_free(ctx, version_str);
    if (os_str)
        xmpp_free(ctx, os_str);

    return 0;
}

static void
_ping_get_handler(xmpp_stanza_t* const stanza)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    const char* id = xmpp_stanza_get_id(stanza);
    const char* to = xmpp_stanza_get_to(stanza);
    const char* from = xmpp_stanza_get_from(stanza);

    if (id) {
        log_debug("IQ ping get handler fired, id: %s.", id);
    } else {
        log_debug("IQ ping get handler fired.");
    }

    if ((from == NULL) || (to == NULL)) {
        return;
    }

    xmpp_stanza_t* pong = xmpp_iq_new(ctx, STANZA_TYPE_RESULT, id);
    xmpp_stanza_set_to(pong, from);
    xmpp_stanza_set_from(pong, to);

    iq_send_stanza(pong);
    xmpp_stanza_release(pong);
}

static void
_version_get_handler(xmpp_stanza_t* const stanza)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    const char* id = xmpp_stanza_get_id(stanza);
    const char* from = xmpp_stanza_get_from(stanza);
    ProfAccount* account = accounts_get_account(session_get_account_name());
    auto_char char* client = account->client != NULL ? strdup(account->client) : NULL;
    account_free(account);
    bool is_custom_client = client != NULL;
    gchar* custom_version_str = NULL;
    if (is_custom_client) {
        custom_version_str = strstr(client, " ");
        if (custom_version_str != NULL) {
            *custom_version_str = '\0'; // Split string on name and version
            custom_version_str++;
        }
    }

    if (id) {
        log_debug("IQ version get handler fired, id: %s.", id);
    } else {
        log_debug("IQ version get handler fired.");
    }

    if (!from)
        return;

    if (prefs_get_boolean(PREF_ADV_NOTIFY_DISCO_OR_VERSION)) {
        cons_show("Received IQ version request (XEP-0092) from %s", from);
    }

    xmpp_stanza_t* response = xmpp_iq_new(ctx, STANZA_TYPE_RESULT, id);
    xmpp_stanza_set_to(response, from);

    xmpp_stanza_t* query = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(query, STANZA_NAME_QUERY);
    xmpp_stanza_set_ns(query, STANZA_NS_VERSION);

    xmpp_stanza_t* name = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(name, "name");
    xmpp_stanza_t* name_txt = xmpp_stanza_new(ctx);
    xmpp_stanza_set_text(name_txt, is_custom_client ? client : "Profanity");
    xmpp_stanza_add_child(name, name_txt);
    xmpp_stanza_add_child(query, name);
    bool include_os = prefs_get_boolean(PREF_REVEAL_OS) && !is_custom_client;
    xmpp_stanza_t* os = NULL;
    xmpp_stanza_t* os_txt = NULL;
    xmpp_stanza_t* version = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(version, "version");
    xmpp_stanza_t* version_txt = xmpp_stanza_new(ctx);
    auto_gchar gchar* prof_version = prof_get_version();

    if (!is_custom_client) {
        xmpp_stanza_set_text(version_txt, prof_version);
        xmpp_stanza_add_child(version, version_txt);
        xmpp_stanza_add_child(query, version);

        if (include_os) {
            os = xmpp_stanza_new(ctx);
            xmpp_stanza_set_name(os, "os");

            os_txt = xmpp_stanza_new(ctx);

#if defined(_WIN32) || defined(__CYGWIN__) || defined(PLATFORM_CYGWIN)
            xmpp_stanza_set_text(os_txt, "Windows");
#elif defined(__linux__)
            xmpp_stanza_set_text(os_txt, "Linux");
#elif defined(__APPLE__)
            xmpp_stanza_set_text(os_txt, "Apple");
#elif defined(__FreeBSD__)
            xmpp_stanza_set_text(os_txt, "FreeBSD");
#elif defined(__NetBSD__)
            xmpp_stanza_set_text(os_txt, "NetBSD");
#elif defined(__OpenBSD__)
            xmpp_stanza_set_text(os_txt, "OpenBSD");
#else
            xmpp_stanza_set_text(os_txt, "Unknown");
#endif
            xmpp_stanza_add_child(os, os_txt);
            xmpp_stanza_add_child(query, os);
        }
    } else if (custom_version_str != NULL) {
        xmpp_stanza_set_text(version_txt, custom_version_str);
        xmpp_stanza_add_child(version, version_txt);
        xmpp_stanza_add_child(query, version);
    }

    xmpp_stanza_add_child(response, query);
    iq_send_stanza(response);

    // Cleanup
    xmpp_stanza_release(version_txt);
    xmpp_stanza_release(name_txt);
    if (!is_custom_client && include_os) {
        xmpp_stanza_release(os_txt);
        xmpp_stanza_release(os);
    }
    xmpp_stanza_release(name);
    xmpp_stanza_release(version);
    xmpp_stanza_release(query);
    xmpp_stanza_release(response);
}

static void
_disco_items_get_handler(xmpp_stanza_t* const stanza)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    const char* id = xmpp_stanza_get_id(stanza);
    const char* from = xmpp_stanza_get_from(stanza);

    if (id) {
        log_debug("IQ disco items get handler fired, id: %s.", id);
    } else {
        log_debug("IQ disco items get handler fired.");
    }

    if (from) {
        xmpp_stanza_t* response = xmpp_iq_new(ctx, STANZA_TYPE_RESULT, xmpp_stanza_get_id(stanza));
        xmpp_stanza_set_to(response, from);

        xmpp_stanza_t* query = xmpp_stanza_new(ctx);
        xmpp_stanza_set_name(query, STANZA_NAME_QUERY);
        xmpp_stanza_set_ns(query, XMPP_NS_DISCO_ITEMS);
        xmpp_stanza_add_child(response, query);

        iq_send_stanza(response);

        xmpp_stanza_release(response);
    }
}

static void
_last_activity_get_handler(xmpp_stanza_t* const stanza)
{
    xmpp_ctx_t* ctx = connection_get_ctx();
    const char* from = xmpp_stanza_get_from(stanza);

    if (!from) {
        return;
    }

    if (prefs_get_boolean(PREF_LASTACTIVITY)) {
        int idls_secs = ui_get_idle_time() / 1000;
        char str[50];
        sprintf(str, "%d", idls_secs);

        xmpp_stanza_t* response = xmpp_iq_new(ctx, STANZA_TYPE_RESULT, xmpp_stanza_get_id(stanza));
        xmpp_stanza_set_to(response, from);

        xmpp_stanza_t* query = xmpp_stanza_new(ctx);
        xmpp_stanza_set_name(query, STANZA_NAME_QUERY);
        xmpp_stanza_set_attribute(query, STANZA_ATTR_XMLNS, STANZA_NS_LASTACTIVITY);
        xmpp_stanza_set_attribute(query, "seconds", str);

        xmpp_stanza_add_child(response, query);
        xmpp_stanza_release(query);

        iq_send_stanza(response);

        xmpp_stanza_release(response);
    } else {
        xmpp_stanza_t* response = xmpp_iq_new(ctx, STANZA_TYPE_ERROR, xmpp_stanza_get_id(stanza));
        xmpp_stanza_set_to(response, from);

        xmpp_stanza_t* error = xmpp_stanza_new(ctx);
        xmpp_stanza_set_name(error, STANZA_NAME_ERROR);
        xmpp_stanza_set_type(error, STANZA_TYPE_CANCEL);

        xmpp_stanza_t* service_unavailable = xmpp_stanza_new(ctx);
        xmpp_stanza_set_name(service_unavailable, "service-unavailable");
        xmpp_stanza_set_ns(service_unavailable, "urn:ietf:params:xml:ns:xmpp-stanzas");

        xmpp_stanza_add_child(error, service_unavailable);
        xmpp_stanza_release(service_unavailable);

        xmpp_stanza_add_child(response, error);
        xmpp_stanza_release(error);

        iq_send_stanza(response);

        xmpp_stanza_release(response);
    }
}

static void
_disco_info_get_handler(xmpp_stanza_t* const stanza)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    const char* from = xmpp_stanza_get_from(stanza);

    xmpp_stanza_t* incoming_query = xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_QUERY);
    const char* node_str = xmpp_stanza_get_attribute(incoming_query, STANZA_ATTR_NODE);

    const char* id = xmpp_stanza_get_id(stanza);

    if (id) {
        log_debug("IQ disco info get handler fired, id: %s.", id);
    } else {
        log_debug("IQ disco info get handler fired.");
    }

    if (from) {
        if (prefs_get_boolean(PREF_ADV_NOTIFY_DISCO_OR_VERSION)) {
            cons_show("Received IQ disco info request (XEP-0232) from %s", from);
        }

        xmpp_stanza_t* response = xmpp_iq_new(ctx, STANZA_TYPE_RESULT, xmpp_stanza_get_id(stanza));
        xmpp_stanza_set_to(response, from);

        xmpp_stanza_t* query = stanza_create_caps_query_element(ctx);
        if (node_str) {
            xmpp_stanza_set_attribute(query, STANZA_ATTR_NODE, node_str);
        }
        xmpp_stanza_add_child(response, query);
        iq_send_stanza(response);

        xmpp_stanza_release(query);
        xmpp_stanza_release(response);
    }
}

static int
_destroy_room_result_id_handler(xmpp_stanza_t* const stanza, void* const userdata)
{
    const char* id = xmpp_stanza_get_id(stanza);

    if (id) {
        log_debug("IQ destroy room result handler fired, id: %s.", id);
    } else {
        log_debug("IQ destroy room result handler fired.");
    }

    const char* from = xmpp_stanza_get_from(stanza);
    if (from == NULL) {
        log_error("No from attribute for IQ destroy room result");
    } else {
        sv_ev_room_destroy(from);
    }

    return 0;
}

static int
_room_config_id_handler(xmpp_stanza_t* const stanza, void* const userdata)
{
    const char* id = xmpp_stanza_get_id(stanza);
    const char* type = xmpp_stanza_get_type(stanza);
    const char* from = xmpp_stanza_get_from(stanza);

    if (id) {
        log_debug("IQ room config handler fired, id: %s.", id);
    } else {
        log_debug("IQ room config handler fired.");
    }

    // handle error responses
    if (g_strcmp0(type, STANZA_TYPE_ERROR) == 0) {
        auto_char char* error_message = stanza_get_error_message(stanza);
        ui_handle_room_configuration_form_error(from, error_message);
        return 0;
    }

    if (from == NULL) {
        log_warning("No from attribute for IQ config request result");
        ui_handle_room_configuration_form_error(from, "No from attribute for room config response.");
        return 0;
    }

    xmpp_stanza_t* query = xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_QUERY);
    if (query == NULL) {
        log_warning("No query element found parsing room config response");
        ui_handle_room_configuration_form_error(from, "No query element found parsing room config response");
        return 0;
    }

    xmpp_stanza_t* x = xmpp_stanza_get_child_by_ns(query, STANZA_NS_DATA);
    if (x == NULL) {
        log_warning("No x element found with %s namespace parsing room config response", STANZA_NS_DATA);
        ui_handle_room_configuration_form_error(from, "No form configuration options available");
        return 0;
    }

    const char* form_type = xmpp_stanza_get_type(x);
    if (g_strcmp0(form_type, "form") != 0) {
        log_warning("x element not of type 'form' parsing room config response");
        ui_handle_room_configuration_form_error(from, "Form not of type 'form' parsing room config response.");
        return 0;
    }

    DataForm* form = form_create(x);
    ProfConfWin* confwin = (ProfConfWin*)wins_new_config(from, form, iq_submit_room_config, iq_room_config_cancel, NULL);
    confwin_handle_configuration(confwin, form);

    return 0;
}

static int
_room_affiliation_set_result_id_handler(xmpp_stanza_t* const stanza, void* const userdata)
{
    const char* id = xmpp_stanza_get_id(stanza);
    const char* type = xmpp_stanza_get_type(stanza);
    const char* from = xmpp_stanza_get_from(stanza);
    ProfPrivilegeSet* affiliation_set = (ProfPrivilegeSet*)userdata;

    if (id) {
        log_debug("IQ affiliation set handler fired, id: %s.", id);
    } else {
        log_debug("IQ affiliation set handler fired.");
    }

    // handle error responses
    if (g_strcmp0(type, STANZA_TYPE_ERROR) == 0) {
        auto_char char* error_message = stanza_get_error_message(stanza);
        log_debug("Error setting affiliation %s list for room %s, user %s: %s", affiliation_set->privilege, from, affiliation_set->item, error_message);
        ProfMucWin* mucwin = wins_get_muc(from);
        if (mucwin) {
            mucwin_affiliation_set_error(mucwin, affiliation_set->item, affiliation_set->privilege, error_message);
        }
    }

    return 0;
}

static int
_room_role_set_result_id_handler(xmpp_stanza_t* const stanza, void* const userdata)
{
    const char* id = xmpp_stanza_get_id(stanza);
    const char* type = xmpp_stanza_get_type(stanza);
    const char* from = xmpp_stanza_get_from(stanza);
    ProfPrivilegeSet* role_set = (ProfPrivilegeSet*)userdata;

    if (id) {
        log_debug("IQ role set handler fired, id: %s.", id);
    } else {
        log_debug("IQ role set handler fired.");
    }

    // handle error responses
    if (g_strcmp0(type, STANZA_TYPE_ERROR) == 0) {
        auto_char char* error_message = stanza_get_error_message(stanza);
        log_debug("Error setting role %s list for room %s, user %s: %s", role_set->privilege, from, role_set->item, error_message);
        ProfMucWin* mucwin = wins_get_muc(from);
        if (mucwin) {
            mucwin_role_set_error(mucwin, role_set->item, role_set->privilege, error_message);
        }
    }

    return 0;
}

static int
_room_affiliation_list_result_id_handler(xmpp_stanza_t* const stanza, void* const userdata)
{
    const char* id = xmpp_stanza_get_id(stanza);
    const char* type = xmpp_stanza_get_type(stanza);
    const char* from = xmpp_stanza_get_from(stanza);
    ProfAffiliationList* affiliation_list = (ProfAffiliationList*)userdata;

    if (id) {
        log_debug("IQ affiliation list result handler fired, id: %s.", id);
    } else {
        log_debug("IQ affiliation list result handler fired.");
    }

    // handle error responses
    if (g_strcmp0(type, STANZA_TYPE_ERROR) == 0) {
        auto_char char* error_message = stanza_get_error_message(stanza);
        log_debug("Error retrieving %s list for room %s: %s", affiliation_list->affiliation, from, error_message);
        ProfMucWin* mucwin = wins_get_muc(from);
        if (mucwin && affiliation_list->show_ui_message) {
            mucwin_affiliation_list_error(mucwin, affiliation_list->affiliation, error_message);
        }
        return 0;
    }
    GSList* jids = NULL;

    xmpp_stanza_t* query = xmpp_stanza_get_child_by_ns(stanza, STANZA_NS_MUC_ADMIN);
    if (query) {
        xmpp_stanza_t* child = xmpp_stanza_get_children(query);
        while (child) {
            const char* name = xmpp_stanza_get_name(child);
            if (g_strcmp0(name, "item") == 0) {
                const char* jid = xmpp_stanza_get_attribute(child, STANZA_ATTR_JID);
                if (jid) {
                    if (g_strcmp0(affiliation_list->affiliation, "member") == 0
                        || g_strcmp0(affiliation_list->affiliation, "owner") == 0
                        || g_strcmp0(affiliation_list->affiliation, "admin") == 0) {
                        muc_members_add(from, jid);
                    }
                    jids = g_slist_insert_sorted(jids, (gpointer)jid, (GCompareFunc)g_strcmp0);
                }
            }
            child = xmpp_stanza_get_next(child);
        }
    }

    muc_jid_autocomplete_add_all(from, jids);
    ProfMucWin* mucwin = wins_get_muc(from);
    if (mucwin && affiliation_list->show_ui_message) {
        mucwin_handle_affiliation_list(mucwin, affiliation_list->affiliation, jids);
    }
    g_slist_free(jids);

    return 0;
}

static int
_room_role_list_result_id_handler(xmpp_stanza_t* const stanza, void* const userdata)
{
    const char* id = xmpp_stanza_get_id(stanza);
    const char* type = xmpp_stanza_get_type(stanza);
    const char* from = xmpp_stanza_get_from(stanza);
    char* role = (char*)userdata;

    if (id) {
        log_debug("IQ role list result handler fired, id: %s.", id);
    } else {
        log_debug("IQ role list result handler fired.");
    }

    // handle error responses
    if (g_strcmp0(type, STANZA_TYPE_ERROR) == 0) {
        auto_char char* error_message = stanza_get_error_message(stanza);
        log_debug("Error retrieving %s list for room %s: %s", role, from, error_message);
        ProfMucWin* mucwin = wins_get_muc(from);
        if (mucwin) {
            mucwin_role_list_error(mucwin, role, error_message);
        }
        return 0;
    }
    GSList* nicks = NULL;

    xmpp_stanza_t* query = xmpp_stanza_get_child_by_ns(stanza, STANZA_NS_MUC_ADMIN);
    if (query) {
        xmpp_stanza_t* child = xmpp_stanza_get_children(query);
        while (child) {
            const char* name = xmpp_stanza_get_name(child);
            if (g_strcmp0(name, "item") == 0) {
                const char* nick = xmpp_stanza_get_attribute(child, STANZA_ATTR_NICK);
                if (nick) {
                    nicks = g_slist_insert_sorted(nicks, (gpointer)nick, (GCompareFunc)g_strcmp0);
                }
            }
            child = xmpp_stanza_get_next(child);
        }
    }

    ProfMucWin* mucwin = wins_get_muc(from);
    if (mucwin) {
        mucwin_handle_role_list(mucwin, role, nicks);
    }
    g_slist_free(nicks);

    return 0;
}

static int
_room_config_submit_id_handler(xmpp_stanza_t* const stanza, void* const userdata)
{
    const char* id = xmpp_stanza_get_id(stanza);
    const char* type = xmpp_stanza_get_type(stanza);
    const char* from = xmpp_stanza_get_from(stanza);

    if (id) {
        log_debug("IQ room config submit handler fired, id: %s.", id);
    } else {
        log_debug("IQ room config submit handler fired.");
    }

    // handle error responses
    if (g_strcmp0(type, STANZA_TYPE_ERROR) == 0) {
        auto_char char* error_message = stanza_get_error_message(stanza);
        ui_handle_room_config_submit_result_error(from, error_message);
        return 0;
    }

    ui_handle_room_config_submit_result(from);

    return 0;
}

static int
_room_kick_result_id_handler(xmpp_stanza_t* const stanza, void* const userdata)
{
    const char* id = xmpp_stanza_get_id(stanza);
    const char* type = xmpp_stanza_get_type(stanza);
    const char* from = xmpp_stanza_get_from(stanza);
    char* nick = (char*)userdata;

    if (id) {
        log_debug("IQ kick result handler fired, id: %s.", id);
    } else {
        log_debug("IQ kick result handler fired.");
    }

    // handle error responses
    ProfMucWin* mucwin = wins_get_muc(from);
    if (mucwin && (g_strcmp0(type, STANZA_TYPE_ERROR) == 0)) {
        auto_char char* error_message = stanza_get_error_message(stanza);
        mucwin_kick_error(mucwin, nick, error_message);
    }

    return 0;
}

static void
_identity_destroy(DiscoIdentity* identity)
{
    if (identity) {
        free(identity->name);
        free(identity->type);
        free(identity->category);
        free(identity);
    }
}

static void
_item_destroy(DiscoItem* item)
{
    if (item) {
        free(item->jid);
        free(item->name);
        free(item);
    }
}

static int
_room_info_response_id_handler(xmpp_stanza_t* const stanza, void* const userdata)
{
    const char* type = xmpp_stanza_get_type(stanza);
    ProfRoomInfoData* cb_data = (ProfRoomInfoData*)userdata;
    log_debug("Received disco#info response for room: %s", cb_data->room);

    // handle error responses
    if (g_strcmp0(type, STANZA_TYPE_ERROR) == 0) {
        ProfMucWin* mucwin = wins_get_muc(cb_data->room);
        if (mucwin && cb_data->display) {
            auto_char char* error_message = stanza_get_error_message(stanza);
            mucwin_room_info_error(mucwin, error_message);
        }
        return 0;
    }

    xmpp_stanza_t* query = xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_QUERY);

    if (query) {
        xmpp_stanza_t* child = xmpp_stanza_get_children(query);
        GSList* identities = NULL;
        GSList* features = NULL;
        while (child) {
            const char* stanza_name = xmpp_stanza_get_name(child);
            if (g_strcmp0(stanza_name, STANZA_NAME_FEATURE) == 0) {
                const char* var = xmpp_stanza_get_attribute(child, STANZA_ATTR_VAR);
                if (var) {
                    features = g_slist_append(features, strdup(var));
                }
            } else if (g_strcmp0(stanza_name, STANZA_NAME_IDENTITY) == 0) {
                const char* name = xmpp_stanza_get_attribute(child, STANZA_ATTR_NAME);
                const char* type = xmpp_stanza_get_type(child);
                const char* category = xmpp_stanza_get_attribute(child, STANZA_ATTR_CATEGORY);

                if (name || category || type) {
                    DiscoIdentity* identity = malloc(sizeof(struct disco_identity_t));

                    if (identity) {
                        if (name) {
                            identity->name = strdup(name);
                            mucwin_set_room_name(cb_data->room, name);
                        } else {
                            identity->name = NULL;
                        }
                        if (category) {
                            identity->category = strdup(category);
                        } else {
                            identity->category = NULL;
                        }
                        if (type) {
                            identity->type = strdup(type);
                        } else {
                            identity->type = NULL;
                        }

                        identities = g_slist_append(identities, identity);
                    }
                }
            }

            child = xmpp_stanza_get_next(child);
        }

        muc_set_features(cb_data->room, features);
        ProfMucWin* mucwin = wins_get_muc(cb_data->room);
        if (mucwin) {
#ifdef HAVE_OMEMO
            if (muc_anonymity_type(mucwin->roomjid) == MUC_ANONYMITY_TYPE_NONANONYMOUS && omemo_automatic_start(cb_data->room)) {
                omemo_start_muc_sessions(cb_data->room);
                mucwin->is_omemo = TRUE;
            }
#endif
            if (cb_data->display) {
                mucwin_room_disco_info(mucwin, identities, features);
            }
        }

        g_slist_free_full(features, free);
        g_slist_free_full(identities, (GDestroyNotify)_identity_destroy);
    }

    return 0;
}

static int
_last_activity_response_id_handler(xmpp_stanza_t* const stanza, void* const userdata)
{
    const char* from = xmpp_stanza_get_from(stanza);
    if (!from) {
        cons_show_error("Invalid last activity response received.");
        log_info("Received last activity response with no from attribute.");
        return 0;
    }

    const char* type = xmpp_stanza_get_type(stanza);

    // handle error responses
    if (g_strcmp0(type, STANZA_TYPE_ERROR) == 0) {
        auto_char char* error_message = stanza_get_error_message(stanza);
        if (from) {
            cons_show_error("Last activity request failed for %s: %s", from, error_message);
        } else {
            cons_show_error("Last activity request failed: %s", error_message);
        }
        return 0;
    }

    xmpp_stanza_t* query = xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_QUERY);

    if (!query) {
        cons_show_error("Invalid last activity response received.");
        log_info("Received last activity response with no query element.");
        return 0;
    }

    const char* seconds_str = xmpp_stanza_get_attribute(query, "seconds");
    if (!seconds_str) {
        cons_show_error("Invalid last activity response received.");
        log_info("Received last activity response with no seconds attribute.");
        return 0;
    }

    int seconds = atoi(seconds_str);
    if (seconds < 0) {
        cons_show_error("Invalid last activity response received.");
        log_info("Received last activity response with negative value.");
        return 0;
    }

    char* msg = xmpp_stanza_get_text(query);

    sv_ev_lastactivity_response(from, seconds, msg);

    xmpp_free(connection_get_ctx(), msg);
    return 0;
}

static int
_disco_info_response_id_handler(xmpp_stanza_t* const stanza, void* const userdata)
{
    const char* from = xmpp_stanza_get_from(stanza);
    const char* type = xmpp_stanza_get_type(stanza);

    if (from) {
        log_debug("Received disco#info response from: %s", from);
    } else {
        log_debug("Received disco#info response");
    }

    // handle error responses
    if (g_strcmp0(type, STANZA_TYPE_ERROR) == 0) {
        auto_char char* error_message = stanza_get_error_message(stanza);
        if (from) {
            cons_show_error("Service discovery failed for %s: %s", from, error_message);
        } else {
            cons_show_error("Service discovery failed: %s", error_message);
        }
        return 0;
    }

    xmpp_stanza_t* query = xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_QUERY);

    if (query) {
        xmpp_stanza_t* child = xmpp_stanza_get_children(query);
        GSList* identities = NULL;
        GSList* features = NULL;
        while (child) {
            const char* stanza_name = xmpp_stanza_get_name(child);
            const char* child_type = xmpp_stanza_get_type(child);

            if (g_strcmp0(stanza_name, STANZA_NAME_FEATURE) == 0) {
                const char* var = xmpp_stanza_get_attribute(child, STANZA_ATTR_VAR);
                if (var) {
                    features = g_slist_append(features, strdup(var));
                }
            } else if (g_strcmp0(stanza_name, STANZA_NAME_IDENTITY) == 0) {
                const char* name = xmpp_stanza_get_attribute(child, STANZA_ATTR_NAME);
                const char* category = xmpp_stanza_get_attribute(child, STANZA_ATTR_CATEGORY);

                if (name || category || child_type) {
                    DiscoIdentity* identity = malloc(sizeof(struct disco_identity_t));

                    if (identity) {
                        if (name) {
                            identity->name = strdup(name);
                        } else {
                            identity->name = NULL;
                        }
                        if (category) {
                            identity->category = strdup(category);
                        } else {
                            identity->category = NULL;
                        }
                        if (child_type) {
                            identity->type = strdup(child_type);
                        } else {
                            identity->type = NULL;
                        }

                        identities = g_slist_append(identities, identity);
                    }
                }
            } else if (g_strcmp0(child_type, STANZA_TYPE_RESULT) == 0) {
                GHashTable* adr = stanza_get_service_contact_addresses(connection_get_ctx(), child);
                cons_show_disco_contact_information(adr);
                g_hash_table_destroy(adr);
            }

            child = xmpp_stanza_get_next(child);
        }

        cons_show_disco_info(from, identities, features);

        g_slist_free_full(features, free);
        g_slist_free_full(identities, (GDestroyNotify)_identity_destroy);
    }

    return 0;
}

static int
_disco_info_response_id_handler_onconnect(xmpp_stanza_t* const stanza, void* const userdata)
{
    const char* from = xmpp_stanza_get_from(stanza);
    const char* type = xmpp_stanza_get_type(stanza);

    if (from) {
        log_debug("Received disco#info response from: %s", from);
    } else {
        log_debug("Received disco#info response");
    }

    // handle error responses
    if (g_strcmp0(type, STANZA_TYPE_ERROR) == 0) {
        auto_char char* error_message = stanza_get_error_message(stanza);
        if (from) {
            log_error("Service discovery failed for %s: %s", from, error_message);
        } else {
            log_error("Service discovery failed: %s", error_message);
        }
        return 0;
    }

    xmpp_stanza_t* query = xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_QUERY);

    if (query) {
        GHashTable* features = connection_get_features(from);
        if (features == NULL) {
            log_error("No matching disco item found for %s", from);
            return 1;
        }

        xmpp_stanza_t* child = xmpp_stanza_get_children(query);
        while (child) {
            const char* stanza_name = xmpp_stanza_get_name(child);
            if (g_strcmp0(stanza_name, STANZA_NAME_FEATURE) == 0) {
                const char* var = xmpp_stanza_get_attribute(child, STANZA_ATTR_VAR);
                if (var) {
                    g_hash_table_add(features, strdup(var));
                }
            }
            child = xmpp_stanza_get_next(child);
        }
    }

    connection_features_received(from);

    return 0;
}

static int
_http_upload_response_id_handler(xmpp_stanza_t* const stanza, void* const userdata)
{
    HTTPUpload* upload = (HTTPUpload*)userdata;
    const char* from = xmpp_stanza_get_from(stanza);
    const char* type = xmpp_stanza_get_type(stanza);

    if (from) {
        log_info("Received http_upload response from: %s", from);
    } else {
        log_info("Received http_upload response");
    }

    // handle error responses
    if (g_strcmp0(type, STANZA_TYPE_ERROR) == 0) {
        auto_char char* error_message = stanza_get_error_message(stanza);
        if (from) {
            cons_show_error("Uploading '%s' failed for %s: %s", upload->filename, from, error_message);
        } else {
            cons_show_error("Uploading '%s' failed: %s", upload->filename, error_message);
        }
        return 0;
    }

    xmpp_stanza_t* slot = xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_SLOT);

    if (slot && g_strcmp0(xmpp_stanza_get_ns(slot), STANZA_NS_HTTP_UPLOAD) == 0) {
        xmpp_stanza_t* put = xmpp_stanza_get_child_by_name(slot, STANZA_NAME_PUT);
        xmpp_stanza_t* get = xmpp_stanza_get_child_by_name(slot, STANZA_NAME_GET);

        if (put && get) {
            const char* put_url = xmpp_stanza_get_attribute(put, "url");
            upload->put_url = strdup(put_url);
            const char* get_url = xmpp_stanza_get_attribute(get, "url");
            upload->get_url = strdup(get_url);

            // Optional "authorization", "cookie", "expires" headers
            upload->authorization = upload->cookie = upload->expires = NULL;
            xmpp_stanza_t* header = xmpp_stanza_get_children(put);
            for (; header; header = xmpp_stanza_get_next(header)) {
                if (g_strcmp0(xmpp_stanza_get_name(header), STANZA_NAME_HEADER) == 0) {
                    const char* header_name = xmpp_stanza_get_attribute(header, STANZA_ATTR_NAME);
                    if (g_strcmp0(header_name, STANZA_HEADER_AUTHORIZATION) == 0) {
                        upload->authorization = xmpp_stanza_get_text(header);
                    } else if (g_strcmp0(header_name, STANZA_HEADER_COOKIE) == 0) {
                        upload->cookie = xmpp_stanza_get_text(header);
                    } else if (g_strcmp0(header_name, STANZA_HEADER_EXPIRES) == 0) {
                        upload->expires = xmpp_stanza_get_text(header);
                    } else {
                        log_warning("[HTTP upload] unknown header: %s", header_name);
                    }
                }
            }

            pthread_create(&(upload->worker), NULL, &http_file_put, upload);
            http_upload_add_upload(upload);
        } else {
            log_error("Invalid XML in HTTP Upload slot");
            return 1;
        }
    }

    return 0;
}

static void
_disco_items_result_handler(xmpp_stanza_t* const stanza)
{
    log_debug("Received disco#items response");
    const char* id = xmpp_stanza_get_id(stanza);
    const char* from = xmpp_stanza_get_from(stanza);
    GSList* items = NULL;

    if ((g_strcmp0(id, "discoitemsreq") != 0) && (g_strcmp0(id, "discoitemsreq_onconnect") != 0)) {
        return;
    }

    log_debug("Response to query: %s", id);

    xmpp_stanza_t* query = xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_QUERY);
    if (query == NULL) {
        return;
    }

    xmpp_stanza_t* child = xmpp_stanza_get_children(query);
    if (child == NULL) {
        return;
    }

    while (child) {
        const char* stanza_name = xmpp_stanza_get_name(child);
        if (stanza_name && (g_strcmp0(stanza_name, STANZA_NAME_ITEM) == 0)) {
            const char* item_jid = xmpp_stanza_get_attribute(child, STANZA_ATTR_JID);
            if (item_jid) {
                DiscoItem* item = malloc(sizeof(struct disco_item_t));
                if (item) {
                    item->jid = strdup(item_jid);
                    const char* item_name = xmpp_stanza_get_attribute(child, STANZA_ATTR_NAME);
                    if (item_name) {
                        item->name = strdup(item_name);
                    } else {
                        item->name = NULL;
                    }
                    items = g_slist_append(items, item);
                }
            }
        }

        child = xmpp_stanza_get_next(child);
    }

    if (g_strcmp0(id, "discoitemsreq") == 0) {
        cons_show_disco_items(items, from);
    } else if (g_strcmp0(id, "discoitemsreq_onconnect") == 0) {
        received_disco_items = TRUE;
        connection_set_disco_items(items);

        while (late_delivery_windows) {
            LateDeliveryUserdata* del_data = late_delivery_windows->data;
            _iq_mam_request(del_data->win, del_data->startdate, del_data->enddate);
            free(del_data);
            late_delivery_windows = g_slist_delete_link(late_delivery_windows,
                                                        late_delivery_windows);
        }
    }

    g_slist_free_full(items, (GDestroyNotify)_item_destroy);
}

void
iq_send_stanza(xmpp_stanza_t* const stanza)
{
    char* text;
    size_t text_size;
    xmpp_stanza_to_text(stanza, &text, &text_size);

    xmpp_conn_t* conn = connection_get_conn();
    auto_char char* plugin_text = plugins_on_iq_stanza_send(text);
    if (plugin_text) {
        xmpp_send_raw_string(conn, "%s", plugin_text);
    } else {
        xmpp_send_raw_string(conn, "%s", text);
    }
    xmpp_free(connection_get_ctx(), text);
}

static void
_iq_free_room_data(ProfRoomInfoData* roominfo)
{
    if (roominfo) {
        free(roominfo->room);
        free(roominfo);
    }
}

static void
_iq_free_affiliation_set(ProfPrivilegeSet* affiliation_set)
{
    if (affiliation_set) {
        free(affiliation_set->item);
        free(affiliation_set->privilege);
        free(affiliation_set);
    }
}

static void
_iq_free_affiliation_list(ProfAffiliationList* affiliation_list)
{
    if (affiliation_list) {
        free(affiliation_list->affiliation);
        free(affiliation_list);
    }
}

static int
_mam_buffer_commit_handler(xmpp_stanza_t* const stanza, void* const userdata)
{
    ProfChatWin* chatwin = (ProfChatWin*)userdata;
    // Remove the "Loading messages …" message
    buffer_remove_entry(((ProfWin*)chatwin)->layout->buffer, 0);
    chatwin_db_history(chatwin, NULL, NULL, TRUE);
    return 0;
}

static const gchar* mam_timestamp_format_string = "%FT%T.%f%:z";

void
iq_mam_request_older(ProfChatWin* win)
{
    if (connection_supports(XMPP_FEATURE_MAM2) == FALSE) {
        log_warning("Server doesn't advertise %s feature.", XMPP_FEATURE_MAM2);
        cons_show_error("Server doesn't support MAM (%s).", XMPP_FEATURE_MAM2);
        return;
    }

    ProfMessage* first_msg = log_database_get_limits_info(win->barejid, FALSE);
    char* firstid = NULL;
    auto_gchar gchar* enddate = NULL;

    // If first message found
    if (first_msg->timestamp) {
        firstid = first_msg->stanzaid;
        enddate = g_date_time_format(first_msg->timestamp, mam_timestamp_format_string);
    } else {
        return;
    }

    xmpp_ctx_t* const ctx = connection_get_ctx();
    xmpp_stanza_t* iq = stanza_create_mam_iq(ctx, win->barejid, NULL, enddate, firstid, NULL);
    iq_id_handler_add(xmpp_stanza_get_id(iq), _mam_buffer_commit_handler, NULL, win);

    message_free(first_msg);

    iq_send_stanza(iq);
    xmpp_stanza_release(iq);

    return;
}

static void
_mam_userdata_free(MamRsmUserdata* data)
{
    free(data->end_datestr);
    data->end_datestr = NULL;
    free(data->start_datestr);
    data->start_datestr = NULL;
    free(data->barejid);
    data->barejid = NULL;
    free(data);
}

void
_iq_mam_request(ProfChatWin* win, GDateTime* startdate, GDateTime* enddate)
{
    if (connection_supports(XMPP_FEATURE_MAM2) == FALSE) {
        log_warning("Server doesn't advertise %s feature.", XMPP_FEATURE_MAM2);
        cons_show_error("Server doesn't support MAM (%s).", XMPP_FEATURE_MAM2);
        g_date_time_unref(startdate);
        g_date_time_unref(enddate);
        return;
    }

    char* firstid = "";
    char* startdate_str = NULL;
    char* enddate_str = NULL;
    gboolean fetch_next = FALSE;

    if (startdate) {
        startdate_str = g_date_time_format(startdate, mam_timestamp_format_string);
        fetch_next = TRUE;
        g_date_time_unref(startdate);
    } else if (!enddate) {
        GDateTime* now = g_date_time_new_now_utc();
        enddate_str = g_date_time_format(now, mam_timestamp_format_string);
        g_date_time_unref(now);
    }

    if (enddate) {
        enddate_str = g_date_time_format(enddate, mam_timestamp_format_string);
        g_date_time_unref(enddate);
    }

    xmpp_ctx_t* const ctx = connection_get_ctx();

    xmpp_stanza_t* iq = stanza_create_mam_iq(ctx, win->barejid, startdate_str, enddate_str, firstid, NULL);

    MamRsmUserdata* data = malloc(sizeof(MamRsmUserdata));
    if (data) {
        data->start_datestr = startdate_str;
        data->end_datestr = enddate_str;
        data->barejid = strdup(win->barejid);
        data->fetch_next = fetch_next;
        data->win = win;

        iq_id_handler_add(xmpp_stanza_get_id(iq), _mam_rsm_id_handler, (ProfIqFreeCallback)_mam_userdata_free, data);
    }

    iq_send_stanza(iq);
    xmpp_stanza_release(iq);

    return;
}

void
iq_mam_request(ProfChatWin* win, GDateTime* enddate)
{
    ProfMessage* last_msg = log_database_get_limits_info(win->barejid, TRUE);
    GDateTime* startdate = g_date_time_add_seconds(last_msg->timestamp, 0);
    message_free(last_msg);

    // Save request for later if disco items haven't been received yet
    if (!received_disco_items) {
        LateDeliveryUserdata* cur_del_data = malloc(sizeof(LateDeliveryUserdata));
        cur_del_data->win = win;
        cur_del_data->enddate = enddate;
        cur_del_data->startdate = startdate;
        late_delivery_windows = g_slist_append(late_delivery_windows, cur_del_data);
        log_debug("Save MAM request of %s for later", win->barejid);
        return;
    }

    _iq_mam_request(win, startdate, enddate);

    return;
}

static int
_mam_rsm_id_handler(xmpp_stanza_t* const stanza, void* const userdata)
{
    const char* type = xmpp_stanza_get_type(stanza);
    if (g_strcmp0(type, "error") == 0) {
        auto_char char* error_message = stanza_get_error_message(stanza);
        cons_show_error("Server error: %s", error_message);
        log_debug("MAM Error: %s", error_message);
    } else if (g_strcmp0(type, "result") == 0) {
        xmpp_stanza_t* fin = xmpp_stanza_get_child_by_name_and_ns(stanza, STANZA_NAME_FIN, STANZA_NS_MAM2);
        if (fin) {
            gboolean is_complete = g_strcmp0(xmpp_stanza_get_attribute(fin, "complete"), "true") == 0;
            MamRsmUserdata* data = (MamRsmUserdata*)userdata;
            ProfWin* window = (ProfWin*)data->win;

            buffer_remove_entry(window->layout->buffer, 0);

            auto_char char* start_str = NULL;
            if (data->start_datestr) {
                start_str = strdup(data->start_datestr);
                // Convert to iso8601
                start_str[strlen(start_str) - 3] = '\0';
            }
            char* end_str = NULL;
            if (data->end_datestr) {
                end_str = strdup(data->end_datestr);
                // Convert to iso8601
                end_str[strlen(end_str) - 3] = '\0';
            }

            if (is_complete || !data->fetch_next) {
                chatwin_db_history(data->win, is_complete ? NULL : start_str, end_str, TRUE);
                return 0;
            }

            chatwin_db_history(data->win, start_str, end_str, TRUE);

            xmpp_stanza_t* set = xmpp_stanza_get_child_by_name_and_ns(fin, STANZA_TYPE_SET, STANZA_NS_RSM);
            if (set) {
                win_print_loading_history(window);

                auto_char char* firstid = NULL;
                xmpp_stanza_t* first = xmpp_stanza_get_child_by_name(set, STANZA_NAME_FIRST);
                if (first) {
                    firstid = xmpp_stanza_get_text(first);

                    // 4.3.2. send same stanza with set,max stanza
                    xmpp_ctx_t* const ctx = connection_get_ctx();

                    if (data->end_datestr) {
                        free(data->end_datestr);
                        data->end_datestr = NULL;
                    }
                    xmpp_stanza_t* iq = stanza_create_mam_iq(ctx, data->barejid, data->start_datestr, NULL, firstid, NULL);

                    MamRsmUserdata* ndata = malloc(sizeof(*ndata));
                    *ndata = *data;
                    if (data->end_datestr)
                        ndata->end_datestr = strdup(data->end_datestr);
                    if (data->start_datestr)
                        ndata->start_datestr = strdup(data->start_datestr);
                    if (data->barejid)
                        ndata->barejid = strdup(data->barejid);
                    iq_id_handler_add(xmpp_stanza_get_id(iq), _mam_rsm_id_handler, (ProfIqFreeCallback)_mam_userdata_free, ndata);

                    iq_send_stanza(iq);
                    xmpp_stanza_release(iq);
                }
            }
        }
    }

    return 0;
}

void
iq_register_change_password(const char* const user, const char* const password)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    xmpp_stanza_t* iq = stanza_change_password(ctx, user, password);

    const char* id = xmpp_stanza_get_id(iq);
    iq_id_handler_add(id, _register_change_password_result_id_handler, NULL, NULL);

    iq_send_stanza(iq);
    xmpp_stanza_release(iq);
}

static int
_register_change_password_result_id_handler(xmpp_stanza_t* const stanza, void* const userdata)
{
    const char* type = xmpp_stanza_get_type(stanza);
    if (g_strcmp0(type, "error") == 0) {
        auto_char char* error_message = stanza_get_error_message(stanza);
        cons_show_error("Server error: %s", error_message);
        log_debug("Password change error: %s", error_message);
    } else {
        cons_show("Password successfully changed.");
        log_debug("Password successfully changed.");
    }
    return 0;
}

static int
_muc_register_nick_response_handler(xmpp_stanza_t* const stanza, void* const userdata)
{
    const char* type = xmpp_stanza_get_type(stanza);
    if (g_strcmp0(type, STANZA_TYPE_ERROR) == 0) {
        xmpp_stanza_t* error = xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_ERROR);
        const char* errtype = xmpp_stanza_get_type(error);
        if (errtype) {
            if (g_strcmp0(errtype, STANZA_TYPE_CANCEL) == 0) {
                // find reason
                xmpp_stanza_t* reason = xmpp_stanza_get_child_by_name_and_ns(stanza, STANZA_NAME_CONFLICT, STANZA_NS_STANZAS);
                if (reason) {
                    cons_show_error("Error while registering nickname: nick already registered with MUC");
                    log_debug("Error while registering nickname: nick already registered with MUC");
                } else {
                    xmpp_stanza_t* reason = xmpp_stanza_get_child_by_name_and_ns(stanza, STANZA_NAME_SERVICE_UNAVAILABLE, STANZA_NS_STANZAS);
                    if (reason) {
                        cons_show_error("Error while registering nickname: registration not supported by MUC");
                        log_debug("Error while registering nickname: registration not supported by MUC");
                    }
                }
            } else if (g_strcmp0(errtype, STANZA_TYPE_MODIFY) == 0) {
                xmpp_stanza_t* reason = xmpp_stanza_get_child_by_name_and_ns(stanza, STANZA_NAME_BAD_REQUEST, STANZA_NS_STANZAS);
                if (reason) {
                    cons_show_error("Error while registering nickname: invalid form");
                    log_debug("Error while registering nickname: invalid form");
                }
            }
        }
    } else if (g_strcmp0(type, STANZA_TYPE_RESULT) == 0) {
        cons_show("Registration request successfully received");
        log_debug("Registration request successfully received");
    }

    return 0;
}

void
iq_submit_muc_register_nick_form(ProfConfWin* confwin)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_ctx_t* const ctx = connection_get_ctx();

    xmpp_stanza_t* iq = stanza_create_muc_register_nick(ctx, id, confwin->roomjid, NULL, confwin->form);

    iq_id_handler_add(id, _muc_register_nick_response_handler, NULL, NULL);

    iq_send_stanza(iq);
    xmpp_stanza_release(iq);
}

static int
_muc_register_nick_handler(xmpp_stanza_t* const stanza, void* const userdata)
{
    const char* type = xmpp_stanza_get_type(stanza);

    // error case
    if (g_strcmp0(type, STANZA_TYPE_ERROR) == 0) {
        xmpp_stanza_t* error = xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_ERROR);
        const char* errtype = xmpp_stanza_get_type(error);
        if (errtype) {
            if (g_strcmp0(errtype, STANZA_TYPE_CANCEL) == 0) {
                // find reason
                xmpp_stanza_t* reason = xmpp_stanza_get_child_by_name_and_ns(stanza, STANZA_NAME_ITEM_NOT_FOUND, STANZA_NS_STANZAS);
                if (reason) {
                    cons_show_error("Error while registering nickname: room does not exist");
                    log_debug("Error while registering nickname: room does not exist");
                } else {
                    xmpp_stanza_t* reason = xmpp_stanza_get_child_by_name_and_ns(stanza, STANZA_NAME_NOT_ALLOWED, STANZA_NS_STANZAS);
                    if (reason) {
                        cons_show_error("Error while registering nickname: not allowed to register");
                        log_debug("Error while registering nickname: not allowed to register");
                    }
                }
            }
        }
    } else if (g_strcmp0(type, STANZA_TYPE_RESULT) == 0) {
        xmpp_stanza_t* query = xmpp_stanza_get_child_by_name_and_ns(stanza, STANZA_NAME_QUERY, STANZA_NS_REGISTER);
        if (query) {
            // user might already be registered
            xmpp_stanza_t* username = xmpp_stanza_get_child_by_name(query, STANZA_NAME_USERNAME);
            if (username) {
                const char* value = xmpp_stanza_get_text(username);
                cons_show("User already registered: %s", value);
            } else {
                xmpp_stanza_t* x_st = xmpp_stanza_get_child_by_name_and_ns(query, STANZA_NAME_X, STANZA_NS_DATA);

                if (x_st) {
                    const char* from = xmpp_stanza_get_from(stanza);

                    DataForm* form = form_create(x_st);
                    ProfConfWin* confwin = (ProfConfWin*)wins_new_config(from, form, iq_submit_muc_register_nick_form, NULL, NULL);
                    confwin_handle_configuration(confwin, form);
                }
            }
        }
    }
    return 0;
}

void
iq_muc_register_nick(const char* const roomjid)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();

    auto_char char* id = connection_create_stanza_id();

    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_GET, id);
    xmpp_stanza_set_to(iq, roomjid);

    xmpp_stanza_t* query = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(query, STANZA_NAME_QUERY);
    xmpp_stanza_set_ns(query, STANZA_NS_REGISTER);
    xmpp_stanza_add_child(iq, query);

    iq_id_handler_add(id, _muc_register_nick_handler, NULL, NULL);

    iq_send_stanza(iq);

    xmpp_stanza_release(iq);
    xmpp_stanza_release(query);
}

void
publish_user_mood(const char* const mood, const char* const text)
{
    xmpp_ctx_t* const ctx = connection_get_ctx();
    char* id = connection_create_stanza_id();

    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_SET, id);

    xmpp_stanza_t* pubsub = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(pubsub, STANZA_NAME_PUBSUB);
    xmpp_stanza_set_ns(pubsub, STANZA_NS_PUBSUB);
    xmpp_stanza_add_child(iq, pubsub);

    xmpp_stanza_t* publish = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(publish, STANZA_NAME_PUBLISH);
    xmpp_stanza_set_attribute(publish, STANZA_ATTR_NODE, STANZA_NS_MOOD);
    xmpp_stanza_add_child(pubsub, publish);

    xmpp_stanza_t* item = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(item, STANZA_NAME_ITEM);
    xmpp_stanza_set_attribute(item, "id", "current");
    xmpp_stanza_add_child(publish, item);

    xmpp_stanza_t* mood_t = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(mood_t, STANZA_NAME_MOOD);
    xmpp_stanza_set_ns(mood_t, STANZA_NS_MOOD);
    xmpp_stanza_add_child(item, mood_t);

    xmpp_stanza_t* x = xmpp_stanza_new(ctx);
    xmpp_stanza_t* text_t = xmpp_stanza_new(ctx);
    xmpp_stanza_t* t = xmpp_stanza_new(ctx);
    if (mood) {
        xmpp_stanza_set_name(x, mood);
        xmpp_stanza_add_child(mood_t, x);

        xmpp_stanza_set_name(text_t, STANZA_NAME_TEXT);
        xmpp_stanza_add_child(mood_t, text_t);

        xmpp_stanza_set_text(t, text);
        xmpp_stanza_add_child(text_t, t);
    }
    iq_send_stanza(iq);

    xmpp_stanza_release(iq);
    xmpp_stanza_release(pubsub);
    xmpp_stanza_release(publish);
    xmpp_stanza_release(item);
    xmpp_stanza_release(mood_t);
    xmpp_stanza_release(x);
    xmpp_stanza_release(text_t);
    xmpp_stanza_release(t);
}
