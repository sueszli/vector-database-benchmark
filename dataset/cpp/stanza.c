/*
 * stanza.c
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
#include <libgen.h>
#include <inttypes.h>
#include <assert.h>

#include <glib.h>

#include <strophe.h>

#include "common.h"
#include "log.h"
#include "xmpp/session.h"
#include "xmpp/stanza.h"
#include "xmpp/capabilities.h"
#include "xmpp/connection.h"
#include "xmpp/form.h"
#include "xmpp/muc.h"
#include "database.h"

static void _stanza_add_unique_id(xmpp_stanza_t* stanza);
static gchar* _stanza_create_sha1_hash(char* str);

#if 0
xmpp_stanza_t*
stanza_create_bookmarks_pubsub_request(xmpp_ctx_t *ctx)
{
    xmpp_stanza_t *iq = xmpp_iq_new(ctx, STANZA_TYPE_GET, NULL);

    xmpp_stanza_t *pubsub = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(pubsub, STANZA_NAME_PUBSUB);
    xmpp_stanza_set_ns(pubsub, STANZA_NS_PUBSUB);

    xmpp_stanza_t *items = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(items, STANZA_NAME_ITEMS);
    xmpp_stanza_set_attribute(items, "node", "storage:bookmarks");

    xmpp_stanza_add_child(pubsub, items);
    xmpp_stanza_add_child(iq, pubsub);
    xmpp_stanza_release(items);
    xmpp_stanza_release(pubsub);

    return iq;
}
#endif

xmpp_stanza_t*
stanza_create_bookmarks_storage_request(xmpp_ctx_t* ctx)
{
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_GET, NULL);
    xmpp_stanza_set_ns(iq, "jabber:client");

    xmpp_stanza_t* query = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(query, STANZA_NAME_QUERY);
    xmpp_stanza_set_ns(query, "jabber:iq:private");

    xmpp_stanza_t* storage = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(storage, STANZA_NAME_STORAGE);
    xmpp_stanza_set_ns(storage, "storage:bookmarks");

    xmpp_stanza_add_child(query, storage);
    xmpp_stanza_add_child(iq, query);
    xmpp_stanza_release(storage);
    xmpp_stanza_release(query);

    return iq;
}

xmpp_stanza_t*
stanza_create_blocked_list_request(xmpp_ctx_t* ctx)
{
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_GET, NULL);

    xmpp_stanza_t* blocklist = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(blocklist, STANZA_NAME_BLOCKLIST);
    xmpp_stanza_set_ns(blocklist, STANZA_NS_BLOCKING);

    xmpp_stanza_add_child(iq, blocklist);
    xmpp_stanza_release(blocklist);

    return iq;
}

#if 0
xmpp_stanza_t*
stanza_create_bookmarks_pubsub_add(xmpp_ctx_t *ctx, const char *const jid,
    const gboolean autojoin, const char *const nick)
{
    auto_char char *id = connection_create_stanza_id();
    xmpp_stanza_t *stanza = xmpp_iq_new(ctx, STANZA_TYPE_SET, id);

    xmpp_stanza_t *pubsub = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(pubsub, STANZA_NAME_PUBSUB);
    xmpp_stanza_set_ns(pubsub, STANZA_NS_PUBSUB);
    xmpp_stanza_add_child(stanza, pubsub);

    xmpp_stanza_t *publish = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(publish, STANZA_NAME_PUBLISH);
    xmpp_stanza_set_attribute(publish, STANZA_ATTR_NODE, "storage:bookmarks");
    xmpp_stanza_add_child(pubsub, publish);

    xmpp_stanza_t *item = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(item, STANZA_NAME_ITEM);
    xmpp_stanza_add_child(publish, item);

    xmpp_stanza_t *conference = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(conference, STANZA_NAME_CONFERENCE);
    xmpp_stanza_set_ns(conference, "storage:bookmarks");
    xmpp_stanza_set_attribute(conference, STANZA_ATTR_JID, jid);

    if (autojoin) {
        xmpp_stanza_set_attribute(conference, STANZA_ATTR_AUTOJOIN, "true");
    } else {
        xmpp_stanza_set_attribute(conference, STANZA_ATTR_AUTOJOIN, "false");
    }

    if (nick) {
        xmpp_stanza_t *nick_st = xmpp_stanza_new(ctx);
        xmpp_stanza_set_name(nick_st, STANZA_NAME_NICK);
        xmpp_stanza_set_text(nick_st, nick);
        xmpp_stanza_add_child(conference, nick_st);
    }

    xmpp_stanza_add_child(item, conference);

    xmpp_stanza_t *publish_options = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(publish_options, STANZA_NAME_PUBLISH_OPTIONS);
    xmpp_stanza_add_child(pubsub, publish_options);

    xmpp_stanza_t *x = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(x, STANZA_NAME_X);
    xmpp_stanza_set_ns(x, STANZA_NS_DATA);
    xmpp_stanza_set_type(x, "submit");
    xmpp_stanza_add_child(publish_options, x);

    xmpp_stanza_t *form_type = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(form_type, STANZA_NAME_FIELD);
    xmpp_stanza_set_attribute(form_type, STANZA_ATTR_VAR, "FORM_TYPE");
    xmpp_stanza_set_type(form_type, "hidden");
    xmpp_stanza_t *form_type_value = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(form_type_value, STANZA_NAME_VALUE);
    xmpp_stanza_t *form_type_value_text = xmpp_stanza_new(ctx);
    xmpp_stanza_set_text(form_type_value_text, "http://jabber.org/protocol/pubsub#publish-options");
    xmpp_stanza_add_child(form_type_value, form_type_value_text);
    xmpp_stanza_add_child(form_type, form_type_value);
    xmpp_stanza_add_child(x, form_type);

    xmpp_stanza_t *persist_items = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(persist_items, STANZA_NAME_FIELD);
    xmpp_stanza_set_attribute(persist_items, STANZA_ATTR_VAR, "pubsub#persist_items");
    xmpp_stanza_t *persist_items_value = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(persist_items_value, STANZA_NAME_VALUE);
    xmpp_stanza_t *persist_items_value_text = xmpp_stanza_new(ctx);
    xmpp_stanza_set_text(persist_items_value_text, "true");
    xmpp_stanza_add_child(persist_items_value, persist_items_value_text);
    xmpp_stanza_add_child(persist_items, persist_items_value);
    xmpp_stanza_add_child(x, persist_items);

    xmpp_stanza_t *access_model = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(access_model, STANZA_NAME_FIELD);
    xmpp_stanza_set_attribute(access_model, STANZA_ATTR_VAR, "pubsub#access_model");
    xmpp_stanza_t *access_model_value = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(access_model_value, STANZA_NAME_VALUE);
    xmpp_stanza_t *access_model_value_text = xmpp_stanza_new(ctx);
    xmpp_stanza_set_text(access_model_value_text, "whitelist");
    xmpp_stanza_add_child(access_model_value, access_model_value_text);
    xmpp_stanza_add_child(access_model, access_model_value);
    xmpp_stanza_add_child(x, access_model);

    return stanza;
}
#endif

xmpp_stanza_t*
stanza_create_http_upload_request(xmpp_ctx_t* ctx, const char* const id,
                                  const char* const jid, HTTPUpload* upload)
{
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_GET, id);
    xmpp_stanza_set_to(iq, jid);

    xmpp_stanza_t* request = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(request, STANZA_NAME_REQUEST);
    xmpp_stanza_set_ns(request, STANZA_NS_HTTP_UPLOAD);

    auto_char char* filename_cpy = strdup(upload->filename);
    // strip spaces from filename (servers don't spaces)
    for (int i = 0; i < strlen(filename_cpy); i++) {
        if (filename_cpy[i] == ' ') {
            filename_cpy[i] = '_';
        }
    }
    xmpp_stanza_set_attribute(request, STANZA_ATTR_FILENAME, basename(filename_cpy));

    auto_gchar gchar* filesize = g_strdup_printf("%jd", (intmax_t)(upload->filesize));
    if (filesize) {
        xmpp_stanza_set_attribute(request, STANZA_ATTR_SIZE, filesize);
    }

    xmpp_stanza_set_attribute(request, STANZA_ATTR_CONTENTTYPE, upload->mime_type);

    xmpp_stanza_add_child(iq, request);
    xmpp_stanza_release(request);

    return iq;
}

xmpp_stanza_t*
stanza_enable_carbons(xmpp_ctx_t* ctx)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_SET, id);

    xmpp_stanza_t* carbons_enable = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(carbons_enable, STANZA_NAME_ENABLE);
    xmpp_stanza_set_ns(carbons_enable, STANZA_NS_CARBONS);

    xmpp_stanza_add_child(iq, carbons_enable);
    xmpp_stanza_release(carbons_enable);

    return iq;
}

xmpp_stanza_t*
stanza_disable_carbons(xmpp_ctx_t* ctx)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_SET, id);

    xmpp_stanza_t* carbons_disable = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(carbons_disable, STANZA_NAME_DISABLE);
    xmpp_stanza_set_ns(carbons_disable, STANZA_NS_CARBONS);

    xmpp_stanza_add_child(iq, carbons_disable);
    xmpp_stanza_release(carbons_disable);

    return iq;
}

xmpp_stanza_t*
stanza_create_chat_state(xmpp_ctx_t* ctx, const char* const fulljid, const char* const state)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* msg = xmpp_message_new(ctx, STANZA_TYPE_CHAT, fulljid, id);

    xmpp_stanza_t* chat_state = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(chat_state, state);
    xmpp_stanza_set_ns(chat_state, STANZA_NS_CHATSTATES);
    xmpp_stanza_add_child(msg, chat_state);
    xmpp_stanza_release(chat_state);

    return msg;
}

xmpp_stanza_t*
stanza_create_room_subject_message(xmpp_ctx_t* ctx, const char* const room, const char* const subject)
{
    xmpp_stanza_t* msg = xmpp_message_new(ctx, STANZA_TYPE_GROUPCHAT, room, NULL);

    xmpp_stanza_t* subject_st = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(subject_st, STANZA_NAME_SUBJECT);
    if (subject) {
        xmpp_stanza_t* subject_text = xmpp_stanza_new(ctx);
        xmpp_stanza_set_text(subject_text, subject);
        xmpp_stanza_add_child(subject_st, subject_text);
        xmpp_stanza_release(subject_text);
    }

    xmpp_stanza_add_child(msg, subject_st);
    xmpp_stanza_release(subject_st);

    return msg;
}

xmpp_stanza_t*
stanza_attach_state(xmpp_ctx_t* ctx, xmpp_stanza_t* stanza, const char* const state)
{
    xmpp_stanza_t* chat_state = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(chat_state, state);
    xmpp_stanza_set_ns(chat_state, STANZA_NS_CHATSTATES);
    xmpp_stanza_add_child(stanza, chat_state);
    xmpp_stanza_release(chat_state);

    return stanza;
}

xmpp_stanza_t*
stanza_attach_carbons_private(xmpp_ctx_t* ctx, xmpp_stanza_t* stanza)
{
    xmpp_stanza_t* private_carbon = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(private_carbon, "private");
    xmpp_stanza_set_ns(private_carbon, STANZA_NS_CARBONS);
    xmpp_stanza_add_child(stanza, private_carbon);
    xmpp_stanza_release(private_carbon);

    return stanza;
}

xmpp_stanza_t*
stanza_attach_hints_no_copy(xmpp_ctx_t* ctx, xmpp_stanza_t* stanza)
{
    xmpp_stanza_t* no_copy = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(no_copy, "no-copy");
    xmpp_stanza_set_ns(no_copy, STANZA_NS_HINTS);
    xmpp_stanza_add_child(stanza, no_copy);
    xmpp_stanza_release(no_copy);

    return stanza;
}

xmpp_stanza_t*
stanza_attach_hints_no_store(xmpp_ctx_t* ctx, xmpp_stanza_t* stanza)
{
    xmpp_stanza_t* no_store = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(no_store, "no-store");
    xmpp_stanza_set_ns(no_store, STANZA_NS_HINTS);
    xmpp_stanza_add_child(stanza, no_store);
    xmpp_stanza_release(no_store);

    return stanza;
}

xmpp_stanza_t*
stanza_attach_hints_store(xmpp_ctx_t* ctx, xmpp_stanza_t* stanza)
{
    xmpp_stanza_t* store = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(store, "store");
    xmpp_stanza_set_ns(store, STANZA_NS_HINTS);
    xmpp_stanza_add_child(stanza, store);
    xmpp_stanza_release(store);

    return stanza;
}

xmpp_stanza_t*
stanza_attach_receipt_request(xmpp_ctx_t* ctx, xmpp_stanza_t* stanza)
{
    xmpp_stanza_t* receipet_request = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(receipet_request, "request");
    xmpp_stanza_set_ns(receipet_request, STANZA_NS_RECEIPTS);
    xmpp_stanza_add_child(stanza, receipet_request);
    xmpp_stanza_release(receipet_request);

    return stanza;
}

xmpp_stanza_t*
stanza_attach_x_oob_url(xmpp_ctx_t* ctx, xmpp_stanza_t* stanza, const char* const url)
{
    xmpp_stanza_t* x_oob = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(x_oob, STANZA_NAME_X);
    xmpp_stanza_set_ns(x_oob, STANZA_NS_X_OOB);

    xmpp_stanza_t* surl = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(surl, STANZA_NAME_URL);

    xmpp_stanza_t* surl_txt = xmpp_stanza_new(ctx);
    xmpp_stanza_set_text(surl_txt, url);
    xmpp_stanza_add_child(surl, surl_txt);
    xmpp_stanza_release(surl_txt);

    xmpp_stanza_add_child(x_oob, surl);
    xmpp_stanza_release(surl);

    xmpp_stanza_add_child(stanza, x_oob);
    xmpp_stanza_release(x_oob);

    return stanza;
}

xmpp_stanza_t*
stanza_create_roster_remove_set(xmpp_ctx_t* ctx, const char* const barejid)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_SET, id);

    xmpp_stanza_t* query = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(query, STANZA_NAME_QUERY);
    xmpp_stanza_set_ns(query, XMPP_NS_ROSTER);

    xmpp_stanza_t* item = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(item, STANZA_NAME_ITEM);
    xmpp_stanza_set_attribute(item, STANZA_ATTR_JID, barejid);
    xmpp_stanza_set_attribute(item, STANZA_ATTR_SUBSCRIPTION, "remove");

    xmpp_stanza_add_child(query, item);
    xmpp_stanza_release(item);
    xmpp_stanza_add_child(iq, query);
    xmpp_stanza_release(query);

    return iq;
}

xmpp_stanza_t*
stanza_create_roster_set(xmpp_ctx_t* ctx, const char* const id,
                         const char* const jid, const char* const handle, GSList* groups)
{
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_SET, id);

    xmpp_stanza_t* query = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(query, STANZA_NAME_QUERY);
    xmpp_stanza_set_ns(query, XMPP_NS_ROSTER);

    xmpp_stanza_t* item = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(item, STANZA_NAME_ITEM);
    xmpp_stanza_set_attribute(item, STANZA_ATTR_JID, jid);

    if (handle) {
        xmpp_stanza_set_attribute(item, STANZA_ATTR_NAME, handle);
    } else {
        xmpp_stanza_set_attribute(item, STANZA_ATTR_NAME, "");
    }

    while (groups) {
        xmpp_stanza_t* group = xmpp_stanza_new(ctx);
        xmpp_stanza_t* groupname = xmpp_stanza_new(ctx);
        xmpp_stanza_set_name(group, STANZA_NAME_GROUP);
        xmpp_stanza_set_text(groupname, groups->data);
        xmpp_stanza_add_child(group, groupname);
        xmpp_stanza_release(groupname);
        xmpp_stanza_add_child(item, group);
        xmpp_stanza_release(group);
        groups = g_slist_next(groups);
    }

    xmpp_stanza_add_child(query, item);
    xmpp_stanza_release(item);
    xmpp_stanza_add_child(iq, query);
    xmpp_stanza_release(query);

    return iq;
}

xmpp_stanza_t*
stanza_create_invite(xmpp_ctx_t* ctx, const char* const room,
                     const char* const contact, const char* const reason, const char* const password)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* message = xmpp_message_new(ctx, NULL, contact, id);

    xmpp_stanza_t* x = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(x, STANZA_NAME_X);
    xmpp_stanza_set_ns(x, STANZA_NS_CONFERENCE);

    xmpp_stanza_set_attribute(x, STANZA_ATTR_JID, room);
    if (reason) {
        xmpp_stanza_set_attribute(x, STANZA_ATTR_REASON, reason);
    }
    if (password) {
        xmpp_stanza_set_attribute(x, STANZA_ATTR_PASSWORD, password);
    }

    xmpp_stanza_add_child(message, x);
    xmpp_stanza_release(x);

    return message;
}

xmpp_stanza_t*
stanza_create_mediated_invite(xmpp_ctx_t* ctx, const char* const room,
                              const char* const contact, const char* const reason)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* message = xmpp_message_new(ctx, NULL, room, id);

    xmpp_stanza_t* x = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(x, STANZA_NAME_X);
    xmpp_stanza_set_ns(x, STANZA_NS_MUC_USER);

    xmpp_stanza_t* invite = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(invite, STANZA_NAME_INVITE);
    xmpp_stanza_set_to(invite, contact);

    if (reason) {
        xmpp_stanza_t* reason_st = xmpp_stanza_new(ctx);
        xmpp_stanza_set_name(reason_st, STANZA_NAME_REASON);
        xmpp_stanza_t* reason_txt = xmpp_stanza_new(ctx);
        xmpp_stanza_set_text(reason_txt, reason);
        xmpp_stanza_add_child(reason_st, reason_txt);
        xmpp_stanza_release(reason_txt);
        xmpp_stanza_add_child(invite, reason_st);
        xmpp_stanza_release(reason_st);
    }

    xmpp_stanza_add_child(x, invite);
    xmpp_stanza_release(invite);
    xmpp_stanza_add_child(message, x);
    xmpp_stanza_release(x);

    return message;
}

xmpp_stanza_t*
stanza_create_room_join_presence(xmpp_ctx_t* const ctx,
                                 const char* const full_room_jid, const char* const passwd)
{
    xmpp_stanza_t* presence = xmpp_presence_new(ctx);
    xmpp_stanza_set_to(presence, full_room_jid);
    _stanza_add_unique_id(presence);

    xmpp_stanza_t* x = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(x, STANZA_NAME_X);
    xmpp_stanza_set_ns(x, STANZA_NS_MUC);

    if (passwd) {
        xmpp_stanza_t* pass = xmpp_stanza_new(ctx);
        xmpp_stanza_set_name(pass, "password");
        xmpp_stanza_t* text = xmpp_stanza_new(ctx);
        xmpp_stanza_set_text(text, passwd);
        xmpp_stanza_add_child(pass, text);
        xmpp_stanza_add_child(x, pass);
        xmpp_stanza_release(text);
        xmpp_stanza_release(pass);
    }

    xmpp_stanza_add_child(presence, x);
    xmpp_stanza_release(x);

    return presence;
}

xmpp_stanza_t*
stanza_create_room_newnick_presence(xmpp_ctx_t* ctx,
                                    const char* const full_room_jid)
{
    xmpp_stanza_t* presence = xmpp_presence_new(ctx);
    _stanza_add_unique_id(presence);
    xmpp_stanza_set_to(presence, full_room_jid);

    return presence;
}

xmpp_stanza_t*
stanza_create_room_leave_presence(xmpp_ctx_t* ctx, const char* const room,
                                  const char* const nick)
{
    GString* full_jid = g_string_new(room);
    g_string_append(full_jid, "/");
    g_string_append(full_jid, nick);

    xmpp_stanza_t* presence = xmpp_presence_new(ctx);
    xmpp_stanza_set_type(presence, STANZA_TYPE_UNAVAILABLE);
    xmpp_stanza_set_to(presence, full_jid->str);
    _stanza_add_unique_id(presence);

    g_string_free(full_jid, TRUE);

    return presence;
}

xmpp_stanza_t*
stanza_create_instant_room_request_iq(xmpp_ctx_t* ctx, const char* const room_jid)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_SET, id);
    xmpp_stanza_set_to(iq, room_jid);

    xmpp_stanza_t* query = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(query, STANZA_NAME_QUERY);
    xmpp_stanza_set_ns(query, STANZA_NS_MUC_OWNER);

    xmpp_stanza_t* x = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(x, STANZA_NAME_X);
    xmpp_stanza_set_type(x, "submit");
    xmpp_stanza_set_ns(x, STANZA_NS_DATA);

    xmpp_stanza_add_child(query, x);
    xmpp_stanza_release(x);

    xmpp_stanza_add_child(iq, query);
    xmpp_stanza_release(query);

    return iq;
}

xmpp_stanza_t*
stanza_create_instant_room_destroy_iq(xmpp_ctx_t* ctx, const char* const room_jid)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_SET, id);
    xmpp_stanza_set_to(iq, room_jid);

    xmpp_stanza_t* query = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(query, STANZA_NAME_QUERY);
    xmpp_stanza_set_ns(query, STANZA_NS_MUC_OWNER);

    xmpp_stanza_t* destroy = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(destroy, STANZA_NAME_DESTROY);

    xmpp_stanza_add_child(query, destroy);
    xmpp_stanza_release(destroy);

    xmpp_stanza_add_child(iq, query);
    xmpp_stanza_release(query);

    return iq;
}

xmpp_stanza_t*
stanza_create_room_config_request_iq(xmpp_ctx_t* ctx, const char* const room_jid)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_GET, id);
    xmpp_stanza_set_to(iq, room_jid);

    xmpp_stanza_t* query = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(query, STANZA_NAME_QUERY);
    xmpp_stanza_set_ns(query, STANZA_NS_MUC_OWNER);

    xmpp_stanza_add_child(iq, query);
    xmpp_stanza_release(query);

    return iq;
}

xmpp_stanza_t*
stanza_create_room_config_cancel_iq(xmpp_ctx_t* ctx, const char* const room_jid)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_SET, id);
    xmpp_stanza_set_to(iq, room_jid);

    xmpp_stanza_t* query = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(query, STANZA_NAME_QUERY);
    xmpp_stanza_set_ns(query, STANZA_NS_MUC_OWNER);

    xmpp_stanza_t* x = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(x, STANZA_NAME_X);
    xmpp_stanza_set_type(x, "cancel");
    xmpp_stanza_set_ns(x, STANZA_NS_DATA);

    xmpp_stanza_add_child(query, x);
    xmpp_stanza_release(x);

    xmpp_stanza_add_child(iq, query);
    xmpp_stanza_release(query);

    return iq;
}

xmpp_stanza_t*
stanza_create_room_affiliation_list_iq(xmpp_ctx_t* ctx, const char* const room, const char* const affiliation)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_GET, id);
    xmpp_stanza_set_to(iq, room);

    xmpp_stanza_t* query = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(query, STANZA_NAME_QUERY);
    xmpp_stanza_set_ns(query, STANZA_NS_MUC_ADMIN);

    xmpp_stanza_t* item = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(item, STANZA_NAME_ITEM);
    xmpp_stanza_set_attribute(item, "affiliation", affiliation);

    xmpp_stanza_add_child(query, item);
    xmpp_stanza_release(item);
    xmpp_stanza_add_child(iq, query);
    xmpp_stanza_release(query);

    return iq;
}

xmpp_stanza_t*
stanza_create_room_role_list_iq(xmpp_ctx_t* ctx, const char* const room, const char* const role)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_GET, id);
    xmpp_stanza_set_to(iq, room);

    xmpp_stanza_t* query = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(query, STANZA_NAME_QUERY);
    xmpp_stanza_set_ns(query, STANZA_NS_MUC_ADMIN);

    xmpp_stanza_t* item = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(item, STANZA_NAME_ITEM);
    xmpp_stanza_set_attribute(item, "role", role);

    xmpp_stanza_add_child(query, item);
    xmpp_stanza_release(item);
    xmpp_stanza_add_child(iq, query);
    xmpp_stanza_release(query);

    return iq;
}

xmpp_stanza_t*
stanza_create_room_affiliation_set_iq(xmpp_ctx_t* ctx, const char* const room, const char* const jid,
                                      const char* const affiliation, const char* const reason)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_SET, id);
    xmpp_stanza_set_to(iq, room);

    xmpp_stanza_t* query = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(query, STANZA_NAME_QUERY);
    xmpp_stanza_set_ns(query, STANZA_NS_MUC_ADMIN);

    xmpp_stanza_t* item = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(item, STANZA_NAME_ITEM);
    xmpp_stanza_set_attribute(item, "affiliation", affiliation);
    xmpp_stanza_set_attribute(item, STANZA_ATTR_JID, jid);

    if (reason) {
        xmpp_stanza_t* reason_st = xmpp_stanza_new(ctx);
        xmpp_stanza_set_name(reason_st, STANZA_NAME_REASON);
        xmpp_stanza_t* reason_text = xmpp_stanza_new(ctx);
        xmpp_stanza_set_text(reason_text, reason);
        xmpp_stanza_add_child(reason_st, reason_text);
        xmpp_stanza_release(reason_text);

        xmpp_stanza_add_child(item, reason_st);
        xmpp_stanza_release(reason_st);
    }

    xmpp_stanza_add_child(query, item);
    xmpp_stanza_release(item);
    xmpp_stanza_add_child(iq, query);
    xmpp_stanza_release(query);

    return iq;
}

xmpp_stanza_t*
stanza_create_room_role_set_iq(xmpp_ctx_t* const ctx, const char* const room, const char* const nick,
                               const char* const role, const char* const reason)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_SET, id);
    xmpp_stanza_set_to(iq, room);

    xmpp_stanza_t* query = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(query, STANZA_NAME_QUERY);
    xmpp_stanza_set_ns(query, STANZA_NS_MUC_ADMIN);

    xmpp_stanza_t* item = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(item, STANZA_NAME_ITEM);
    xmpp_stanza_set_attribute(item, "role", role);
    xmpp_stanza_set_attribute(item, STANZA_ATTR_NICK, nick);

    if (reason) {
        xmpp_stanza_t* reason_st = xmpp_stanza_new(ctx);
        xmpp_stanza_set_name(reason_st, STANZA_NAME_REASON);
        xmpp_stanza_t* reason_text = xmpp_stanza_new(ctx);
        xmpp_stanza_set_text(reason_text, reason);
        xmpp_stanza_add_child(reason_st, reason_text);
        xmpp_stanza_release(reason_text);

        xmpp_stanza_add_child(item, reason_st);
        xmpp_stanza_release(reason_st);
    }

    xmpp_stanza_add_child(query, item);
    xmpp_stanza_release(item);
    xmpp_stanza_add_child(iq, query);
    xmpp_stanza_release(query);

    return iq;
}

xmpp_stanza_t*
stanza_create_room_kick_iq(xmpp_ctx_t* const ctx, const char* const room, const char* const nick,
                           const char* const reason)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_SET, id);
    xmpp_stanza_set_to(iq, room);

    xmpp_stanza_t* query = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(query, STANZA_NAME_QUERY);
    xmpp_stanza_set_ns(query, STANZA_NS_MUC_ADMIN);

    xmpp_stanza_t* item = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(item, STANZA_NAME_ITEM);
    xmpp_stanza_set_attribute(item, STANZA_ATTR_NICK, nick);
    xmpp_stanza_set_attribute(item, "role", "none");

    if (reason) {
        xmpp_stanza_t* reason_st = xmpp_stanza_new(ctx);
        xmpp_stanza_set_name(reason_st, STANZA_NAME_REASON);
        xmpp_stanza_t* reason_text = xmpp_stanza_new(ctx);
        xmpp_stanza_set_text(reason_text, reason);
        xmpp_stanza_add_child(reason_st, reason_text);
        xmpp_stanza_release(reason_text);

        xmpp_stanza_add_child(item, reason_st);
        xmpp_stanza_release(reason_st);
    }

    xmpp_stanza_add_child(query, item);
    xmpp_stanza_release(item);
    xmpp_stanza_add_child(iq, query);
    xmpp_stanza_release(query);

    return iq;
}

xmpp_stanza_t*
stanza_create_software_version_iq(xmpp_ctx_t* ctx, const char* const fulljid)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_GET, id);
    xmpp_stanza_set_to(iq, fulljid);

    xmpp_stanza_t* query = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(query, STANZA_NAME_QUERY);
    xmpp_stanza_set_ns(query, STANZA_NS_VERSION);

    xmpp_stanza_add_child(iq, query);
    xmpp_stanza_release(query);

    return iq;
}

xmpp_stanza_t*
stanza_create_roster_iq(xmpp_ctx_t* ctx)
{
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_GET, "roster");

    xmpp_stanza_t* query = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(query, STANZA_NAME_QUERY);
    xmpp_stanza_set_ns(query, XMPP_NS_ROSTER);

    xmpp_stanza_add_child(iq, query);
    xmpp_stanza_release(query);

    return iq;
}

xmpp_stanza_t*
stanza_create_disco_info_iq(xmpp_ctx_t* ctx, const char* const id, const char* const to,
                            const char* const node)
{
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_GET, id);
    xmpp_stanza_set_to(iq, to);

    xmpp_stanza_t* query = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(query, STANZA_NAME_QUERY);
    xmpp_stanza_set_ns(query, XMPP_NS_DISCO_INFO);
    if (node) {
        xmpp_stanza_set_attribute(query, STANZA_ATTR_NODE, node);
    }

    xmpp_stanza_add_child(iq, query);
    xmpp_stanza_release(query);

    return iq;
}

xmpp_stanza_t*
stanza_create_disco_items_iq(xmpp_ctx_t* ctx, const char* const id,
                             const char* const jid, const char* const node)
{
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_GET, id);
    xmpp_stanza_set_to(iq, jid);

    xmpp_stanza_t* query = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(query, STANZA_NAME_QUERY);
    xmpp_stanza_set_ns(query, XMPP_NS_DISCO_ITEMS);
    if (node) {
        xmpp_stanza_set_attribute(query, STANZA_ATTR_NODE, node);
    }

    xmpp_stanza_add_child(iq, query);
    xmpp_stanza_release(query);

    return iq;
}

xmpp_stanza_t*
stanza_create_last_activity_iq(xmpp_ctx_t* ctx, const char* const id, const char* const to)
{
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_GET, id);
    xmpp_stanza_set_to(iq, to);

    xmpp_stanza_t* query = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(query, STANZA_NAME_QUERY);
    xmpp_stanza_set_ns(query, STANZA_NS_LASTACTIVITY);

    xmpp_stanza_add_child(iq, query);
    xmpp_stanza_release(query);

    return iq;
}

xmpp_stanza_t*
stanza_create_room_config_submit_iq(xmpp_ctx_t* ctx, const char* const room, DataForm* form)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_SET, id);
    xmpp_stanza_set_to(iq, room);

    xmpp_stanza_t* query = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(query, STANZA_NAME_QUERY);
    xmpp_stanza_set_ns(query, STANZA_NS_MUC_OWNER);

    xmpp_stanza_t* x = form_create_submission(form);
    xmpp_stanza_add_child(query, x);
    xmpp_stanza_release(x);

    xmpp_stanza_add_child(iq, query);
    xmpp_stanza_release(query);

    return iq;
}

xmpp_stanza_t*
stanza_create_caps_query_element(xmpp_ctx_t* ctx)
{
    xmpp_stanza_t* query = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(query, STANZA_NAME_QUERY);
    xmpp_stanza_set_ns(query, XMPP_NS_DISCO_INFO);

    xmpp_stanza_t* identity = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(identity, "identity");
    xmpp_stanza_set_attribute(identity, "category", "client");

    ProfAccount* account = accounts_get_account(session_get_account_name());
    gchar* client = account->client;
    bool is_custom_client = client != NULL;

    GString* name_str = g_string_new(is_custom_client ? client : "Profanity ");
    if (!is_custom_client) {
        xmpp_stanza_set_type(identity, "console");
        auto_gchar gchar* prof_version = prof_get_version();
        g_string_append(name_str, prof_version);
    }

    account_free(account);

    xmpp_stanza_set_attribute(identity, "name", name_str->str);
    g_string_free(name_str, TRUE);
    xmpp_stanza_add_child(query, identity);
    xmpp_stanza_release(identity);

    GList* features = caps_get_features();
    GList* curr = features;
    while (curr) {
        xmpp_stanza_t* feature = xmpp_stanza_new(ctx);
        xmpp_stanza_set_name(feature, STANZA_NAME_FEATURE);
        xmpp_stanza_set_attribute(feature, STANZA_ATTR_VAR, curr->data);
        xmpp_stanza_add_child(query, feature);
        xmpp_stanza_release(feature);

        curr = g_list_next(curr);
    }
    g_list_free_full(features, free);

    return query;
}

gboolean
stanza_contains_chat_state(xmpp_stanza_t* stanza)
{
    return ((xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_ACTIVE) != NULL) || (xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_COMPOSING) != NULL) || (xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_PAUSED) != NULL) || (xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_GONE) != NULL) || (xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_INACTIVE) != NULL));
}

xmpp_stanza_t*
stanza_create_ping_iq(xmpp_ctx_t* ctx, const char* const target)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_GET, id);
    if (target) {
        xmpp_stanza_set_to(iq, target);
    }

    xmpp_stanza_t* ping = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(ping, STANZA_NAME_PING);

    xmpp_stanza_set_ns(ping, STANZA_NS_PING);

    xmpp_stanza_add_child(iq, ping);
    xmpp_stanza_release(ping);

    return iq;
}

gchar*
stanza_create_caps_sha1_from_query(xmpp_stanza_t* const query)
{
    GSList* identities = NULL;
    GSList* features = NULL;
    GSList* form_names = NULL;
    GHashTable* forms = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, (GDestroyNotify)form_destroy);

    xmpp_stanza_t* child = xmpp_stanza_get_children(query);
    while (child) {
        if (g_strcmp0(xmpp_stanza_get_name(child), STANZA_NAME_IDENTITY) == 0) {
            const char* category = xmpp_stanza_get_attribute(child, "category");
            const char* type = xmpp_stanza_get_type(child);
            const char* lang = xmpp_stanza_get_attribute(child, "xml:lang");
            const char* name = xmpp_stanza_get_attribute(child, "name");

            GString* identity_str = g_string_new(category);
            g_string_append(identity_str, "/");
            if (type) {
                g_string_append(identity_str, type);
            }
            g_string_append(identity_str, "/");
            if (lang) {
                g_string_append(identity_str, lang);
            }
            g_string_append(identity_str, "/");
            if (name) {
                g_string_append(identity_str, name);
            }
            g_string_append(identity_str, "<");
            identities = g_slist_insert_sorted(identities, g_strdup(identity_str->str), (GCompareFunc)g_strcmp0);
            g_string_free(identity_str, TRUE);
        } else if (g_strcmp0(xmpp_stanza_get_name(child), STANZA_NAME_FEATURE) == 0) {
            const char* feature_str = xmpp_stanza_get_attribute(child, "var");
            features = g_slist_insert_sorted(features, g_strdup(feature_str), (GCompareFunc)g_strcmp0);
        } else if (g_strcmp0(xmpp_stanza_get_name(child), STANZA_NAME_X) == 0) {
            if (g_strcmp0(xmpp_stanza_get_ns(child), STANZA_NS_DATA) == 0) {
                DataForm* form = form_create(child);
                char* form_type = form_get_form_type_field(form);
                form_names = g_slist_insert_sorted(form_names, g_strdup(form_type), (GCompareFunc)g_strcmp0);
                g_hash_table_insert(forms, g_strdup(form_type), form);
            }
        }
        child = xmpp_stanza_get_next(child);
    }

    GString* s = g_string_new("");

    GSList* curr = identities;
    while (curr) {
        g_string_append(s, curr->data);
        curr = g_slist_next(curr);
    }

    curr = features;
    while (curr) {
        g_string_append(s, curr->data);
        g_string_append(s, "<");
        curr = g_slist_next(curr);
    }

    curr = form_names;
    while (curr) {
        DataForm* form = g_hash_table_lookup(forms, curr->data);
        char* form_type = form_get_form_type_field(form);
        g_string_append(s, form_type);
        g_string_append(s, "<");

        GSList* sorted_fields = form_get_non_form_type_fields_sorted(form);
        GSList* curr_field = sorted_fields;
        while (curr_field) {
            FormField* field = curr_field->data;
            g_string_append(s, field->var);
            g_string_append(s, "<");

            GSList* sorted_values = form_get_field_values_sorted(field);
            GSList* curr_value = sorted_values;
            while (curr_value) {
                g_string_append(s, curr_value->data);
                g_string_append(s, "<");
                curr_value = g_slist_next(curr_value);
            }
            g_slist_free(sorted_values);
            curr_field = g_slist_next(curr_field);
        }
        g_slist_free(sorted_fields);

        curr = g_slist_next(curr);
    }

    gchar* result = _stanza_create_sha1_hash(s->str);

    g_string_free(s, TRUE);
    g_slist_free_full(identities, g_free);
    g_slist_free_full(features, g_free);
    g_slist_free_full(form_names, g_free);
    g_hash_table_destroy(forms);

    return result;
}

xmpp_stanza_t*
stanza_get_child_by_name_and_from(xmpp_stanza_t* const stanza, const char* const name, const char* const from)
{
    xmpp_stanza_t* child;
    const char* child_from;
    const char* child_name;

    for (child = xmpp_stanza_get_children(stanza); child; child = xmpp_stanza_get_next(child)) {
        child_name = xmpp_stanza_get_name(child);
        if (child_name && g_strcmp0(name, child_name) == 0) {
            child_from = xmpp_stanza_get_attribute(child, STANZA_ATTR_FROM);
            if (child_from && g_strcmp0(from, child_from) == 0) {
                break;
            }
        }
    }

    return child;
}

GDateTime*
stanza_get_delay(xmpp_stanza_t* const stanza)
{
    return stanza_get_delay_from(stanza, NULL);
}

static GDateTime*
_stanza_get_delay_timestamp_xep0203(xmpp_stanza_t* const delay_stanza)
{
    GTimeVal utc_stamp;
    const char* xmlns = xmpp_stanza_get_attribute(delay_stanza, STANZA_ATTR_XMLNS);

    if (xmlns && (g_strcmp0(xmlns, "urn:xmpp:delay") == 0)) {
        const char* stamp = xmpp_stanza_get_attribute(delay_stanza, STANZA_ATTR_STAMP);

        if (stamp && (g_time_val_from_iso8601(stamp, &utc_stamp))) {

            GDateTime* datetime = g_date_time_new_from_iso8601(stamp, NULL);
            GDateTime* local_datetime = g_date_time_to_local(datetime);
            g_date_time_unref(datetime);

            return local_datetime;
        }
    }

    return NULL;
}

static GDateTime*
_stanza_get_delay_timestamp_xep0091(xmpp_stanza_t* const x_stanza)
{
    GTimeVal utc_stamp;
    const char* xmlns = xmpp_stanza_get_attribute(x_stanza, STANZA_ATTR_XMLNS);

    if (xmlns && (g_strcmp0(xmlns, "jabber:x:delay") == 0)) {
        const char* stamp = xmpp_stanza_get_attribute(x_stanza, STANZA_ATTR_STAMP);
        if (stamp && (g_time_val_from_iso8601(stamp, &utc_stamp))) {

            GDateTime* datetime = g_date_time_new_from_iso8601(stamp, NULL);
            GDateTime* local_datetime = g_date_time_to_local(datetime);
            g_date_time_unref(datetime);

            return local_datetime;
        }
    }

    return NULL;
}

GDateTime*
stanza_get_delay_from(xmpp_stanza_t* const stanza, gchar* from)
{
    xmpp_stanza_t* delay = NULL;

    // first check for XEP-0203 delayed delivery
    if (from) {
        delay = stanza_get_child_by_name_and_from(stanza, STANZA_NAME_DELAY, from);
    } else {
        delay = xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_DELAY);
    }

    if (delay) {
        return _stanza_get_delay_timestamp_xep0203(delay);
    }

    // otherwise check for XEP-0091 legacy delayed delivery
    // stamp format : CCYYMMDDThh:mm:ss
    if (from) {
        delay = stanza_get_child_by_name_and_from(stanza, STANZA_NAME_X, from);
    } else {
        delay = xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_X);
    }

    if (delay) {
        return _stanza_get_delay_timestamp_xep0091(delay);
    }

    return NULL;
}

GDateTime*
stanza_get_oldest_delay(xmpp_stanza_t* const stanza)
{
    xmpp_stanza_t* child;
    const char* child_name;
    GDateTime* oldest = NULL;

    for (child = xmpp_stanza_get_children(stanza); child; child = xmpp_stanza_get_next(child)) {

        child_name = xmpp_stanza_get_name(child);

        if (child_name && g_strcmp0(child_name, STANZA_NAME_DELAY) == 0) {
            GDateTime* tmp = _stanza_get_delay_timestamp_xep0203(child);

            if (oldest == NULL) {
                oldest = tmp;
            } else if (g_date_time_compare(oldest, tmp) == 1) {
                g_date_time_unref(oldest);
                oldest = tmp;
            } else {
                g_date_time_unref(tmp);
            }
        }

        if (child_name && g_strcmp0(child_name, STANZA_NAME_X) == 0) {
            GDateTime* tmp = _stanza_get_delay_timestamp_xep0091(child);

            if (oldest == NULL) {
                oldest = tmp;
            } else if (g_date_time_compare(oldest, tmp) == 1) {
                g_date_time_unref(oldest);
                oldest = tmp;
            } else {
                g_date_time_unref(tmp);
            }
        }
    }

    return oldest;
}

char*
stanza_get_status(xmpp_stanza_t* stanza, char* def)
{
    xmpp_stanza_t* status = xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_STATUS);

    if (status) {
        return stanza_text_strdup(status);
    } else if (def) {
        return strdup(def);
    } else {
        return NULL;
    }
}

char*
stanza_get_show(xmpp_stanza_t* stanza, char* def)
{
    xmpp_stanza_t* show = xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_SHOW);

    if (show) {
        return stanza_text_strdup(show);
    } else if (def) {
        return strdup(def);
    } else {
        return NULL;
    }
}

gboolean
stanza_is_muc_presence(xmpp_stanza_t* const stanza)
{
    if (stanza == NULL) {
        return FALSE;
    }
    if (g_strcmp0(xmpp_stanza_get_name(stanza), STANZA_NAME_PRESENCE) != 0) {
        return FALSE;
    }

    xmpp_stanza_t* x = xmpp_stanza_get_child_by_ns(stanza, STANZA_NS_MUC_USER);

    if (x) {
        return TRUE;
    } else {
        return FALSE;
    }
}

gboolean
stanza_muc_requires_config(xmpp_stanza_t* const stanza)
{
    // no stanza, or not presence stanza
    if ((stanza == NULL) || (g_strcmp0(xmpp_stanza_get_name(stanza), STANZA_NAME_PRESENCE) != 0)) {
        return FALSE;
    }

    // muc user namespaced x element
    xmpp_stanza_t* x = xmpp_stanza_get_child_by_ns(stanza, STANZA_NS_MUC_USER);
    if (x == NULL) {
        return FALSE;
    }

    // check for item element with owner affiliation
    xmpp_stanza_t* item = xmpp_stanza_get_child_by_name(x, "item");
    if (item == NULL) {
        return FALSE;
    }
    const char* affiliation = xmpp_stanza_get_attribute(item, "affiliation");
    if (g_strcmp0(affiliation, "owner") != 0) {
        return FALSE;
    }

    // check for status code 201
    xmpp_stanza_t* x_children = xmpp_stanza_get_children(x);
    while (x_children) {
        if (g_strcmp0(xmpp_stanza_get_name(x_children), STANZA_NAME_STATUS) == 0) {
            const char* code = xmpp_stanza_get_attribute(x_children, STANZA_ATTR_CODE);
            if (g_strcmp0(code, "201") == 0) {
                return TRUE;
            }
        }
        x_children = xmpp_stanza_get_next(x_children);
    }

    return FALSE;
}

gboolean
stanza_is_muc_self_presence(xmpp_stanza_t* const stanza,
                            const char* const self_jid)
{
    // no stanza, or not presence stanza
    if ((stanza == NULL) || (g_strcmp0(xmpp_stanza_get_name(stanza), STANZA_NAME_PRESENCE) != 0)) {
        return FALSE;
    }

    // muc user namespaced x element
    xmpp_stanza_t* x = xmpp_stanza_get_child_by_ns(stanza, STANZA_NS_MUC_USER);

    if (x == NULL) {
        return FALSE;
    }

    // check for status child element with 110 code
    xmpp_stanza_t* x_children = xmpp_stanza_get_children(x);
    while (x_children) {
        if (g_strcmp0(xmpp_stanza_get_name(x_children), STANZA_NAME_STATUS) == 0) {
            const char* code = xmpp_stanza_get_attribute(x_children, STANZA_ATTR_CODE);
            if (g_strcmp0(code, "110") == 0) {
                return TRUE;
            }
        }
        x_children = xmpp_stanza_get_next(x_children);
    }

    // check for item child element with jid property
    xmpp_stanza_t* item = xmpp_stanza_get_child_by_name(x, STANZA_NAME_ITEM);
    if (item) {
        const char* jid = xmpp_stanza_get_attribute(item, STANZA_ATTR_JID);
        if (jid) {
            if (g_str_has_prefix(self_jid, jid)) {
                return TRUE;
            }
        }
    }

    // check if 'from' attribute identifies this user
    const char* from = xmpp_stanza_get_from(stanza);
    if (from) {
        auto_jid Jid* from_jid = jid_create(from);
        if (muc_active(from_jid->barejid)) {
            char* nick = muc_nick(from_jid->barejid);
            if (g_strcmp0(from_jid->resourcepart, nick) == 0) {
                return TRUE;
            }
        }

        // check if a new nickname maps to a pending nick change for this user
        if (muc_nick_change_pending(from_jid->barejid)) {
            char* new_nick = from_jid->resourcepart;
            if (new_nick) {
                char* nick = muc_nick(from_jid->barejid);
                char* old_nick = muc_old_nick(from_jid->barejid, new_nick);
                if (g_strcmp0(old_nick, nick) == 0) {
                    return TRUE;
                }
            }
        }
    }

    // self presence not found
    return FALSE;
}

GSList*
stanza_get_status_codes_by_ns(xmpp_stanza_t* const stanza, char* ns)
{
    xmpp_stanza_t* ns_child = xmpp_stanza_get_child_by_ns(stanza, ns);
    if (ns_child == NULL) {
        return NULL;
    }

    GSList* codes = NULL;
    xmpp_stanza_t* child = xmpp_stanza_get_children(ns_child);
    while (child) {
        const char* name = xmpp_stanza_get_name(child);
        if (g_strcmp0(name, STANZA_NAME_STATUS) == 0) {
            const char* code = xmpp_stanza_get_attribute(child, STANZA_ATTR_CODE);
            if (code) {
                codes = g_slist_append(codes, strdup(code));
            }
        }
        child = xmpp_stanza_get_next(child);
    }

    return codes;
}

gboolean
stanza_room_destroyed(xmpp_stanza_t* stanza)
{
    const char* stanza_name = xmpp_stanza_get_name(stanza);
    if (g_strcmp0(stanza_name, STANZA_NAME_PRESENCE) != 0) {
        return FALSE;
    }

    xmpp_stanza_t* x = xmpp_stanza_get_child_by_ns(stanza, STANZA_NS_MUC_USER);
    if (x == NULL) {
        return FALSE;
    }

    xmpp_stanza_t* destroy = xmpp_stanza_get_child_by_name(x, STANZA_NAME_DESTROY);
    if (destroy) {
        return TRUE;
    }

    return FALSE;
}

const char*
stanza_get_muc_destroy_alternative_room(xmpp_stanza_t* stanza)
{
    const char* stanza_name = xmpp_stanza_get_name(stanza);
    if (g_strcmp0(stanza_name, STANZA_NAME_PRESENCE) != 0) {
        return NULL;
    }

    xmpp_stanza_t* x = xmpp_stanza_get_child_by_ns(stanza, STANZA_NS_MUC_USER);
    if (x == NULL) {
        return NULL;
    }

    xmpp_stanza_t* destroy = xmpp_stanza_get_child_by_name(x, STANZA_NAME_DESTROY);
    if (destroy == NULL) {
        return NULL;
    }

    const char* jid = xmpp_stanza_get_attribute(destroy, STANZA_ATTR_JID);

    return jid;
}

char*
stanza_get_muc_destroy_alternative_password(xmpp_stanza_t* stanza)
{
    const char* stanza_name = xmpp_stanza_get_name(stanza);
    if (g_strcmp0(stanza_name, STANZA_NAME_PRESENCE) != 0) {
        return NULL;
    }

    xmpp_stanza_t* x = xmpp_stanza_get_child_by_ns(stanza, STANZA_NS_MUC_USER);
    if (x == NULL) {
        return NULL;
    }

    xmpp_stanza_t* destroy = xmpp_stanza_get_child_by_name(x, STANZA_NAME_DESTROY);
    if (destroy == NULL) {
        return NULL;
    }

    xmpp_stanza_t* password_st = xmpp_stanza_get_child_by_name(destroy, STANZA_NAME_PASSWORD);
    if (password_st == NULL) {
        return NULL;
    }

    return stanza_text_strdup(password_st);
}

char*
stanza_get_muc_destroy_reason(xmpp_stanza_t* stanza)
{
    const char* stanza_name = xmpp_stanza_get_name(stanza);
    if (g_strcmp0(stanza_name, STANZA_NAME_PRESENCE) != 0) {
        return NULL;
    }

    xmpp_stanza_t* x = xmpp_stanza_get_child_by_ns(stanza, STANZA_NS_MUC_USER);
    if (x == NULL) {
        return NULL;
    }

    xmpp_stanza_t* destroy = xmpp_stanza_get_child_by_name(x, STANZA_NAME_DESTROY);
    if (destroy == NULL) {
        return NULL;
    }

    xmpp_stanza_t* reason_st = xmpp_stanza_get_child_by_name(destroy, STANZA_NAME_REASON);
    if (reason_st == NULL) {
        return NULL;
    }

    return stanza_text_strdup(reason_st);
}

const char*
stanza_get_actor(xmpp_stanza_t* stanza)
{
    const char* stanza_name = xmpp_stanza_get_name(stanza);
    if (g_strcmp0(stanza_name, STANZA_NAME_PRESENCE) != 0) {
        return NULL;
    }

    xmpp_stanza_t* x = xmpp_stanza_get_child_by_ns(stanza, STANZA_NS_MUC_USER);
    if (x == NULL) {
        return NULL;
    }

    xmpp_stanza_t* item = xmpp_stanza_get_child_by_name(x, STANZA_NAME_ITEM);
    if (item == NULL) {
        return NULL;
    }

    xmpp_stanza_t* actor = xmpp_stanza_get_child_by_name(item, STANZA_NAME_ACTOR);
    if (actor == NULL) {
        return NULL;
    }

    const char* nick = xmpp_stanza_get_attribute(actor, STANZA_ATTR_NICK);
    if (nick) {
        return nick;
    }

    const char* jid = xmpp_stanza_get_attribute(actor, STANZA_ATTR_JID);
    if (jid) {
        return jid;
    }

    return NULL;
}

char*
stanza_get_reason(xmpp_stanza_t* stanza)
{
    const char* stanza_name = xmpp_stanza_get_name(stanza);
    if (g_strcmp0(stanza_name, STANZA_NAME_PRESENCE) != 0) {
        return NULL;
    }

    xmpp_stanza_t* x = xmpp_stanza_get_child_by_ns(stanza, STANZA_NS_MUC_USER);
    if (x == NULL) {
        return NULL;
    }

    xmpp_stanza_t* item = xmpp_stanza_get_child_by_name(x, STANZA_NAME_ITEM);
    if (item == NULL) {
        return NULL;
    }

    xmpp_stanza_t* reason_st = xmpp_stanza_get_child_by_name(item, STANZA_NAME_REASON);
    if (reason_st == NULL) {
        return NULL;
    }

    return stanza_text_strdup(reason_st);
}

gboolean
stanza_is_room_nick_change(xmpp_stanza_t* const stanza)
{
    // no stanza, or not presence stanza
    if ((stanza == NULL) || (g_strcmp0(xmpp_stanza_get_name(stanza), STANZA_NAME_PRESENCE) != 0)) {
        return FALSE;
    }

    // muc user namespaced x element
    xmpp_stanza_t* x = xmpp_stanza_get_child_by_ns(stanza, STANZA_NS_MUC_USER);
    if (x == NULL) {
        return FALSE;
    }

    // check for status child element with 303 code
    xmpp_stanza_t* x_children = xmpp_stanza_get_children(x);
    while (x_children) {
        if (g_strcmp0(xmpp_stanza_get_name(x_children), STANZA_NAME_STATUS) == 0) {
            const char* code = xmpp_stanza_get_attribute(x_children, STANZA_ATTR_CODE);
            if (g_strcmp0(code, "303") == 0) {
                return TRUE;
            }
        }
        x_children = xmpp_stanza_get_next(x_children);
    }

    return FALSE;
}

const char*
stanza_get_new_nick(xmpp_stanza_t* const stanza)
{
    if (!stanza_is_room_nick_change(stanza)) {
        return NULL;
    }

    xmpp_stanza_t* x = xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_X);
    xmpp_stanza_t* x_children = xmpp_stanza_get_children(x);

    while (x_children) {
        if (g_strcmp0(xmpp_stanza_get_name(x_children), STANZA_NAME_ITEM) == 0) {
            const char* nick = xmpp_stanza_get_attribute(x_children, STANZA_ATTR_NICK);
            if (nick) {
                return nick;
            }
        }
        x_children = xmpp_stanza_get_next(x_children);
    }

    return NULL;
}

int
stanza_get_idle_time(xmpp_stanza_t* const stanza)
{
    xmpp_stanza_t* query = xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_QUERY);

    if (query == NULL) {
        return 0;
    }

    const char* ns = xmpp_stanza_get_ns(query);
    if (ns == NULL) {
        return 0;
    }

    if (g_strcmp0(ns, STANZA_NS_LASTACTIVITY) != 0) {
        return 0;
    }

    const char* seconds_str = xmpp_stanza_get_attribute(query, STANZA_ATTR_SECONDS);
    if (seconds_str == NULL) {
        return 0;
    }

    int result = atoi(seconds_str);
    if (result < 1) {
        return 0;
    } else {
        return result;
    }
}

XMPPCaps*
stanza_parse_caps(xmpp_stanza_t* const stanza)
{
    xmpp_stanza_t* caps_st = xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_C);

    if (!caps_st) {
        return NULL;
    }

    const char* ns = xmpp_stanza_get_ns(caps_st);
    if (g_strcmp0(ns, STANZA_NS_CAPS) != 0) {
        return NULL;
    }

    const char* hash = xmpp_stanza_get_attribute(caps_st, STANZA_ATTR_HASH);
    const char* node = xmpp_stanza_get_attribute(caps_st, STANZA_ATTR_NODE);
    const char* ver = xmpp_stanza_get_attribute(caps_st, STANZA_ATTR_VER);

    XMPPCaps* caps = (XMPPCaps*)malloc(sizeof(XMPPCaps));
    caps->hash = hash ? strdup(hash) : NULL;
    caps->node = node ? strdup(node) : NULL;
    caps->ver = ver ? strdup(ver) : NULL;

    return caps;
}

EntityCapabilities*
stanza_create_caps_from_query_element(xmpp_stanza_t* query)
{
    auto_char char* software = NULL;
    auto_char char* software_version = NULL;
    auto_char char* os = NULL;
    auto_char char* os_version = NULL;

    xmpp_stanza_t* softwareinfo = xmpp_stanza_get_child_by_ns(query, STANZA_NS_DATA);
    if (softwareinfo) {
        DataForm* form = form_create(softwareinfo);
        FormField* formField = NULL;

        char* form_type = form_get_form_type_field(form);
        if (g_strcmp0(form_type, STANZA_DATAFORM_SOFTWARE) == 0) {
            GSList* field = form->fields;
            while (field) {
                formField = field->data;
                if (formField->values) {
                    if (g_strcmp0(formField->var, "software") == 0) {
                        if (software == NULL) {
                            software = strdup(formField->values->data);
                        }
                    } else if (g_strcmp0(formField->var, "software_version") == 0) {
                        if (software_version == NULL) {
                            software_version = strdup(formField->values->data);
                        }
                    } else if (g_strcmp0(formField->var, "os") == 0) {
                        if (os == NULL) {
                            os = strdup(formField->values->data);
                        }
                    } else if (g_strcmp0(formField->var, "os_version") == 0) {
                        if (os_version == NULL) {
                            os_version = strdup(formField->values->data);
                        }
                    }
                }
                field = g_slist_next(field);
            }
        }

        form_destroy(form);
    }

    xmpp_stanza_t* child = xmpp_stanza_get_children(query);
    GSList* identity_stanzas = NULL;
    GSList* features = NULL;
    while (child) {
        if (g_strcmp0(xmpp_stanza_get_name(child), "feature") == 0) {
            features = g_slist_append(features, strdup(xmpp_stanza_get_attribute(child, "var")));
        }
        if (g_strcmp0(xmpp_stanza_get_name(child), "identity") == 0) {
            identity_stanzas = g_slist_append(identity_stanzas, child);
        }

        child = xmpp_stanza_get_next(child);
    }

    // find identity by locale
    const gchar* const* langs = g_get_language_names();
    int num_langs = g_strv_length((gchar**)langs);
    xmpp_stanza_t* found = NULL;
    GSList* curr_identity = identity_stanzas;
    while (curr_identity) {
        xmpp_stanza_t* id_stanza = curr_identity->data;
        const char* stanza_lang = xmpp_stanza_get_attribute(id_stanza, "xml:lang");
        if (stanza_lang) {
            for (int i = 0; i < num_langs; i++) {
                if (g_strcmp0(langs[i], stanza_lang) == 0) {
                    found = id_stanza;
                    break;
                }
            }
        }
        if (found) {
            break;
        }
        curr_identity = g_slist_next(curr_identity);
    }

    // not lang match, use default with no lang
    if (!found) {
        curr_identity = identity_stanzas;
        while (curr_identity) {
            xmpp_stanza_t* id_stanza = curr_identity->data;
            const char* stanza_lang = xmpp_stanza_get_attribute(id_stanza, "xml:lang");
            if (!stanza_lang) {
                found = id_stanza;
                break;
            }

            curr_identity = g_slist_next(curr_identity);
        }
    }

    // no matching lang, no identity without lang, use first
    if (!found) {
        if (identity_stanzas) {
            found = identity_stanzas->data;
        }
    }

    g_slist_free(identity_stanzas);

    const char* category = NULL;
    const char* type = NULL;
    const char* name = NULL;
    if (found) {
        category = xmpp_stanza_get_attribute(found, "category");
        type = xmpp_stanza_get_type(found);
        name = xmpp_stanza_get_attribute(found, "name");
    }

    EntityCapabilities* result = caps_create(category, type, name, software, software_version, os, os_version, features);
    g_slist_free_full(features, free);

    return result;
}

char*
stanza_get_error_message(xmpp_stanza_t* stanza)
{
    xmpp_stanza_t* error_stanza = xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_ERROR);

    // return nothing if no error stanza
    if (error_stanza == NULL) {
        return strdup("unknown");
    }

    // check for text child
    xmpp_stanza_t* text_stanza = xmpp_stanza_get_child_by_name(error_stanza, STANZA_NAME_TEXT);

    // check for text
    if (text_stanza) {
        char* err_msg = stanza_text_strdup(text_stanza);
        if (err_msg) {
            return err_msg;
        }

        // otherwise check each defined-condition RFC-6120 8.3.3
    } else {
        gchar* defined_conditions[] = {
            STANZA_NAME_BAD_REQUEST,
            STANZA_NAME_CONFLICT,
            STANZA_NAME_FEATURE_NOT_IMPLEMENTED,
            STANZA_NAME_FORBIDDEN,
            STANZA_NAME_GONE,
            STANZA_NAME_INTERNAL_SERVER_ERROR,
            STANZA_NAME_ITEM_NOT_FOUND,
            STANZA_NAME_JID_MALFORMED,
            STANZA_NAME_NOT_ACCEPTABLE,
            STANZA_NAME_NOT_ALLOWED,
            STANZA_NAME_NOT_AUTHORISED,
            STANZA_NAME_POLICY_VIOLATION,
            STANZA_NAME_RECIPIENT_UNAVAILABLE,
            STANZA_NAME_REDIRECT,
            STANZA_NAME_REGISTRATION_REQUIRED,
            STANZA_NAME_REMOTE_SERVER_NOT_FOUND,
            STANZA_NAME_REMOTE_SERVER_TIMEOUT,
            STANZA_NAME_RESOURCE_CONSTRAINT,
            STANZA_NAME_SERVICE_UNAVAILABLE,
            STANZA_NAME_SUBSCRIPTION_REQUIRED,
            STANZA_NAME_UNEXPECTED_REQUEST
        };

        for (int i = 0; i < ARRAY_SIZE(defined_conditions); i++) {
            xmpp_stanza_t* cond_stanza = xmpp_stanza_get_child_by_name(error_stanza, defined_conditions[i]);
            if (cond_stanza) {
                char* result = strdup(xmpp_stanza_get_name(cond_stanza));
                return result;
            }
        }
    }

    // if undefined-condition or no condition, return nothing
    return strdup("unknown");
}

// Note that the `count' must be 2 * number of key/value pairs
void
stanza_attach_publish_options_va(xmpp_ctx_t* const ctx, xmpp_stanza_t* const iq, int count, ...)
{
    xmpp_stanza_t* publish_options = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(publish_options, STANZA_NAME_PUBLISH_OPTIONS);

    xmpp_stanza_t* x = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(x, STANZA_NAME_X);
    xmpp_stanza_set_ns(x, STANZA_NS_DATA);
    xmpp_stanza_set_type(x, "submit");
    xmpp_stanza_add_child(publish_options, x);

    xmpp_stanza_t* form_type = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(form_type, STANZA_NAME_FIELD);
    xmpp_stanza_set_attribute(form_type, STANZA_ATTR_VAR, "FORM_TYPE");
    xmpp_stanza_set_type(form_type, "hidden");
    xmpp_stanza_t* form_type_value = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(form_type_value, STANZA_NAME_VALUE);
    xmpp_stanza_t* form_type_value_text = xmpp_stanza_new(ctx);
    xmpp_stanza_set_text(form_type_value_text, XMPP_FEATURE_PUBSUB_PUBLISH_OPTIONS);
    xmpp_stanza_add_child(form_type_value, form_type_value_text);
    xmpp_stanza_add_child(form_type, form_type_value);
    xmpp_stanza_add_child(x, form_type);

    xmpp_stanza_t* pubsub = xmpp_stanza_get_child_by_ns(iq, STANZA_NS_PUBSUB);
    xmpp_stanza_add_child(pubsub, publish_options);

    va_list ap;
    va_start(ap, count);
    int j;
    for (j = 0; j < count; j += 2) {
        const char* const option = va_arg(ap, char* const);
        const char* const value = va_arg(ap, char* const);

        xmpp_stanza_t* field = xmpp_stanza_new(ctx);
        xmpp_stanza_set_name(field, STANZA_NAME_FIELD);
        xmpp_stanza_set_attribute(field, STANZA_ATTR_VAR, option);
        xmpp_stanza_t* field_value = xmpp_stanza_new(ctx);
        xmpp_stanza_set_name(field_value, STANZA_NAME_VALUE);
        xmpp_stanza_t* field_value_text = xmpp_stanza_new(ctx);
        xmpp_stanza_set_text(field_value_text, value);
        xmpp_stanza_add_child(field_value, field_value_text);
        xmpp_stanza_add_child(field, field_value);
        xmpp_stanza_add_child(x, field);

        xmpp_stanza_release(field_value_text);
        xmpp_stanza_release(field_value);
        xmpp_stanza_release(field);
    }
    va_end(ap);

    xmpp_stanza_release(form_type_value_text);
    xmpp_stanza_release(form_type_value);
    xmpp_stanza_release(form_type);
    xmpp_stanza_release(x);
    xmpp_stanza_release(publish_options);
}

void
stanza_attach_publish_options(xmpp_ctx_t* const ctx, xmpp_stanza_t* const iq, const char* const option, const char* const value)
{
    stanza_attach_publish_options_va(ctx, iq, 2, option, value);
}

void
stanza_attach_priority(xmpp_ctx_t* const ctx, xmpp_stanza_t* const presence, const int pri)
{
    if (pri == 0) {
        return;
    }

    char pri_str[10];
    snprintf(pri_str, sizeof(pri_str), "%d", pri);

    xmpp_stanza_t* priority = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(priority, STANZA_NAME_PRIORITY);

    xmpp_stanza_t* value = xmpp_stanza_new(ctx);
    xmpp_stanza_set_text(value, pri_str);

    xmpp_stanza_add_child(priority, value);
    xmpp_stanza_release(value);

    xmpp_stanza_add_child(presence, priority);
    xmpp_stanza_release(priority);
}

void
stanza_attach_show(xmpp_ctx_t* const ctx, xmpp_stanza_t* const presence,
                   const char* const show)
{
    if (show == NULL) {
        return;
    }

    xmpp_stanza_t* show_stanza = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(show_stanza, STANZA_NAME_SHOW);
    xmpp_stanza_t* text = xmpp_stanza_new(ctx);
    xmpp_stanza_set_text(text, show);
    xmpp_stanza_add_child(show_stanza, text);
    xmpp_stanza_add_child(presence, show_stanza);
    xmpp_stanza_release(text);
    xmpp_stanza_release(show_stanza);
}

void
stanza_attach_status(xmpp_ctx_t* const ctx, xmpp_stanza_t* const presence,
                     const char* const status)
{
    if (status == NULL) {
        return;
    }

    xmpp_stanza_t* status_stanza = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(status_stanza, STANZA_NAME_STATUS);
    xmpp_stanza_t* text = xmpp_stanza_new(ctx);
    xmpp_stanza_set_text(text, status);
    xmpp_stanza_add_child(status_stanza, text);
    xmpp_stanza_add_child(presence, status_stanza);
    xmpp_stanza_release(text);
    xmpp_stanza_release(status_stanza);
}

void
stanza_attach_last_activity(xmpp_ctx_t* const ctx,
                            xmpp_stanza_t* const presence, const int idle)
{
    xmpp_stanza_t* query = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(query, STANZA_NAME_QUERY);
    xmpp_stanza_set_ns(query, STANZA_NS_LASTACTIVITY);
    char idle_str[10];
    snprintf(idle_str, sizeof(idle_str), "%d", idle);
    xmpp_stanza_set_attribute(query, STANZA_ATTR_SECONDS, idle_str);
    xmpp_stanza_add_child(presence, query);
    xmpp_stanza_release(query);
}

void
stanza_attach_caps(xmpp_ctx_t* const ctx, xmpp_stanza_t* const presence)
{
    xmpp_stanza_t* caps = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(caps, STANZA_NAME_C);
    xmpp_stanza_set_ns(caps, STANZA_NS_CAPS);
    xmpp_stanza_t* query = stanza_create_caps_query_element(ctx);

    char* sha1 = caps_get_my_sha1(ctx);
    xmpp_stanza_set_attribute(caps, STANZA_ATTR_HASH, "sha-1");
    xmpp_stanza_set_attribute(caps, STANZA_ATTR_NODE, "http://profanity-im.github.io");
    xmpp_stanza_set_attribute(caps, STANZA_ATTR_VER, sha1);
    xmpp_stanza_add_child(presence, caps);
    xmpp_stanza_release(caps);
    xmpp_stanza_release(query);
}

const char*
stanza_get_presence_string_from_type(resource_presence_t presence_type)
{
    switch (presence_type) {
    case RESOURCE_AWAY:
        return STANZA_TEXT_AWAY;
    case RESOURCE_DND:
        return STANZA_TEXT_DND;
    case RESOURCE_CHAT:
        return STANZA_TEXT_CHAT;
    case RESOURCE_XA:
        return STANZA_TEXT_XA;
    default:
        return NULL;
    }
}

Resource*
stanza_resource_from_presence(XMPPPresence* presence)
{
    // create Resource
    Resource* resource = NULL;
    resource_presence_t resource_presence = resource_presence_from_string(presence->show);
    if (presence->jid->resourcepart == NULL) { // hack for servers that do not send full jid
        resource = resource_new("__prof_default", resource_presence, presence->status, presence->priority);
    } else {
        resource = resource_new(presence->jid->resourcepart, resource_presence, presence->status, presence->priority);
    }

    return resource;
}

char*
stanza_text_strdup(xmpp_stanza_t* stanza)
{
    xmpp_ctx_t* ctx = connection_get_ctx();

    char* string = NULL;
    char* stanza_text = xmpp_stanza_get_text(stanza);
    if (stanza_text) {
        string = strdup(stanza_text);
        xmpp_free(ctx, stanza_text);
    }

    return string;
}

void
stanza_free_caps(XMPPCaps* caps)
{
    if (caps == NULL) {
        return;
    }

    if (caps->hash) {
        free(caps->hash);
    }
    if (caps->node) {
        free(caps->node);
    }
    if (caps->ver) {
        free(caps->ver);
    }
    FREE_SET_NULL(caps);
}

void
stanza_free_presence(XMPPPresence* presence)
{
    if (presence == NULL) {
        return;
    }

    if (presence->jid) {
        jid_destroy(presence->jid);
    }
    if (presence->last_activity) {
        g_date_time_unref(presence->last_activity);
    }
    if (presence->show) {
        free(presence->show);
    }
    if (presence->status) {
        free(presence->status);
    }
    FREE_SET_NULL(presence);
}

XMPPPresence*
stanza_parse_presence(xmpp_stanza_t* stanza, int* err)
{
    const char* from = xmpp_stanza_get_from(stanza);
    if (!from) {
        *err = STANZA_PARSE_ERROR_NO_FROM;
        return NULL;
    }

    Jid* from_jid = jid_create(from);
    if (!from_jid) {
        *err = STANZA_PARSE_ERROR_INVALID_FROM;
        return NULL;
    }

    XMPPPresence* result = (XMPPPresence*)malloc(sizeof(XMPPPresence));
    result->jid = from_jid;

    result->show = stanza_get_show(stanza, "online");
    result->status = stanza_get_status(stanza, NULL);

    int idle_seconds = stanza_get_idle_time(stanza);
    if (idle_seconds > 0) {
        GDateTime* now = g_date_time_new_now_local();
        result->last_activity = g_date_time_add_seconds(now, 0 - idle_seconds);
        g_date_time_unref(now);
    } else {
        result->last_activity = NULL;
    }

    result->priority = 0;
    xmpp_stanza_t* priority_stanza = xmpp_stanza_get_child_by_name(stanza, STANZA_NAME_PRIORITY);
    if (priority_stanza) {
        char* priority_str = xmpp_stanza_get_text(priority_stanza);
        if (priority_str) {
            result->priority = atoi(priority_str);
        }
        xmpp_ctx_t* ctx = connection_get_ctx();
        xmpp_free(ctx, priority_str);
    }

    return result;
}

xmpp_stanza_t*
stanza_create_command_exec_iq(xmpp_ctx_t* ctx, const char* const target,
                              const char* const node)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_SET, id);
    xmpp_stanza_set_to(iq, target);

    xmpp_stanza_t* command = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(command, STANZA_NAME_COMMAND);

    xmpp_stanza_set_ns(command, STANZA_NS_COMMAND);
    xmpp_stanza_set_attribute(command, "node", node);
    xmpp_stanza_set_attribute(command, "action", "execute");

    xmpp_stanza_add_child(iq, command);
    xmpp_stanza_release(command);

    return iq;
}

xmpp_stanza_t*
stanza_create_command_config_submit_iq(xmpp_ctx_t* ctx, const char* const room,
                                       const char* const node, const char* const sessionid, DataForm* form)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_SET, id);
    xmpp_stanza_set_to(iq, room);

    xmpp_stanza_t* command = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(command, STANZA_NAME_COMMAND);
    xmpp_stanza_set_ns(command, STANZA_NS_COMMAND);
    xmpp_stanza_set_attribute(command, "node", node);
    if (sessionid != NULL) {
        xmpp_stanza_set_attribute(command, "sessionid", sessionid);
    }

    xmpp_stanza_t* x = form_create_submission(form);
    xmpp_stanza_add_child(command, x);
    xmpp_stanza_release(x);

    xmpp_stanza_add_child(iq, command);
    xmpp_stanza_release(command);

    return iq;
}

xmpp_stanza_t*
stanza_create_omemo_devicelist_request(xmpp_ctx_t* ctx, const char* const id,
                                       const char* const jid)
{
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_GET, id);
    xmpp_stanza_set_to(iq, jid);

    xmpp_stanza_t* pubsub = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(pubsub, STANZA_NAME_PUBSUB);
    xmpp_stanza_set_ns(pubsub, STANZA_NS_PUBSUB);

    xmpp_stanza_t* items = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(items, "items");
    xmpp_stanza_set_attribute(items, STANZA_ATTR_NODE, STANZA_NS_OMEMO_DEVICELIST);

    xmpp_stanza_add_child(pubsub, items);
    xmpp_stanza_add_child(iq, pubsub);

    xmpp_stanza_release(items);
    xmpp_stanza_release(pubsub);

    return iq;
}

xmpp_stanza_t*
stanza_create_omemo_devicelist_subscribe(xmpp_ctx_t* ctx, const char* const jid)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_SET, id);

    xmpp_stanza_t* pubsub = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(pubsub, STANZA_NAME_PUBSUB);
    xmpp_stanza_set_ns(pubsub, STANZA_NS_PUBSUB);

    xmpp_stanza_t* subscribe = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(subscribe, STANZA_NAME_SUBSCRIBE);
    xmpp_stanza_set_attribute(subscribe, STANZA_ATTR_NODE, STANZA_NS_OMEMO_DEVICELIST);
    xmpp_stanza_set_attribute(subscribe, "jid", jid);

    xmpp_stanza_add_child(pubsub, subscribe);
    xmpp_stanza_add_child(iq, pubsub);

    xmpp_stanza_release(subscribe);
    xmpp_stanza_release(pubsub);

    return iq;
}

xmpp_stanza_t*
stanza_create_omemo_devicelist_publish(xmpp_ctx_t* ctx, GList* const ids)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_SET, id);

    xmpp_stanza_t* pubsub = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(pubsub, STANZA_NAME_PUBSUB);
    xmpp_stanza_set_ns(pubsub, STANZA_NS_PUBSUB);

    xmpp_stanza_t* publish = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(publish, STANZA_NAME_PUBLISH);
    xmpp_stanza_set_attribute(publish, STANZA_ATTR_NODE, STANZA_NS_OMEMO_DEVICELIST);

    xmpp_stanza_t* item = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(item, STANZA_NAME_ITEM);
    xmpp_stanza_set_attribute(item, "id", "current");

    xmpp_stanza_t* list = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(list, "list");
    xmpp_stanza_set_ns(list, "eu.siacs.conversations.axolotl");

    for (GList* i = ids; i != NULL; i = i->next) {
        xmpp_stanza_t* device = xmpp_stanza_new(ctx);
        xmpp_stanza_set_name(device, "device");
        auto_gchar gchar* id = g_strdup_printf("%d", GPOINTER_TO_INT(i->data));
        xmpp_stanza_set_attribute(device, "id", id);

        xmpp_stanza_add_child(list, device);
        xmpp_stanza_release(device);
    }

    xmpp_stanza_add_child(item, list);
    xmpp_stanza_add_child(publish, item);
    xmpp_stanza_add_child(pubsub, publish);
    xmpp_stanza_add_child(iq, pubsub);

    xmpp_stanza_release(list);
    xmpp_stanza_release(item);
    xmpp_stanza_release(publish);
    xmpp_stanza_release(pubsub);

    return iq;
}

xmpp_stanza_t*
stanza_create_omemo_bundle_publish(xmpp_ctx_t* ctx, const char* const id,
                                   uint32_t device_id,
                                   const unsigned char* const identity_key, size_t identity_key_length,
                                   const unsigned char* const signed_prekey, size_t signed_prekey_length,
                                   const unsigned char* const signed_prekey_signature, size_t signed_prekey_signature_length,
                                   GList* const prekeys, GList* const prekeys_id, GList* const prekeys_length)
{
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_SET, id);

    xmpp_stanza_t* pubsub = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(pubsub, STANZA_NAME_PUBSUB);
    xmpp_stanza_set_ns(pubsub, STANZA_NS_PUBSUB);

    xmpp_stanza_t* publish = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(publish, STANZA_NAME_PUBLISH);
    auto_gchar gchar* node = g_strdup_printf("%s:%d", "eu.siacs.conversations.axolotl.bundles", device_id);
    xmpp_stanza_set_attribute(publish, STANZA_ATTR_NODE, node);

    xmpp_stanza_t* item = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(item, STANZA_NAME_ITEM);
    xmpp_stanza_set_attribute(item, "id", "current");

    xmpp_stanza_t* bundle = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(bundle, "bundle");
    xmpp_stanza_set_ns(bundle, "eu.siacs.conversations.axolotl");

    xmpp_stanza_t* signed_prekey_public_stanza = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(signed_prekey_public_stanza, "signedPreKeyPublic");
    xmpp_stanza_set_attribute(signed_prekey_public_stanza, "signedPreKeyId", "1");

    xmpp_stanza_t* signed_prekey_public_stanza_text = xmpp_stanza_new(ctx);
    auto_gchar gchar* signed_prekey_b64 = g_base64_encode(signed_prekey, signed_prekey_length);
    xmpp_stanza_set_text(signed_prekey_public_stanza_text, signed_prekey_b64);
    xmpp_stanza_add_child(signed_prekey_public_stanza, signed_prekey_public_stanza_text);
    xmpp_stanza_release(signed_prekey_public_stanza_text);

    xmpp_stanza_t* signed_prekey_signature_stanza = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(signed_prekey_signature_stanza, "signedPreKeySignature");

    xmpp_stanza_t* signed_prekey_signature_stanza_text = xmpp_stanza_new(ctx);
    auto_gchar gchar* signed_prekey_signature_b64 = g_base64_encode(signed_prekey_signature, signed_prekey_signature_length);
    xmpp_stanza_set_text(signed_prekey_signature_stanza_text, signed_prekey_signature_b64);
    xmpp_stanza_add_child(signed_prekey_signature_stanza, signed_prekey_signature_stanza_text);
    xmpp_stanza_release(signed_prekey_signature_stanza_text);

    xmpp_stanza_t* identity_key_stanza = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(identity_key_stanza, "identityKey");

    xmpp_stanza_t* identity_key_stanza_text = xmpp_stanza_new(ctx);
    auto_gchar gchar* identity_key_b64 = g_base64_encode(identity_key, identity_key_length);
    xmpp_stanza_set_text(identity_key_stanza_text, identity_key_b64);
    xmpp_stanza_add_child(identity_key_stanza, identity_key_stanza_text);
    xmpp_stanza_release(identity_key_stanza_text);

    xmpp_stanza_t* prekeys_stanza = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(prekeys_stanza, "prekeys");

    GList *p, *i, *l;
    for (p = prekeys, i = prekeys_id, l = prekeys_length; p != NULL; p = p->next, i = i->next, l = l->next) {
        xmpp_stanza_t* prekey = xmpp_stanza_new(ctx);
        xmpp_stanza_set_name(prekey, "preKeyPublic");
        auto_gchar gchar* id = g_strdup_printf("%d", GPOINTER_TO_INT(i->data));
        xmpp_stanza_set_attribute(prekey, "preKeyId", id);

        xmpp_stanza_t* prekey_text = xmpp_stanza_new(ctx);
        auto_gchar gchar* prekey_b64 = g_base64_encode(p->data, GPOINTER_TO_INT(l->data));
        xmpp_stanza_set_text(prekey_text, prekey_b64);

        xmpp_stanza_add_child(prekey, prekey_text);
        xmpp_stanza_add_child(prekeys_stanza, prekey);
        xmpp_stanza_release(prekey_text);
        xmpp_stanza_release(prekey);
    }

    xmpp_stanza_add_child(bundle, signed_prekey_public_stanza);
    xmpp_stanza_add_child(bundle, signed_prekey_signature_stanza);
    xmpp_stanza_add_child(bundle, identity_key_stanza);
    xmpp_stanza_add_child(bundle, prekeys_stanza);
    xmpp_stanza_add_child(item, bundle);
    xmpp_stanza_add_child(publish, item);
    xmpp_stanza_add_child(pubsub, publish);
    xmpp_stanza_add_child(iq, pubsub);

    xmpp_stanza_release(signed_prekey_public_stanza);
    xmpp_stanza_release(signed_prekey_signature_stanza);
    xmpp_stanza_release(identity_key_stanza);
    xmpp_stanza_release(prekeys_stanza);
    xmpp_stanza_release(bundle);
    xmpp_stanza_release(item);
    xmpp_stanza_release(publish);
    xmpp_stanza_release(pubsub);

    return iq;
}

xmpp_stanza_t*
stanza_create_omemo_bundle_request(xmpp_ctx_t* ctx, const char* const id, const char* const jid, uint32_t device_id)
{
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_GET, id);
    xmpp_stanza_set_to(iq, jid);

    xmpp_stanza_t* pubsub = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(pubsub, STANZA_NAME_PUBSUB);
    xmpp_stanza_set_ns(pubsub, STANZA_NS_PUBSUB);

    xmpp_stanza_t* items = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(items, "items");
    auto_gchar gchar* node = g_strdup_printf("%s:%d", STANZA_NS_OMEMO_BUNDLES, device_id);
    xmpp_stanza_set_attribute(items, STANZA_ATTR_NODE, node);

    xmpp_stanza_add_child(pubsub, items);
    xmpp_stanza_add_child(iq, pubsub);

    xmpp_stanza_release(items);
    xmpp_stanza_release(pubsub);

    return iq;
}

xmpp_stanza_t*
stanza_create_pubsub_configure_request(xmpp_ctx_t* ctx, const char* const id, const char* const jid, const char* const node)
{
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_GET, id);
    xmpp_stanza_set_to(iq, jid);

    xmpp_stanza_t* pubsub = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(pubsub, STANZA_NAME_PUBSUB);
    xmpp_stanza_set_ns(pubsub, STANZA_NS_PUBSUB_OWNER);

    xmpp_stanza_t* configure = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(configure, STANZA_NAME_CONFIGURE);
    xmpp_stanza_set_attribute(configure, STANZA_ATTR_NODE, node);

    xmpp_stanza_add_child(pubsub, configure);
    xmpp_stanza_add_child(iq, pubsub);

    xmpp_stanza_release(configure);
    xmpp_stanza_release(pubsub);

    return iq;
}

xmpp_stanza_t*
stanza_create_pubsub_configure_submit(xmpp_ctx_t* ctx, const char* const id, const char* const jid, const char* const node, DataForm* form)
{
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_SET, id);
    xmpp_stanza_set_to(iq, jid);

    xmpp_stanza_t* pubsub = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(pubsub, STANZA_NAME_PUBSUB);
    xmpp_stanza_set_ns(pubsub, STANZA_NS_PUBSUB_OWNER);

    xmpp_stanza_t* configure = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(configure, STANZA_NAME_CONFIGURE);
    xmpp_stanza_set_attribute(configure, STANZA_ATTR_NODE, node);

    xmpp_stanza_t* x = form_create_submission(form);

    xmpp_stanza_add_child(configure, x);
    xmpp_stanza_add_child(pubsub, configure);
    xmpp_stanza_add_child(iq, pubsub);

    xmpp_stanza_release(x);
    xmpp_stanza_release(configure);
    xmpp_stanza_release(pubsub);

    return iq;
}

xmpp_stanza_t*
stanza_attach_origin_id(xmpp_ctx_t* ctx, xmpp_stanza_t* stanza, const char* const id)
{
    xmpp_stanza_t* origin_id = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(origin_id, STANZA_NAME_ORIGIN_ID);
    xmpp_stanza_set_ns(origin_id, STANZA_NS_STABLE_ID);
    xmpp_stanza_set_attribute(origin_id, STANZA_ATTR_ID, id);

    xmpp_stanza_add_child(stanza, origin_id);

    xmpp_stanza_release(origin_id);

    return stanza;
}

static void
_stanza_add_unique_id(xmpp_stanza_t* stanza)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_set_id(stanza, id);
}

static gchar*
_stanza_create_sha1_hash(char* str)
{
    unsigned char* digest = (unsigned char*)malloc(XMPP_SHA1_DIGEST_SIZE);
    assert(digest != NULL);

    xmpp_sha1_digest((unsigned char*)str, strlen(str), digest);

    gchar* b64 = g_base64_encode(digest, XMPP_SHA1_DIGEST_SIZE);
    assert(b64 != NULL);
    free(digest);

    return b64;
}

xmpp_stanza_t*
stanza_create_avatar_retrieve_data_request(xmpp_ctx_t* ctx, const char* stanza_id, const char* const item_id, const char* const jid)
{
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_GET, stanza_id);
    xmpp_stanza_set_to(iq, jid);

    xmpp_stanza_t* pubsub = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(pubsub, STANZA_NAME_PUBSUB);
    xmpp_stanza_set_ns(pubsub, STANZA_NS_PUBSUB);

    xmpp_stanza_t* items = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(items, "items");
    auto_gchar gchar* node = g_strdup_printf("%s", STANZA_NS_USER_AVATAR_DATA);
    xmpp_stanza_set_attribute(items, STANZA_ATTR_NODE, node);

    xmpp_stanza_t* item = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(item, STANZA_NAME_ITEM);
    xmpp_stanza_set_attribute(item, "id", item_id);

    xmpp_stanza_add_child(items, item);
    xmpp_stanza_add_child(pubsub, items);
    xmpp_stanza_add_child(iq, pubsub);

    xmpp_stanza_release(item);
    xmpp_stanza_release(items);
    xmpp_stanza_release(pubsub);

    return iq;
}

xmpp_stanza_t*
stanza_create_avatar_data_publish_iq(xmpp_ctx_t* ctx, const char* img_data, gsize len)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_SET, id);
    xmpp_stanza_set_attribute(iq, STANZA_ATTR_FROM, connection_get_fulljid());

    xmpp_stanza_t* pubsub = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(pubsub, STANZA_NAME_PUBSUB);
    xmpp_stanza_set_ns(pubsub, STANZA_NS_PUBSUB);

    xmpp_stanza_t* publish = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(publish, STANZA_NAME_PUBLISH);
    xmpp_stanza_set_attribute(publish, STANZA_ATTR_NODE, STANZA_NS_USER_AVATAR_DATA);

    xmpp_stanza_t* item = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(item, STANZA_NAME_ITEM);
    char* sha1 = xmpp_sha1(ctx, (guchar*)img_data, len);
    xmpp_stanza_set_attribute(item, "id", sha1);
    xmpp_free(ctx, sha1);

    xmpp_stanza_t* data = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(data, STANZA_NAME_DATA);
    xmpp_stanza_set_ns(data, STANZA_NS_USER_AVATAR_DATA);

    xmpp_stanza_t* text = xmpp_stanza_new(ctx);
    auto_gchar gchar* base64 = g_base64_encode((guchar*)img_data, len);
    xmpp_stanza_set_text(text, base64);

    xmpp_stanza_add_child(data, text);
    xmpp_stanza_add_child(item, data);
    xmpp_stanza_add_child(publish, item);
    xmpp_stanza_add_child(pubsub, publish);
    xmpp_stanza_add_child(iq, pubsub);

    xmpp_stanza_release(text);
    xmpp_stanza_release(data);
    xmpp_stanza_release(item);
    xmpp_stanza_release(publish);
    xmpp_stanza_release(pubsub);

    return iq;
}

xmpp_stanza_t*
stanza_create_avatar_metadata_publish_iq(xmpp_ctx_t* ctx, const char* img_data, gsize len, int height, int width)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_SET, id);
    xmpp_stanza_set_attribute(iq, STANZA_ATTR_FROM, connection_get_fulljid());

    xmpp_stanza_t* pubsub = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(pubsub, STANZA_NAME_PUBSUB);
    xmpp_stanza_set_ns(pubsub, STANZA_NS_PUBSUB);

    xmpp_stanza_t* publish = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(publish, STANZA_NAME_PUBLISH);
    xmpp_stanza_set_attribute(publish, STANZA_ATTR_NODE, STANZA_NS_USER_AVATAR_METADATA);

    xmpp_stanza_t* item = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(item, STANZA_NAME_ITEM);
    char* sha1 = xmpp_sha1(ctx, (guchar*)img_data, len);
    xmpp_stanza_set_attribute(item, "id", sha1);

    xmpp_stanza_t* metadata = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(metadata, STANZA_NAME_METADATA);
    xmpp_stanza_set_ns(metadata, STANZA_NS_USER_AVATAR_METADATA);

    xmpp_stanza_t* info = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(info, STANZA_NAME_INFO);
    xmpp_stanza_set_attribute(info, "id", sha1);
    xmpp_free(ctx, sha1);
    auto_gchar gchar* bytes = g_strdup_printf("%" G_GSIZE_FORMAT, len);
    auto_gchar gchar* h = g_strdup_printf("%d", height);
    auto_gchar gchar* w = g_strdup_printf("%d", width);
    xmpp_stanza_set_attribute(info, "bytes", bytes);
    xmpp_stanza_set_attribute(info, "type", "img/png");
    xmpp_stanza_set_attribute(info, "height", h);
    xmpp_stanza_set_attribute(info, "width", w);

    xmpp_stanza_add_child(metadata, info);
    xmpp_stanza_add_child(item, metadata);
    xmpp_stanza_add_child(publish, item);
    xmpp_stanza_add_child(pubsub, publish);
    xmpp_stanza_add_child(iq, pubsub);

    xmpp_stanza_release(info);
    xmpp_stanza_release(metadata);
    xmpp_stanza_release(item);
    xmpp_stanza_release(publish);
    xmpp_stanza_release(pubsub);

    return iq;
}

xmpp_stanza_t*
stanza_disable_avatar_publish_iq(xmpp_ctx_t* ctx)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_SET, id);
    xmpp_stanza_set_attribute(iq, STANZA_ATTR_FROM, connection_get_fulljid());

    xmpp_stanza_t* pubsub = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(pubsub, STANZA_NAME_PUBSUB);
    xmpp_stanza_set_ns(pubsub, STANZA_NS_PUBSUB);

    xmpp_stanza_t* publish = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(publish, STANZA_NAME_PUBLISH);
    xmpp_stanza_set_attribute(publish, STANZA_ATTR_NODE, STANZA_NS_USER_AVATAR_METADATA);

    xmpp_stanza_t* item = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(item, STANZA_NAME_ITEM);

    xmpp_stanza_t* metadata = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(metadata, STANZA_NAME_METADATA);
    xmpp_stanza_set_ns(metadata, STANZA_NS_USER_AVATAR_METADATA);

    xmpp_stanza_add_child(item, metadata);
    xmpp_stanza_add_child(publish, item);
    xmpp_stanza_add_child(pubsub, publish);
    xmpp_stanza_add_child(iq, pubsub);

    xmpp_stanza_release(metadata);
    xmpp_stanza_release(item);
    xmpp_stanza_release(publish);
    xmpp_stanza_release(pubsub);

    return iq;
}

xmpp_stanza_t*
stanza_create_vcard_request_iq(xmpp_ctx_t* ctx, const char* const jid, const char* const stanza_id)
{
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_GET, stanza_id);
    xmpp_stanza_set_from(iq, connection_get_fulljid());
    if (jid) {
        xmpp_stanza_set_to(iq, jid);
    }

    xmpp_stanza_t* vcard = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(vcard, STANZA_NAME_VCARD);
    xmpp_stanza_set_ns(vcard, STANZA_NS_VCARD);

    xmpp_stanza_add_child(iq, vcard);
    xmpp_stanza_release(vcard);

    return iq;
}

xmpp_stanza_t*
stanza_attach_correction(xmpp_ctx_t* ctx, xmpp_stanza_t* stanza, const char* const replace_id)
{
    xmpp_stanza_t* replace_stanza = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(replace_stanza, "replace");
    xmpp_stanza_set_id(replace_stanza, replace_id);
    xmpp_stanza_set_ns(replace_stanza, STANZA_NS_LAST_MESSAGE_CORRECTION);
    xmpp_stanza_add_child(stanza, replace_stanza);
    xmpp_stanza_release(replace_stanza);

    return stanza;
}

static xmpp_stanza_t*
_text_stanza(xmpp_ctx_t* ctx, const char* name, const char* text)
{
    xmpp_stanza_t* res = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(res, name);

    xmpp_stanza_t* t = xmpp_stanza_new(ctx);
    xmpp_stanza_set_text(t, text);
    xmpp_stanza_add_child_ex(res, t, 0);

    return res;
}

xmpp_stanza_t*
stanza_create_mam_iq(xmpp_ctx_t* ctx, const char* const jid, const char* const startdate, const char* const enddate, const char* const firstid, const char* const lastid)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_SET, id);
    // xmpp_stanza_set_to(iq, jid);

    xmpp_stanza_t* query = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(query, STANZA_NAME_QUERY);
    xmpp_stanza_set_ns(query, STANZA_NS_MAM2);

    xmpp_stanza_t* x = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(x, STANZA_NAME_X);
    xmpp_stanza_set_ns(x, STANZA_NS_DATA);
    xmpp_stanza_set_type(x, "submit");

    // field FORM_TYPE MAM2
    xmpp_stanza_t* field_form_type = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(field_form_type, STANZA_NAME_FIELD);
    xmpp_stanza_set_attribute(field_form_type, STANZA_ATTR_VAR, "FORM_TYPE");
    xmpp_stanza_set_type(field_form_type, "hidden");

    xmpp_stanza_t* value_mam = _text_stanza(ctx, STANZA_NAME_VALUE, STANZA_NS_MAM2);

    xmpp_stanza_add_child_ex(field_form_type, value_mam, 0);

    // field 'with'
    xmpp_stanza_t* field_with = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(field_with, STANZA_NAME_FIELD);
    xmpp_stanza_set_attribute(field_with, STANZA_ATTR_VAR, "with");

    xmpp_stanza_t* value_with = _text_stanza(ctx, STANZA_NAME_VALUE, jid);

    xmpp_stanza_add_child_ex(field_with, value_with, 0);

    // 4.3.2 set/rsm
    xmpp_stanza_t* set = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(set, STANZA_TYPE_SET);
    xmpp_stanza_set_ns(set, STANZA_NS_RSM);

    xmpp_stanza_t* max = _text_stanza(ctx, STANZA_NAME_MAX, PROF_STRINGIFY(MESSAGES_TO_RETRIEVE));
    xmpp_stanza_add_child_ex(set, max, 0);

    if (lastid) {
        xmpp_stanza_t* after = _text_stanza(ctx, STANZA_NAME_AFTER, lastid);
        xmpp_stanza_add_child_ex(set, after, 0);
    }

    if (firstid) {
        xmpp_stanza_t* before = _text_stanza(ctx, STANZA_NAME_BEFORE, firstid);
        xmpp_stanza_add_child_ex(set, before, 0);
    }

    // add and release
    xmpp_stanza_add_child_ex(iq, query, 0);
    xmpp_stanza_add_child_ex(query, x, 0);
    xmpp_stanza_add_child_ex(x, field_form_type, 0);
    xmpp_stanza_add_child_ex(x, field_with, 0);

    // field 'start'
    if (startdate) {
        xmpp_stanza_t* field_start = xmpp_stanza_new(ctx);
        xmpp_stanza_set_name(field_start, STANZA_NAME_FIELD);
        xmpp_stanza_set_attribute(field_start, STANZA_ATTR_VAR, "start");

        xmpp_stanza_t* value_start = _text_stanza(ctx, STANZA_NAME_VALUE, startdate);

        xmpp_stanza_add_child_ex(field_start, value_start, 0);
        xmpp_stanza_add_child_ex(x, field_start, 0);
    }

    if (enddate) {
        xmpp_stanza_t* field_end = xmpp_stanza_new(ctx);
        xmpp_stanza_set_name(field_end, STANZA_NAME_FIELD);
        xmpp_stanza_set_attribute(field_end, STANZA_ATTR_VAR, "end");

        xmpp_stanza_t* value_end = _text_stanza(ctx, STANZA_NAME_VALUE, enddate);

        xmpp_stanza_add_child_ex(field_end, value_end, 0);
        xmpp_stanza_add_child_ex(x, field_end, 0);
    }

    xmpp_stanza_add_child_ex(query, set, 0);

    return iq;
}

xmpp_stanza_t*
stanza_change_password(xmpp_ctx_t* ctx, const char* const user, const char* const password)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_SET, id);

    xmpp_stanza_t* change_password = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(change_password, STANZA_NAME_QUERY);
    xmpp_stanza_set_ns(change_password, STANZA_NS_REGISTER);

    xmpp_stanza_t* username_st = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(username_st, STANZA_NAME_USERNAME);
    xmpp_stanza_t* username_text = xmpp_stanza_new(ctx);
    xmpp_stanza_set_text(username_text, user);
    xmpp_stanza_add_child(username_st, username_text);
    xmpp_stanza_release(username_text);

    xmpp_stanza_t* password_st = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(password_st, STANZA_NAME_PASSWORD);
    xmpp_stanza_t* password_text = xmpp_stanza_new(ctx);
    xmpp_stanza_set_text(password_text, password);
    xmpp_stanza_add_child(password_st, password_text);
    xmpp_stanza_release(password_text);

    xmpp_stanza_add_child(change_password, username_st);
    xmpp_stanza_release(username_st);

    xmpp_stanza_add_child(change_password, password_st);
    xmpp_stanza_release(password_st);

    xmpp_stanza_add_child(iq, change_password);
    xmpp_stanza_release(change_password);

    return iq;
}

xmpp_stanza_t*
stanza_register_new_account(xmpp_ctx_t* ctx, const char* const user, const char* const password)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_SET, id);

    xmpp_stanza_t* register_new_account = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(register_new_account, STANZA_NAME_QUERY);
    xmpp_stanza_set_ns(register_new_account, STANZA_NS_REGISTER);

    xmpp_stanza_t* username_st = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(username_st, STANZA_NAME_USERNAME);
    xmpp_stanza_t* username_text = xmpp_stanza_new(ctx);
    xmpp_stanza_set_text(username_text, user);
    xmpp_stanza_add_child(username_st, username_text);
    xmpp_stanza_release(username_text);

    xmpp_stanza_t* password_st = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(password_st, STANZA_NAME_PASSWORD);
    xmpp_stanza_t* password_text = xmpp_stanza_new(ctx);
    xmpp_stanza_set_text(password_text, password);
    xmpp_stanza_add_child(password_st, password_text);
    xmpp_stanza_release(password_text);

    xmpp_stanza_add_child(register_new_account, username_st);
    xmpp_stanza_release(username_st);

    xmpp_stanza_add_child(register_new_account, password_st);
    xmpp_stanza_release(password_st);

    xmpp_stanza_add_child(iq, register_new_account);
    xmpp_stanza_release(register_new_account);

    return iq;
}

xmpp_stanza_t*
stanza_request_voice(xmpp_ctx_t* ctx, const char* const room)
{
    auto_char char* id = connection_create_stanza_id();
    xmpp_stanza_t* message = xmpp_message_new(ctx, NULL, room, id);

    xmpp_stanza_t* request_voice_st = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(request_voice_st, STANZA_NAME_X);
    xmpp_stanza_set_type(request_voice_st, STANZA_TYPE_SUBMIT);
    xmpp_stanza_set_ns(request_voice_st, STANZA_NS_DATA);

    xmpp_stanza_t* form_type = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(form_type, STANZA_NAME_FIELD);
    xmpp_stanza_set_attribute(form_type, STANZA_ATTR_VAR, "FORM_TYPE");

    xmpp_stanza_t* value_request = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(value_request, STANZA_NAME_VALUE);

    xmpp_stanza_t* request_text = xmpp_stanza_new(ctx);
    xmpp_stanza_set_text(request_text, STANZA_NS_VOICEREQUEST);

    xmpp_stanza_add_child(value_request, request_text);
    xmpp_stanza_release(request_text);

    xmpp_stanza_add_child(form_type, value_request);
    xmpp_stanza_release(value_request);

    xmpp_stanza_add_child(request_voice_st, form_type);
    xmpp_stanza_release(form_type);

    xmpp_stanza_t* request_role = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(request_role, STANZA_NAME_FIELD);
    xmpp_stanza_set_attribute(request_role, STANZA_ATTR_VAR, "muc#role");
    xmpp_stanza_set_attribute(request_role, STANZA_ATTR_TYPE, "list-single");
    xmpp_stanza_set_attribute(request_role, STANZA_ATTR_LABEL, "Requested role");

    xmpp_stanza_t* value_role = xmpp_stanza_new(ctx);
    xmpp_stanza_set_name(value_role, STANZA_NAME_VALUE);

    xmpp_stanza_t* role_text = xmpp_stanza_new(ctx);
    xmpp_stanza_set_text(role_text, "participant");

    xmpp_stanza_add_child(value_role, role_text);
    xmpp_stanza_release(role_text);

    xmpp_stanza_add_child(request_role, value_role);
    xmpp_stanza_release(value_role);

    xmpp_stanza_add_child(request_voice_st, request_role);
    xmpp_stanza_release(request_role);

    xmpp_stanza_add_child(message, request_voice_st);
    xmpp_stanza_release(request_voice_st);

    return message;
}

xmpp_stanza_t*
stanza_create_approve_voice(xmpp_ctx_t* ctx, const char* const id, const char* const jid, const char* const node, DataForm* form)
{
    auto_char char* stid = connection_create_stanza_id();
    xmpp_stanza_t* message = xmpp_message_new(ctx, NULL, jid, stid);

    xmpp_stanza_t* x = form_create_submission(form);

    xmpp_stanza_add_child(message, x);
    xmpp_stanza_release(x);

    return message;
}

xmpp_stanza_t*
stanza_create_muc_register_nick(xmpp_ctx_t* ctx, const char* const id, const char* const jid, const char* const node, DataForm* form)
{
    xmpp_stanza_t* iq = xmpp_iq_new(ctx, STANZA_TYPE_SET, id);

    xmpp_stanza_set_to(iq, jid);

    xmpp_stanza_t* x = form_create_submission(form);

    xmpp_stanza_add_child(iq, x);
    xmpp_stanza_release(x);

    return iq;
}

static void
_contact_addresses_list_free(GSList* list)
{
    if (list) {
        g_slist_free_full(list, g_free);
    }
}

GHashTable*
stanza_get_service_contact_addresses(xmpp_ctx_t* ctx, xmpp_stanza_t* stanza)
{
    GHashTable* addresses = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, (GDestroyNotify)_contact_addresses_list_free);

    xmpp_stanza_t* fields = xmpp_stanza_get_children(stanza);
    while (fields) {
        const char* child_name = xmpp_stanza_get_name(fields);
        const char* child_type = xmpp_stanza_get_type(fields);

        if (g_strcmp0(child_name, STANZA_NAME_FIELD) == 0 && g_strcmp0(child_type, STANZA_TYPE_LIST_MULTI) == 0) {
            // extract key (eg 'admin-addresses')
            const char* var = xmpp_stanza_get_attribute(fields, STANZA_ATTR_VAR);

            // extract values (a list of contact addresses eg mailto:xmpp@shakespeare.lit, xmpp:admins@shakespeare.lit)
            xmpp_stanza_t* values = xmpp_stanza_get_children(fields);
            GSList* val_list = NULL;
            while (values) {
                const char* value_name = xmpp_stanza_get_name(values);
                if (value_name && (g_strcmp0(value_name, STANZA_NAME_VALUE) == 0)) {
                    char* value_text = xmpp_stanza_get_text(values);
                    if (value_text) {
                        val_list = g_slist_append(val_list, g_strdup(value_text));

                        xmpp_free(ctx, value_text);
                    }
                }

                values = xmpp_stanza_get_next(values);
            }

            // add to list
            if (g_slist_length(val_list) > 0) {
                g_hash_table_insert(addresses, g_strdup(var), val_list);
            }
        }

        fields = xmpp_stanza_get_next(fields);
    }

    return addresses;
}
