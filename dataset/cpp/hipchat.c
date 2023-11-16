/***************************************************************************\
*                                                                           *
*  BitlBee - An IRC to IM gateway                                           *
*  Jabber module - HipChat specific functions                               *
*                                                                           *
*  Copyright 2015 Xamarin Inc                                               *
*                                                                           *
*  This program is free software; you can redistribute it and/or modify     *
*  it under the terms of the GNU General Public License as published by     *
*  the Free Software Foundation; either version 2 of the License, or        *
*  (at your option) any later version.                                      *
*                                                                           *
*  This program is distributed in the hope that it will be useful,          *
*  but WITHOUT ANY WARRANTY; without even the implied warranty of           *
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            *
*  GNU General Public License for more details.                             *
*                                                                           *
*  You should have received a copy of the GNU General Public License along  *
*  with this program; if not, write to the Free Software Foundation, Inc.,  *
*  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.              *
*                                                                           *
\***************************************************************************/

#include "jabber.h"

xt_status hipchat_handle_success(struct im_connection *ic, struct xt_node *node)
{
	struct jabber_data *jd = ic->proto_data;
	char *sep, *jid;

	jid = xt_find_attr(node, "jid");

	sep = strchr(jid, '/');
	if (sep) {
		*sep = '\0';
	}

	jabber_set_me(ic, jid);
	imcb_log(ic, "Setting Hipchat JID to %s", jid);

	if (sep) {
		*sep = '/';
	}

	jd->muc_host = g_strdup(xt_find_attr(node, "muc_host"));

	/* Hipchat's auth doesn't expect a restart here */
	jd->flags &= ~JFLAG_STREAM_RESTART;

	if (!jabber_get_roster(ic) ||
	    !jabber_iq_disco_server(ic) ||
	    !jabber_get_hipchat_profile(ic)) {
		return XT_ABORT;
	}

	return XT_HANDLED;
}

int jabber_get_hipchat_profile(struct im_connection *ic)
{
	struct jabber_data *jd = ic->proto_data;
	struct xt_node *node;
	int st;

	imcb_log(ic, "Fetching hipchat profile for %s", jd->me);

	node = xt_new_node("query", NULL, NULL);
	xt_add_attr(node, "xmlns", XMLNS_HIPCHAT_PROFILE);
	node = jabber_make_packet("iq", "get", jd->me, node);

	jabber_cache_add(ic, node, jabber_parse_hipchat_profile);
	st = jabber_write_packet(ic, node);

	return st;
}

xt_status jabber_parse_hipchat_profile(struct im_connection *ic, struct xt_node *node, struct xt_node *orig)
{
	struct xt_node *query, *name_node;

	if (!(query = xt_find_node(node->children, "query"))) {
		imcb_log(ic, "Warning: Received NULL profile packet");
		return XT_ABORT;
	}

	name_node = xt_find_node(query->children, "name");
	if (!name_node) {
		imcb_log(ic, "Warning: Can't find real name in profile. Joining groupchats will not be possible.");
		return XT_ABORT;
	}

	set_setstr(&ic->acc->set, "display_name", name_node->text);
	return XT_HANDLED;

}

/* Returns a newly allocated string that tries to match the "slug" part of the JID using an
 * approximation of the method used by the server. This might fail in some rare conditions
 * (old JIDs generated a different way, locale settings unicode, etc) */
char *hipchat_make_channel_slug(const char *name)
{
	char *lower;
	char *new = g_malloc(strlen(name) + 1);
	int i = 0;

	do {
		if (*name == ' ') {
			new[i++] = '_';
		} else if (*name && !strchr("\"&'/:<>@", *name)) {
			new[i++] = *name;
		}
	} while (*(name++));

	new[i] = '\0';

	lower = g_utf8_strdown(new, -1);
	g_free(new);

	return lower;
}

char *hipchat_guess_channel_name(struct im_connection *ic, const char *name)
{
	struct jabber_data *jd = ic->proto_data;
	char *slug, *retval, *underscore;
	
	if (!(underscore = strchr(jd->username, '_')) || !jd->muc_host) {
		return NULL;
	}

	slug = hipchat_make_channel_slug(name);

	/* Get the organization ID from the username, before the underscore */
	*underscore = '\0';

	retval = g_strdup_printf("%s_%s@%s", jd->username, slug, jd->muc_host);

	*underscore = '_';

	g_free(slug);

	return retval;
}
