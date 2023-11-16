/*
 * SPDX-License-Identifier: ISC
 * SPDX-URL: https://spdx.org/licenses/ISC.html
 *
 * Copyright (C) 2005-2007 William Pitcock, et al.
 *
 * This file contains protocol support for P10 ircd's.
 * Some sources used: Run's documentation, beware's description,
 * raw data sent by asuka.
 */

#include <atheme.h>
#include <atheme/protocol/asuka.h>

static struct ircd Asuka = {
	.ircdname = "Asuka 1.2.1 and later",
	.tldprefix = "$",
	.uses_uid = true,
	.uses_rcommand = false,
	.uses_owner = false,
	.uses_protect = false,
	.uses_halfops = false,
	.uses_p10 = true,
	.uses_vhost = true,
	.oper_only_modes = 0,
	.owner_mode = 0,
	.protect_mode = 0,
	.halfops_mode = 0,
	.owner_mchar = "+",
	.protect_mchar = "+",
	.halfops_mchar = "+",
	.type = PROTOCOL_ASUKA,
	.perm_mode = 0,
	.oimmune_mode = 0,
	.ban_like_modes = "b",
	.except_mchar = 0,
	.invex_mchar = 0,
	.flags = IRCD_CIDR_BANS,
};

static const struct cmode asuka_mode_list[] = {
  { 'i', CMODE_INVITE },
  { 'm', CMODE_MOD    },
  { 'n', CMODE_NOEXT  },
  { 'p', CMODE_PRIV   },
  { 's', CMODE_SEC    },
  { 't', CMODE_TOPIC  },
  { 'c', CMODE_NOCOLOR },
  { 'C', CMODE_NOCTCP },
  { 'D', CMODE_DELAYED },
  { 'u', CMODE_NOQUIT },
  { 'N', CMODE_NONOTICE },
  { '\0', 0 }
};

static struct extmode asuka_ignore_mode_list[] = {
  { '\0', 0 }
};

static const struct cmode asuka_status_mode_list[] = {
  { 'o', CSTATUS_OP    },
  { 'v', CSTATUS_VOICE },
  { '\0', 0 }
};

static const struct cmode asuka_prefix_mode_list[] = {
  { '@', CSTATUS_OP    },
  { '+', CSTATUS_VOICE },
  { '\0', 0 }
};

static const struct cmode asuka_user_mode_list[] = {
  { 'i', UF_INVIS    },
  { 'o', UF_IRCOP    },
  { 'd', UF_DEAF     },
  { 'k', UF_IMMUNE   },
  { '\0', 0 }
};

static void check_hidehost(struct user *u);

// NOTICE wrapper
static void
asuka_notice_channel_sts(struct user *from, struct channel *target, const char *text)
{
	if (target->modes & CMODE_NONOTICE)
	{
		/* asuka sucks */
		/* remove that stupid +N mode before it blocks our notice
		 * -- jilles */
		sts("%s M %s -N", from ? from->uid : me.numeric, target->name);
		target->modes &= ~CMODE_NONOTICE;
	}
	if (from == NULL || chanuser_find(target, from))
		sts("%s O %s :%s", from ? from->uid : me.numeric, target->name, text);
	else
		sts("%s O %s :[%s:%s] %s", me.numeric, target->name, from->nick, target->name, text);
}

static void
asuka_wallchops(struct user *sender, struct channel *channel, const char *message)
{
	if (channel->modes & CMODE_NONOTICE)
	{
		/* asuka sucks */
		/* remove that stupid +N mode before it blocks our notice
		 * -- jilles */
		sts("%s M %s -N", sender->uid, channel->name);
		channel->modes &= ~CMODE_NONOTICE;
	}
	sts("%s WC %s :%s", sender->uid, channel->name, message);
}

// protocol-specific stuff to do on login
static void
asuka_on_login(struct user *u, struct myuser *account, const char *wantedhost)
{
	return_if_fail(u != NULL);

	sts("%s AC %s %s %lu", me.numeric, u->uid, entity(u->myuser)->name,
			(unsigned long)account->registered);
	check_hidehost(u);
}

/* P10 does not support logout, so kill the user
 * we can't keep track of which logins are stale and which aren't -- jilles */
static bool
asuka_on_logout(struct user *u, const char *account)
{
	return_val_if_fail(u != NULL, false);

	kill_user(nicksvs.me ? nicksvs.me->me : NULL, u, "Forcing logout %s -> %s", u->nick, account);
	return true;
}

static void
m_nick(struct sourceinfo *si, int parc, char *parv[])
{
	struct user *u;
	char ipstring[HOSTIPLEN + 1];
	char *p;
	int i;

	// got the right number of args for an introduction?
	if (parc >= 8)
	{
		/* -> AB N jilles 1 1137687480 jilles jaguar.test +oiwgrx jilles B]AAAB ABAAE :Jilles Tjoelker */
		/* -> AB N test4 1 1137690148 jilles jaguar.test +iw B]AAAB ABAAG :Jilles Tjoelker */
		slog(LG_DEBUG, "m_nick(): new user on `%s': %s", si->s->name, parv[0]);

		decode_p10_ip(parv[parc - 3], ipstring);
		u = user_add(parv[0], parv[3], parv[4], NULL, ipstring, parv[parc - 2], parv[parc - 1], si->s, atoi(parv[2]));
		if (u == NULL)
			return;

		if (parv[5][0] == '+')
		{
			user_mode(u, parv[5]);
			i = 1;
			if (strchr(parv[5], 'r'))
			{
				p = strchr(parv[5+i], ':');
				if (p != NULL)
					*p++ = '\0';
				handle_burstlogin(u, parv[5+i], p ? atol(p) : 0);

				// killed to force logout?
				if (user_find(parv[parc - 2]) == NULL)
					return;

				i++;
			}
			if (strchr(parv[5], 'h'))
			{
				p = strchr(parv[5+i], '@');
				if (p == NULL)
				{
					strshare_unref(u->vhost);
					u->vhost = strshare_get(parv[5 + i]);
				}
				else
				{
					char userbuf[USERLEN + 1];

					strshare_unref(u->vhost);
					u->vhost = strshare_get(p + 1);

					mowgli_strlcpy(userbuf, parv[5+i], sizeof userbuf);
					p = strchr(userbuf, '@');
					if (p != NULL)
						*p = '\0';

					strshare_unref(u->user);
					u->user = strshare_get(userbuf);
				}
				i++;
			}
			if (strchr(parv[5], 'x'))
			{
				u->flags |= UF_HIDEHOSTREQ;

				// this must be after setting the account name
				check_hidehost(u);
			}
		}

		handle_nickchange(u);
	}
	// if it's only 2 then it's a nickname change
	else if (parc == 2)
	{
		if (!si->su)
		{
			slog(LG_DEBUG, "m_nick(): server trying to change nick: %s", si->s != NULL ? si->s->name : "<none>");
			return;
		}

		slog(LG_DEBUG, "m_nick(): nickname change from `%s': %s", si->su->nick, parv[0]);

		if (user_changenick(si->su, parv[0], atoi(parv[1])))
			return;

		handle_nickchange(si->su);
	}
	else
	{
		slog(LG_DEBUG, "m_nick(): got NICK with wrong (%d) number of params", parc);

		for (i = 0; i < parc; i++)
			slog(LG_DEBUG, "m_nick():   parv[%d] = %s", i, parv[i]);
	}
}

static void
m_mode(struct sourceinfo *si, int parc, char *parv[])
{
	struct user *u;
	char *p;

	if (*parv[0] == '#')
		channel_mode(NULL, channel_find(parv[0]), parc - 1, &parv[1]);
	else
	{
		// Yes this is a nick and not a UID -- jilles
		u = user_find_named(parv[0]);
		if (u == NULL)
		{
			slog(LG_DEBUG, "m_mode(): user mode for unknown user %s", parv[0]);
			return;
		}
		user_mode(u, parv[1]);
		if (strchr(parv[1], 'x'))
		{
			u->flags |= UF_HIDEHOSTREQ;
			check_hidehost(u);
		}
		if (strchr(parv[1], 'h'))
		{
			if (parc > 2)
			{
				// assume +h
				p = strchr(parv[2], '@');
				if (p == NULL)
				{
					strshare_unref(u->vhost);
					u->vhost = strshare_get(parv[2]);
				}
				else
				{
					char userbuf[USERLEN + 1];

					strshare_unref(u->vhost);
					u->vhost = strshare_get(p + 1);

					mowgli_strlcpy(userbuf, parv[2], sizeof userbuf);

					p = strchr(userbuf, '@');
					if (p != NULL)
						*p = '\0';

					strshare_unref(u->user);
					u->user = strshare_get(userbuf);
				}
				slog(LG_DEBUG, "m_mode(): user %s setting vhost %s@%s", u->nick, u->user, u->vhost);
			}
			else
			{
				// must be -h
				// XXX we don't know the original ident
				slog(LG_DEBUG, "m_mode(): user %s turning off vhost", u->nick);

				strshare_unref(u->vhost);
				u->vhost = strshare_get(u->host);

				// revert to +x vhost if applicable
				check_hidehost(u);
			}
		}
	}
}

static void
check_hidehost(struct user *u)
{
	static bool warned = false;
	char buf[HOSTLEN + 1];

	// do they qualify?
	if (!(u->flags & UF_HIDEHOSTREQ) || u->myuser == NULL || (u->myuser->flags & MU_WAITAUTH))
		return;
	// don't use this if they have some other kind of vhost
	if (strcmp(u->host, u->vhost))
	{
		slog(LG_DEBUG, "check_hidehost(): +x overruled by other vhost for %s", u->nick);
		return;
	}
	if (me.hidehostsuffix == NULL)
	{
		if (!warned)
		{
			wallops("Misconfiguration: serverinfo::hidehostsuffix not set");
			warned = true;
		}
		return;
	}

	snprintf(buf, sizeof buf, "%s.%s", entity(u->myuser)->name, me.hidehostsuffix);

	strshare_unref(u->vhost);
	u->vhost = strshare_get(buf);

	slog(LG_DEBUG, "check_hidehost(): %s -> %s", u->nick, u->vhost);
}

static void
mod_init(struct module *const restrict m)
{
	MODULE_TRY_REQUEST_DEPENDENCY(m, "protocol/p10-generic")

	// Symbol relocation voodoo.
	notice_channel_sts = &asuka_notice_channel_sts;
	wallchops = &asuka_wallchops;
	ircd_on_login = &asuka_on_login;
	ircd_on_logout = &asuka_on_logout;

	mode_list = asuka_mode_list;
	ignore_mode_list = asuka_ignore_mode_list;
	status_mode_list = asuka_status_mode_list;
	prefix_mode_list = asuka_prefix_mode_list;
	user_mode_list = asuka_user_mode_list;
	ignore_mode_list_size = ARRAY_SIZE(asuka_ignore_mode_list);

	ircd = &Asuka;

	// override these
	pcommand_delete("N");
	pcommand_delete("M");
	pcommand_delete("OM");
	pcommand_add("N", m_nick, 2, MSRC_USER | MSRC_SERVER);
	pcommand_add("M", m_mode, 2, MSRC_USER | MSRC_SERVER);
	pcommand_add("OM", m_mode, 2, MSRC_USER); // OPMODE, treat as MODE
}

static void
mod_deinit(const enum module_unload_intent ATHEME_VATTR_UNUSED intent)
{

}

SIMPLE_DECLARE_MODULE_V1("protocol/asuka", MODULE_UNLOAD_CAPABILITY_NEVER)
