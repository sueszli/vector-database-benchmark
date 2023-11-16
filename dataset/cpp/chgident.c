/*
 *   IRC - Internet Relay Chat, src/modules/chgident.c
 *   (C) 1999-2001 Carsten Munk (Techie/Stskeeps) <stskeeps@tspre.org>
 *
 *   See file AUTHORS in IRC package for additional names of
 *   the programmers. 
 *
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 1, or (at your option)
 *   any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program; if not, write to the Free Software
 *   Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

#include "unrealircd.h"

#define MSG_CHGIDENT 	"CHGIDENT"

CMD_FUNC(cmd_chgident);

ModuleHeader MOD_HEADER
  = {
	"chgident",	/* Name of module */
	"5.0", /* Version */
	"/chgident", /* Short description of module */
	"UnrealIRCd Team",
	"unrealircd-6",
    };

MOD_INIT()
{
	CommandAdd(modinfo->handle, MSG_CHGIDENT, cmd_chgident, MAXPARA, CMD_USER|CMD_SERVER);
	MARK_AS_OFFICIAL_MODULE(modinfo);
	return MOD_SUCCESS;
}

MOD_LOAD()
{
	return MOD_SUCCESS;
}

MOD_UNLOAD()
{
	return MOD_SUCCESS;
}

/* 
 * cmd_chgident - 12/07/1999 (two months after I made SETIDENT) - Stskeeps
 * :prefix CHGIDENT <nick> <new identname>
 * parv[1] - nickname
 * parv[2] - identname
 *
*/

CMD_FUNC(cmd_chgident)
{
	Client *target;
	const char *s;
	int legalident = 1;

	if (!ValidatePermissionsForPath("client:set:ident",client,NULL,NULL,NULL))
	{
		sendnumeric(client, ERR_NOPRIVILEGES);
		return;
	}


	if ((parc < 3) || !*parv[2])
	{
		sendnumeric(client, ERR_NEEDMOREPARAMS, "CHGIDENT");
		return;
	}

	if (strlen(parv[2]) > (USERLEN))
	{
		sendnotice(client, "*** ChgIdent Error: Requested ident too long -- rejected.");
		return;
	}

	if (!valid_username(parv[2]))
	{
		sendnotice(client, "*** /ChgIdent Error: A ident may contain a-z, A-Z, 0-9, '-' & '.' - Please only use them");
		return;
	}

	target = find_client(parv[1], NULL);
	if (!MyUser(client) && !target && (target = find_server_by_uid(parv[1])))
	{
		/* CHGIDENT for a UID that is not online.
		 * Let's assume it may not YET be online and forward the message to
		 * the remote server and stop processing ourselves.
		 * That server will then handle pre-registered processing of the
		 * CHGIDENT and later communicate the host when the user actually
		 * comes online in the UID message.
		 */
		sendto_one(target, recv_mtags, ":%s CHGIDENT %s %s", client->id, parv[1], parv[2]);
		return;
	}

	if (!target || !target->user)
	{
		sendnumeric(client, ERR_NOSUCHNICK, parv[1]);
		return;
	}
	userhost_save_current(target);

	switch (UHOST_ALLOWED)
	{
		case UHALLOW_NEVER:
			if (MyUser(client))
			{
				sendnumeric(client, ERR_DISABLED, "CHGIDENT",
					"This command is disabled on this server");
				return;
			}
			break;
		case UHALLOW_ALWAYS:
			break;
		case UHALLOW_NOCHANS:
			if (IsUser(target) && MyUser(client) && target->user->joined)
			{
				sendnotice(client, "*** /ChgIdent can not be used while %s is on a channel", target->name);
				return;
			}
			break;
		case UHALLOW_REJOIN:
			/* join sent later when the ident has been changed */
			break;
	}
	if (!IsULine(client))
	{
		const char *issuer = command_issued_by_rpc(recv_mtags);
		if (issuer)
		{
			unreal_log(ULOG_INFO, "chgcmds", "CHGIDENT_COMMAND", client,
				   "CHGIDENT: $issuer changed the username of $target.details to be $new_username",
				   log_data_string("issuer", issuer),
				   log_data_string("change_type", "username"),
				   log_data_client("target", target),
				   log_data_string("new_username", parv[2]));
		} else {
			unreal_log(ULOG_INFO, "chgcmds", "CHGIDENT_COMMAND", client,
				   "CHGIDENT: $client changed the username of $target.details to be $new_username",
				   log_data_string("change_type", "username"),
				   log_data_client("target", target),
				   log_data_string("new_username", parv[2]));
		}
	}

	/* Send to other servers too, unless the client is still in the registration phase (SASL) */
	if (IsUser(target))
		sendto_server(client, 0, 0, recv_mtags, ":%s CHGIDENT %s %s", client->id, target->id, parv[2]);

	ircsnprintf(target->user->username, sizeof(target->user->username), "%s", parv[2]);
	userhost_changed(target);
}
