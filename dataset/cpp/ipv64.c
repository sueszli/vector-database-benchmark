/* Plugin for ipv64.net
 *
 * Copyright (C) 2023 Dennis Schröder <info@ipv64.net>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, visit the Free Software Foundation
 * website at http://www.gnu.org/licenses/gpl-2.0.html or write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 * Boston, MA  02110-1301, USA.
 */

#include "plugin.h"

#define IPV64_UPDATE_IP_REQUEST					\
	"GET %s?"						\
	"token=%s&"						\
	"ip=%s "						\
	"HTTP/1.0\r\n"					        \
	"Host: %s\r\n"					        \
	"Authorization: Basic %s\r\n"				\
	"User-Agent: %s\r\n\r\n"

#define IPV64_UPDATE_IP6_REQUEST				\
	"GET %s?"						\
	"token=%s&"		  		  	 	\
	"ip6=%s "					        \
	"HTTP/1.0\r\n"					        \
	"Host: %s\r\n"					        \
	"Authorization: Basic %s\r\n"				\
	"User-Agent: %s\r\n\r\n"


static int request  (ddns_t       *ctx,   ddns_info_t *info, ddns_alias_t *alias);
static int response (http_trans_t *trans, ddns_info_t *info, ddns_alias_t *alias);

static ddns_system_t plugin = {
	.name         = "ipv4@ipv64.net",
	.alias        = "default@ipv64.net",

	.request      = (req_fn_t)request,
	.response     = (rsp_fn_t)response,

	.checkip_name = "ifconfig.me",
	.checkip_url  = "/ip",
	.checkip_ssl  = DYNDNS_MY_IP_SSL,

	.server_name  = "ipv4.ipv64.net",
	.server_url   =  "/nic/update"
};

static ddns_system_t plugin_v6 = {
	.name         = "ipv6@ipv64.net",

	.request      = (req_fn_t)request,
	.response     = (rsp_fn_t)response,

	.checkip_name = "ifconfig.me",
	.checkip_url  = "/ip",
	.checkip_ssl  = DYNDNS_MY_IP_SSL,

	.server_name  = "ipv6.ipv64.net",
	.server_url   =  "/nic/update"
};

static int request(ddns_t *ctx, ddns_info_t *info, ddns_alias_t *alias)
{
	if (strstr(info->system->name, "ipv6"))
		return snprintf(ctx->request_buf, ctx->request_buflen,
			info->system->server_req,
			info->server_url,
			alias->name,
			alias->address,
			info->server_name.name,
			info->creds.encoded_password,
			info->user_agent);

	return snprintf(ctx->request_buf, ctx->request_buflen,
			info->system->server_req,
			info->server_url,
			alias->name,
			alias->address,
			info->server_name.name,
			info->creds.encoded_password,
			info->user_agent);
}

static int response(http_trans_t *trans, ddns_info_t *info, ddns_alias_t *alias)
{
	(void)info;
	(void)alias;

	DO(http_status_valid(trans->status));

	return 0;
}

PLUGIN_INIT(plugin_init)
{
	plugin_register(&plugin, IPV64_UPDATE_IP_REQUEST);
	plugin_register(&plugin_v6, IPV64_UPDATE_IP6_REQUEST);
}

PLUGIN_EXIT(plugin_exit)
{
	plugin_unregister(&plugin);
	plugin_unregister(&plugin_v6);
}

/**
 * Local Variables:
 *  indent-tabs-mode: t
 *  c-file-style: "linux"
 * End:
 */
