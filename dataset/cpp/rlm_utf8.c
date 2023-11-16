/*
 *   This program is is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or (at
 *   your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program; if not, write to the Free Software
 *   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA
 */

/**
 * $Id: 9444bcc944d11c20ac0f2eafc440b27f6553aaa1 $
 * @file rlm_utf8.c
 * @brief Enforce UTF8 encoding in strings.
 *
 * @copyright 2000,2006 The FreeRADIUS server project
 */
RCSID("$Id: 9444bcc944d11c20ac0f2eafc440b27f6553aaa1 $")

#include <freeradius-devel/server/base.h>
#include <freeradius-devel/server/module_rlm.h>

/*
 *	Reject any non-UTF8 data.
 */
static unlang_action_t CC_HINT(nonnull) mod_utf8_clean(rlm_rcode_t *p_result, UNUSED module_ctx_t const *mctx, request_t *request)
{
	size_t		i, len;

	fr_pair_list_foreach(&request->request_pairs, vp) {
		if (vp->vp_type != FR_TYPE_STRING) continue;

		for (i = 0; i < vp->vp_length; i += len) {
			len = fr_utf8_char(&vp->vp_octets[i], -1);
			if (len == 0) RETURN_MODULE_FAIL;
		}
	}

	RETURN_MODULE_NOOP;
}

/*
 *	The module name should be the only globally exported symbol.
 *	That is, everything else should be 'static'.
 *
 *	If the module needs to temporarily modify it's instantiation
 *	data, the type should be changed to MODULE_TYPE_THREAD_UNSAFE.
 *	The server will then take care of ensuring that the module
 *	is single-threaded.
 */
extern module_rlm_t rlm_utf8;
module_rlm_t rlm_utf8 = {
	.common = {
		.magic		= MODULE_MAGIC_INIT,
		.name		= "utf8",
		.type		= MODULE_TYPE_THREAD_SAFE
	},
	.method_names = (module_method_name_t[]){
		{ .name1 = CF_IDENT_ANY,	.name2 = CF_IDENT_ANY,		.method = mod_utf8_clean },
		MODULE_NAME_TERMINATOR
	}
};
