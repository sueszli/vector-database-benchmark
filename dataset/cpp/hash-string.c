/*
 *   Copyright (C) 2007-2016 Tristan Heaven <tristan@tristanheaven.net>
 *
 *   This file is part of GtkHash.
 *
 *   GtkHash is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 2 of the License, or
 *   (at your option) any later version.
 *
 *   GtkHash is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with GtkHash. If not, see <https://gnu.org/licenses/gpl-2.0.txt>.
 */

#ifdef HAVE_CONFIG_H
	#include "config.h"
#endif

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <glib.h>

#include "hash-string.h"
#include "hash-func.h"
#include "hash-lib.h"

void gtkhash_hash_string(struct hash_func_s *funcs, const char * const str,
	const enum digest_format_e format, const uint8_t * const hmac_key,
	const size_t key_size)
{
	g_assert(str);
	g_assert(DIGEST_FORMAT_IS_VALID(format));

	const size_t len = strlen(str);

	for (int i = 0; i < HASH_FUNCS_N; i++) {
		if (!funcs[i].enabled)
			continue;

		gtkhash_hash_lib_start(&funcs[i], hmac_key, key_size);

		// Assuming this won't take too long
		gtkhash_hash_lib_update(&funcs[i], (const uint8_t *)str, len);

		gtkhash_hash_lib_finish(&funcs[i]);

		char *digest = gtkhash_hash_func_get_digest(&funcs[i], format);
		gtkhash_hash_string_digest_cb(funcs[i].id, digest);
		g_free(digest);

		gtkhash_hash_func_clear_digest(&funcs[i]);
	}

	gtkhash_hash_string_finish_cb();
}
