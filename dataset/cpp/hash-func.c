/*
 *   Copyright (C) 2007-2020 Tristan Heaven <tristan@tristanheaven.net>
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
#include <glib.h>

#include "hash-func.h"
#include "hash-lib.h"
#include "digest.h"

static const struct {
	const char * const name;
	const uint8_t digest_size;
	const uint8_t block_size; // for HMAC
} hash_func_data[HASH_FUNCS_N] = {
	[HASH_FUNC_ADLER32]    = { "ADLER32",       4,   0, },
	[HASH_FUNC_BLAKE2B]    = { "BLAKE2b",      64, 128, },
	[HASH_FUNC_BLAKE2BP]   = { "BLAKE2bp",     64,   0, },
	[HASH_FUNC_BLAKE2S]    = { "BLAKE2s",      32,  64, },
	[HASH_FUNC_BLAKE2SP]   = { "BLAKE2sp",     32,   0, },
	[HASH_FUNC_CRC32]      = { "CRC32",         4,   0, },
	[HASH_FUNC_CRC32C]     = { "CRC32C",        4,   0, },
	[HASH_FUNC_GOST]       = { "GOST",         32,   0, },
	[HASH_FUNC_MD2]        = { "MD2",          16,   0, },
	[HASH_FUNC_MD4]        = { "MD4",          16,   0, },
	[HASH_FUNC_MD5]        = { "MD5",          16,  64, },
	[HASH_FUNC_MD6_224]    = { "MD6-224",      28,   0, },
	[HASH_FUNC_MD6_256]    = { "MD6-256",      32,   0, },
	[HASH_FUNC_MD6_384]    = { "MD6-384",      48,   0, },
	[HASH_FUNC_MD6_512]    = { "MD6-512",      64,   0, },
	[HASH_FUNC_MDC2]       = { "MDC2",         16,   0, },
	[HASH_FUNC_RIPEMD128]  = { "RIPEMD128",    16,  64, },
	[HASH_FUNC_RIPEMD160]  = { "RIPEMD160",    20,  64, },
	[HASH_FUNC_RIPEMD256]  = { "RIPEMD256",    32,   0, },
	[HASH_FUNC_RIPEMD320]  = { "RIPEMD320",    40,   0, },
	[HASH_FUNC_SHA1]       = { "SHA1",         20,  64, },
	[HASH_FUNC_SHA224]     = { "SHA224",       28,  64, },
	[HASH_FUNC_SHA256]     = { "SHA256",       32,  64, },
	[HASH_FUNC_SHA384]     = { "SHA384",       48, 128, },
	[HASH_FUNC_SHA512]     = { "SHA512",       64, 128, },
	[HASH_FUNC_SHA3_224]   = { "SHA3-224",     28, 144, },
	[HASH_FUNC_SHA3_256]   = { "SHA3-256",     32, 136, },
	[HASH_FUNC_SHA3_384]   = { "SHA3-384",     48, 104, },
	[HASH_FUNC_SHA3_512]   = { "SHA3-512",     64,  72, },
	[HASH_FUNC_SM3]        = { "SM3",          32,   0, },
	[HASH_FUNC_TIGER192]   = { "TIGER192",     24,   0, },
	[HASH_FUNC_WHIRLPOOL]  = { "WHIRLPOOL",    64,   0, },
	[HASH_FUNC_XXH64]      = { "XXH64",         8,   0, },
};

enum hash_func_e gtkhash_hash_func_get_id_from_name(const char *name)
{
	g_assert(name);

	for (int i = 0; i < HASH_FUNCS_N; i++)
		if (g_ascii_strcasecmp(name, hash_func_data[i].name) == 0)
			return i;

	return HASH_FUNC_INVALID;
}

void gtkhash_hash_func_set_digest(struct hash_func_s *func, uint8_t *digest,
	size_t size)
{
	g_assert(func);
	g_assert(digest);
	g_assert(size == func->digest_size);

	gtkhash_digest_set_data(func->digest, digest, size);
}

char *gtkhash_hash_func_get_digest(struct hash_func_s *func,
	const enum digest_format_e format)
{
	g_assert(func);
	g_assert(DIGEST_FORMAT_IS_VALID(format));

	char *digest = gtkhash_digest_get_data(func->digest, format);
	g_assert(digest);

	return digest;
}

void gtkhash_hash_func_clear_digest(struct hash_func_s *func)
{
	g_assert(func);

	gtkhash_digest_free_data(func->digest);
}

void gtkhash_hash_func_init(struct hash_func_s *func,
	const enum hash_func_e id)
{
	g_assert(func);
	g_assert(HASH_FUNC_IS_VALID(id));

	func->id = id;
	func->supported = gtkhash_hash_lib_is_supported(id);
	func->enabled = false;
	func->name = hash_func_data[id].name;
	func->digest = gtkhash_digest_new();
	func->lib_data = NULL;
	func->hmac_data = NULL;
	func->digest_size = hash_func_data[id].digest_size;
	func->block_size = hash_func_data[id].block_size;
	func->hmac_supported = (hash_func_data[id].block_size > 0);
}

void gtkhash_hash_func_init_all(struct hash_func_s *funcs)
{
	g_assert(funcs);

	for (int i = 0; i < HASH_FUNCS_N; i++)
		gtkhash_hash_func_init(&funcs[i], i);
}

void gtkhash_hash_func_deinit(struct hash_func_s *func)
{
	g_assert(func);

	gtkhash_digest_free(func->digest);
	func->digest = NULL;
}

void gtkhash_hash_func_deinit_all(struct hash_func_s *funcs)
{
	g_assert(funcs);

	for (int i = 0; i < HASH_FUNCS_N; i++)
		gtkhash_hash_func_deinit(&funcs[i]);
}
