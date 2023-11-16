/*
 *   Copyright (C) 2007-2021 Tristan Heaven <tristan@tristanheaven.net>
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
#include <unistd.h>
#include <errno.h>
#include <sys/socket.h>
#include <linux/if_alg.h>
#include <glib.h>

#include "hash-lib.h"
#include "hash-func.h"

HASH_LIB_DECL(linux)

#define LIB_DATA ((struct hash_lib_linux_s *)func->lib_data)

struct hash_lib_linux_s {
	const char *name;
	int sockfd, connfd;
};

static const char *gtkhash_hash_lib_linux_get_name(const enum hash_func_e id)
{
	switch (id) {
		case HASH_FUNC_BLAKE2B:   return "blake2b-512";
		case HASH_FUNC_BLAKE2S:   return "blake2s-256";
		case HASH_FUNC_CRC32C:    return "crc32c";
		case HASH_FUNC_MD4:       return "md4";
		case HASH_FUNC_MD5:       return "md5";
		case HASH_FUNC_RIPEMD128: return "rmd128";
		case HASH_FUNC_RIPEMD160: return "rmd160";
		case HASH_FUNC_RIPEMD256: return "rmd256";
		case HASH_FUNC_RIPEMD320: return "rmd320";
		case HASH_FUNC_SHA1:      return "sha1";
		case HASH_FUNC_SHA224:    return "sha224";
		case HASH_FUNC_SHA256:    return "sha256";
		case HASH_FUNC_SHA384:    return "sha384";
		case HASH_FUNC_SHA512:    return "sha512";
		case HASH_FUNC_SHA3_224:  return "sha3-224";
		case HASH_FUNC_SHA3_256:  return "sha3-256";
		case HASH_FUNC_SHA3_384:  return "sha3-384";
		case HASH_FUNC_SHA3_512:  return "sha3-512";
		case HASH_FUNC_TIGER192:  return "tgr192";
		case HASH_FUNC_WHIRLPOOL: return "wp512";
		case HASH_FUNC_XXH64:     return "xxhash64";

		default:
			return NULL;
	}
}

bool gtkhash_hash_lib_linux_is_supported(const enum hash_func_e id)
{
	struct hash_lib_linux_s data;

	if (!(data.name = gtkhash_hash_lib_linux_get_name(id)))
		return false;

	if ((data.sockfd = socket(AF_ALG, SOCK_SEQPACKET, 0)) == -1) {
		g_debug("Kernel AF_ALG: %s", g_strerror(errno));
		return false;
	}

	struct sockaddr_alg addr = {
		.salg_family = AF_ALG,
		.salg_type = "hash",
	};

	strcpy((char *)addr.salg_name, data.name);

	if (bind(data.sockfd, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
		g_debug("Kernel AF_ALG '%s': %s", data.name, g_strerror(errno));
		close(data.sockfd);
		return false;
	}

	if ((data.connfd = accept(data.sockfd, NULL, NULL)) == -1) {
		g_debug("Kernel AF_ALG '%s': %s", data.name, g_strerror(errno));
		close(data.sockfd);
		return false;
	}

	// Update
	ssize_t bytes = 0;
	if ((bytes = send(data.connfd, "1234567", 8, MSG_MORE)) != 8) {
		if (bytes < 0)
			g_debug("Kernel AF_ALG '%s': %s", data.name, g_strerror(errno));
		close(data.connfd);
		close(data.sockfd);
		return false;
	}

	// Finish
	uint8_t digest[4];
	if ((bytes = read(data.connfd, &digest, 4)) != 4) {
		if (bytes < 0)
			g_debug("Kernel AF_ALG '%s': %s", data.name, g_strerror(errno));
		close(data.connfd);
		close(data.sockfd);
		return false;
	}

	close(data.connfd);
	close(data.sockfd);
	return true;
}

void gtkhash_hash_lib_linux_start(struct hash_func_s *func)
{
	func->lib_data = g_new(struct hash_lib_linux_s, 1);

	struct sockaddr_alg addr = {
		.salg_family = AF_ALG,
		.salg_type = "hash",
	};

	LIB_DATA->name = gtkhash_hash_lib_linux_get_name(func->id);
	g_assert(LIB_DATA->name);
	strcpy((char *)addr.salg_name, LIB_DATA->name);

	LIB_DATA->sockfd = socket(AF_ALG, SOCK_SEQPACKET, 0);
	g_assert(LIB_DATA->sockfd != -1);

	int res = bind(LIB_DATA->sockfd, (struct sockaddr *)&addr, sizeof(addr));
	g_assert(res != -1); (void)res;

	LIB_DATA->connfd = accept(LIB_DATA->sockfd, NULL, NULL);
	g_assert(LIB_DATA->connfd != -1);
}

void gtkhash_hash_lib_linux_update(struct hash_func_s *func,
	const uint8_t *buffer, const size_t size)
{
	ssize_t send_size = send(LIB_DATA->connfd, buffer, size, MSG_MORE);
	g_assert(send_size == (ssize_t)size); (void)send_size;
}

void gtkhash_hash_lib_linux_stop(struct hash_func_s *func)
{
	close(LIB_DATA->connfd);
	close(LIB_DATA->sockfd);
	g_free(LIB_DATA);
}

uint8_t *gtkhash_hash_lib_linux_finish(struct hash_func_s *func, size_t *size)
{
	uint8_t *digest = NULL;

	// Kernel CRC32C, XXH64 are little-endian
	if (func->id == HASH_FUNC_CRC32C || func->id == HASH_FUNC_XXH64) {
		union {
			uint64_t u64;
			uint32_t u32;
		} digest_u;

		*size = read(LIB_DATA->connfd, &digest_u, func->digest_size);

		if (func->digest_size == 8)
			digest_u.u64 = GUINT64_SWAP_LE_BE(digest_u.u64);
		else if (func->digest_size == 4)
			digest_u.u32 = GUINT32_SWAP_LE_BE(digest_u.u32);

		digest = g_memdup(&digest_u, func->digest_size);
	} else {
		digest = g_malloc(func->digest_size);
		*size = read(LIB_DATA->connfd, digest, func->digest_size);
	}

	close(LIB_DATA->connfd);
	close(LIB_DATA->sockfd);
	g_free(LIB_DATA);

	return digest;
}
