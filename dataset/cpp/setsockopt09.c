// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Copyright (c) 2022 SUSE LLC
 * Author: Marcos Paulo de Souza <mpdesouza@suse.com>
 * LTP port: Martin Doucha <mdoucha@suse.cz>
 */

/*\
 * [Description]
 *
 * Check for possible double free of rx_owner_map after switching packet
 * interface versions aka CVE-2021-22600.
 *
 * Kernel crash fixed in:
 *
 *  commit ec6af094ea28f0f2dda1a6a33b14cd57e36a9755
 *  Author: Willem de Bruijn <willemb@google.com>
 *  Date:   Wed Dec 15 09:39:37 2021 -0500
 *
 *  net/packet: rx_owner_map depends on pg_vec
 *
 *  commit c800aaf8d869f2b9b47b10c5c312fe19f0a94042
 *  Author: WANG Cong <xiyou.wangcong@gmail.com>
 *  Date:   Mon Jul 24 10:07:32 2017 -0700
 *
 *  packet: fix use-after-free in prb_retire_rx_blk_timer_expired()
 */

#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>

#include "tst_test.h"
#include "lapi/if_packet.h"

static int sock = -1;
static unsigned int pagesize;

static void setup(void)
{
	pagesize = SAFE_SYSCONF(_SC_PAGESIZE);
	tst_setup_netns();
}

static void run(void)
{
	unsigned int i, version = TPACKET_V3;
	struct tpacket_req3 req = {
		.tp_block_size = 4 * pagesize,
		.tp_frame_size = TPACKET_ALIGNMENT << 7,
		.tp_retire_blk_tov = 64,
		.tp_feature_req_word = TP_FT_REQ_FILL_RXHASH
	};

	for (i = 0; i < 5; i++) {
		req.tp_block_nr = 256;
		req.tp_frame_nr = req.tp_block_size * req.tp_block_nr;
		req.tp_frame_nr /= req.tp_frame_size;

		sock = SAFE_SOCKET(AF_PACKET, SOCK_RAW, 0);
		TEST(setsockopt(sock, SOL_PACKET, PACKET_VERSION, &version,
			sizeof(version)));

		if (TST_RET == -1 && TST_ERR == EINVAL)
			tst_brk(TCONF | TTERRNO, "TPACKET_V3 not supported");

		if (TST_RET) {
			tst_brk(TBROK | TTERRNO,
				"setsockopt(PACKET_VERSION, TPACKET_V3)");
		}

		/* Allocate owner map and then free it again */
		SAFE_SETSOCKOPT(sock, SOL_PACKET, PACKET_RX_RING, &req,
			sizeof(req));
		req.tp_block_nr = 0;
		req.tp_frame_nr = 0;
		SAFE_SETSOCKOPT(sock, SOL_PACKET, PACKET_RX_RING, &req,
			sizeof(req));

		/* Switch packet version and trigger double free of owner map */
		SAFE_SETSOCKOPT_INT(sock, SOL_PACKET, PACKET_VERSION,
			TPACKET_V2);
		SAFE_SETSOCKOPT(sock, SOL_PACKET, PACKET_RX_RING, &req,
			sizeof(req));
		SAFE_CLOSE(sock);

		/* Wait for socket timer to expire just in case */
		usleep(req.tp_retire_blk_tov * 3000);

		if (tst_taint_check()) {
			tst_res(TFAIL, "Kernel is vulnerable");
			return;
		}
	}

	tst_res(TPASS, "Nothing bad happened, probably");
}

static void cleanup(void)
{
	if (sock >= 0)
		SAFE_CLOSE(sock);
}

static struct tst_test test = {
	.test_all = run,
	.setup = setup,
	.cleanup = cleanup,
	.taint_check = TST_TAINT_W | TST_TAINT_D,
	.needs_kconfigs = (const char *[]) {
		"CONFIG_USER_NS=y",
		"CONFIG_NET_NS=y",
		NULL
	},
	.save_restore = (const struct tst_path_val[]) {
		{"/proc/sys/user/max_user_namespaces", "1024", TST_SR_SKIP},
		{}
	},
	.tags = (const struct tst_tag[]) {
		{"linux-git", "ec6af094ea28"},
		{"linux-git", "c800aaf8d869"},
		{"CVE", "2021-22600"},
		{}
	}
};
