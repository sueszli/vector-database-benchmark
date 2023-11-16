/*
 * Soft:        Keepalived is a failover program for the LVS project
 *              <www.linuxvirtualserver.org>. It monitor & manipulate
 *              a loadbalanced server pool using multi-layer checks.
 *
 * Part:        PING checker.
 *
 * Author:      Jie Liu, <liujie165@huawei.com>
 *
 *              This program is distributed in the hope that it will be useful,
 *              but WITHOUT ANY WARRANTY; without even the implied warranty of
 *              MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *              See the GNU General Public License for more details.
 *
 *              This program is free software; you can redistribute it and/or
 *              modify it under the terms of the GNU General Public License
 *              as published by the Free Software Foundation; either version
 *              2 of the License, or (at your option) any later version.
 *
 * Copyright (C) 2019-2019 Alexandre Cassen, <acassen@gmail.com>
 */
#include "config.h"

/* system includes */
#include <netinet/in.h>
#include <netinet/ip_icmp.h>
#include <netinet/icmp6.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <fcntl.h>

#include "check_ping.h"
#include "check_api.h"
#include "logger.h"
#include "layer4.h"
#include "parser.h"
#include "smtp.h"
#include "ipwrapper.h"
#include "check_parser.h"

#define ICMP_BUFSIZE 128
#define SOCK_RECV_BUFF 128*1024

static const char * const ping_group_range = "/proc/sys/net/ipv4/ping_group_range";

static gid_t save_gid_min;
static bool checked_ping_group_range;

static void icmp_connect_thread(thread_ref_t);
static void icmp_ping_thread(thread_ref_t);

bool
set_ping_group_range(bool set)
{
	char buf[10 + 1 + 10 + 1 + 1];	/* 2000000000<TAB>4294967295<NL> */
	int fd;
	ssize_t len, ret;
	unsigned long val[2];
	char *endptr;

	if (set == checked_ping_group_range)
		return true;

	fd = open(ping_group_range, O_RDWR);
	if (fd == -1)
		return false;
	len = read(fd, buf, sizeof(buf));

	if (len == -1 || len == sizeof(buf)) {
		close(fd);
		return false;
	}

	buf[len] = '\0';

	val[0] = strtoul(buf, &endptr, 10);
#if ULONG_MAX >= 1UL << 32
	if (val[0] >= 1UL << 32 || (*endptr != '\t' && *endptr != ' ')) {
		close(fd);
		return false;
	}
#endif

	val[1] = strtol(endptr + 1, &endptr, 10);
#if ULONG_MAX >= 1UL << 32
	if (val[1] >= 1UL << 32 || *endptr != '\n') {
		close(fd);
		return false;
	}
#endif

	checked_ping_group_range = set;

	if ((set && val[0] == 0) ||
	    (!set && val[0] == save_gid_min)) {
		close(fd);
		return true;
	}

	if (set && val[0] > 1 && val[1] > 1)
		log_message(LOG_INFO, "Warning: %s being expanded from %lu %lu to 0 %lu",
				ping_group_range, val[0], val[1], val[1]);

	len = sprintf(buf, "%u\t%lu\t\n", set ? 0 : save_gid_min, val[1]);

	lseek(fd, 0, SEEK_SET);
	ret = write(fd, buf, len);
	if (ret == -1)
		log_message(LOG_INFO, "Write %s failed - errno %d", ping_group_range, errno);
	else if (ret != len)
		log_message(LOG_INFO, "Write %s wrote %zd bytes instead of %zd", ping_group_range, ret, len);
	close(fd);

	if (set)
		save_gid_min = val[0];

	return true;
}

/* Configuration stream handling */
static void
free_ping_check(checker_t *checker)
{
	FREE(checker->co);
	FREE_PTR(checker->data);
	FREE(checker);
}

static void
dump_ping_check(FILE *fp, const checker_t *checker)
{
	ping_check_t *ping_checker = CHECKER_ARG(checker);

	conf_write(fp, "   Keepalive method = PING_CHECK");
	conf_write(fp, "     ICMP seq no = %u", ping_checker->seq_no);
}

static bool
compare_ping_check(const checker_t *a, checker_t *b)
{
	return compare_conn_opts(a->co, b->co);
}

static const checker_funcs_t ping_checker_funcs = { CHECKER_PING, free_ping_check, dump_ping_check, compare_ping_check, NULL };

static void
ping_check_handler(__attribute__((unused)) const vector_t *strvec)
{
	ping_check_t *ping_check = sizeof(ping_check_t) ? MALLOC(sizeof (ping_check_t)) : NULL;

	/* queue new checker */
	queue_checker(&ping_checker_funcs, icmp_connect_thread, ping_check, CHECKER_NEW_CO(), true);

	if (!checked_ping_group_range)
		set_ping_group_range(true);
}

static void
ping_check_end_handler(void)
{
	if (!check_conn_opts(current_checker->co)) {
		dequeue_new_checker();
		return;
	}
}

void
install_ping_check_keyword(void)
{
	vpp_t check_ptr;

	/* We don't want some common keywords */
	install_keyword("PING_CHECK", &ping_check_handler);
	check_ptr = install_sublevel(VPP &current_checker);
	install_checker_common_keywords(true);
	install_level_end_handler(ping_check_end_handler);
	install_sublevel_end(check_ptr);
}

static enum connect_result
ping_it(int fd, checker_t *checker, conn_opts_t* co)
{
	ping_check_t *ping_checker = CHECKER_ARG(checker);

	struct icmphdr *icmp_hdr;
	char send_buf[sizeof(*icmp_hdr) + ICMP_BUFSIZE] __attribute__((aligned(__alignof__(struct icmphdr))));

	set_buf(send_buf + sizeof(*icmp_hdr), ICMP_BUFSIZE);

	icmp_hdr = PTR_CAST(struct icmphdr, send_buf);

	memset(icmp_hdr, 0, sizeof(*icmp_hdr));
	icmp_hdr->type = ICMP_ECHO;
	icmp_hdr->un.echo.sequence = htons(++ping_checker->seq_no);

	if (sendto(fd, send_buf, sizeof(send_buf), 0, PTR_CAST(struct sockaddr, &co->dst), sizeof(struct sockaddr)) < 0) {
		log_message(LOG_INFO, "send ICMP packet fail");
		return connect_error;
	}
	return connect_success;
}

static enum connect_result
recv_it(int fd, checker_t *checker)
{
	ping_check_t *ping_checker = CHECKER_ARG(checker);
	ssize_t len;
	const struct icmphdr *icmp_hdr;
	char recv_buf[sizeof(*icmp_hdr) + ICMP_BUFSIZE] __attribute__((aligned(__alignof__(struct icmphdr))));

	len = recv(fd, recv_buf, sizeof(recv_buf), 0);

	if (len < 0) {
		log_message(LOG_INFO, "recv ICMP packet error");
		return connect_error;
	}

	if ((size_t)len < sizeof(*icmp_hdr)) {
		log_message(LOG_INFO, "Error, got short ICMP packet, %zd bytes", len);
		return connect_error;
	}

	icmp_hdr = PTR_CAST_CONST(struct icmphdr, recv_buf);
	if (icmp_hdr->type != ICMP_ECHOREPLY) {
		log_message(LOG_INFO, "Got ICMP packet with type 0x%x", icmp_hdr->type);
		return connect_error;
	}

	if (ntohs(icmp_hdr->un.echo.sequence) != ping_checker->seq_no) {
		log_message(LOG_INFO, "Ping reply expected seq no %u, received %u", ping_checker->seq_no, ntohs(icmp_hdr->un.echo.sequence));
		return connect_error;
	}

	return connect_success;
}

static enum connect_result
ping6_it(int fd, checker_t *checker, conn_opts_t* co)
{
	ping_check_t *ping_checker = CHECKER_ARG(checker);
	struct icmp6_hdr* icmp6_hdr;
	char send_buf[sizeof(*icmp6_hdr) + ICMP_BUFSIZE] __attribute__((aligned(__alignof__(struct icmp6_hdr))));

	set_buf(send_buf + sizeof(*icmp6_hdr), ICMP_BUFSIZE);

	icmp6_hdr = PTR_CAST(struct icmp6_hdr, &send_buf);

	memset(icmp6_hdr, 0, sizeof(*icmp6_hdr));
	icmp6_hdr->icmp6_type = ICMP6_ECHO_REQUEST;
	icmp6_hdr->icmp6_seq = htons(++ping_checker->seq_no);

	if (sendto(fd, send_buf, sizeof(send_buf), 0, PTR_CAST(struct sockaddr, &co->dst), sizeof(struct sockaddr_in6)) < 0) {
		log_message(LOG_INFO, "send ICMPv6 packet fail - errno %d", errno);
		return connect_error;
	}

	return connect_success;
}

static enum connect_result
recv6_it(int fd, checker_t *checker)
{
	ping_check_t *ping_checker = CHECKER_ARG(checker);
	ssize_t len;
	const struct icmp6_hdr* icmp6_hdr;
	char recv_buf[sizeof (*icmp6_hdr) + ICMP_BUFSIZE] __attribute__((aligned(__alignof__(struct icmp6_hdr))));

	len = recv(fd, recv_buf, sizeof(recv_buf), 0);

	if (len < 0) {
		log_message(LOG_INFO, "recv ICMPv6 packet error");
		return connect_error;
	}

	if ((size_t)len < sizeof(*icmp6_hdr)) {
		log_message(LOG_INFO, "Error, got short ICMPv6 packet, %zd bytes", len);
		return connect_error;
	}

	icmp6_hdr = PTR_CAST_CONST(struct icmp6_hdr, recv_buf);
	if (icmp6_hdr->icmp6_type != ICMP6_ECHO_REPLY) {
		log_message(LOG_INFO, "Got ICMPv6 packet with type 0x%x", icmp6_hdr->icmp6_type);
		return connect_error;
	}

	if (ntohs(icmp6_hdr->icmp6_seq) != ping_checker->seq_no) {
		log_message(LOG_INFO, "Ping reply expected seq no %u, received %u", ping_checker->seq_no, ntohs(icmp6_hdr->icmp6_seq));
		return connect_error;
	}

	return connect_success;
}

static void
icmp_epilog(thread_ref_t thread, bool is_success)
{
	checker_t *checker;
	unsigned long delay;
	bool checker_was_up;
	bool rs_was_alive;

	checker = THREAD_ARG(thread);

	delay = checker->delay_loop;
	if (is_success || ((checker->is_up || !checker->has_run) && checker->retry_it >= checker->retry)) {
		checker->retry_it = 0;

		if (is_success && (!checker->is_up || !checker->has_run)) {
			log_message(LOG_INFO, "ICMP connection to %s success."
					, FMT_CHK(checker));
			checker_was_up = checker->is_up;
			rs_was_alive = checker->rs->alive;
			update_svr_checker_state(UP, checker);
			if (checker->rs->smtp_alert && !checker_was_up &&
			    (rs_was_alive != checker->rs->alive || !global_data->no_checker_emails))
				smtp_alert(SMTP_MSG_RS, checker, NULL,
					   "=> ICMP CHECK succeed on service <=");
		} else if (!is_success &&
			   (checker->is_up || !checker->has_run)) {
			if (checker->retry && checker->has_run)
				log_message(LOG_INFO
				    , "ICMP CHECK on service %s of %s failed after %u retries."
				    , FMT_CHK(checker), FMT_VS(checker->vs)
				    , checker->retry);
			else
				log_message(LOG_INFO
				    , "ICMP CHECK on service %s failed."
				    , FMT_CHK(checker));
			checker_was_up = checker->is_up;
			rs_was_alive = checker->rs->alive;
			update_svr_checker_state(DOWN, checker);
			if (checker->rs->smtp_alert && checker_was_up &&
			    (rs_was_alive != checker->rs->alive || !global_data->no_checker_emails))
				smtp_alert(SMTP_MSG_RS, checker, NULL,
					   "=> ICMP CHECK failed on service <=");
		}
	} else if (checker->is_up) {
		delay = checker->delay_before_retry;
		++checker->retry_it;
	}

	checker->has_run = true;

	if (is_success)
		thread_add_read(thread->master, icmp_ping_thread, checker, thread->u.f.fd, delay, THREAD_DESTROY_CLOSE_FD);
	else {
		thread_close_fd(thread);
		thread_add_timer(thread->master, icmp_connect_thread, checker, delay);
	}
}

static void
icmp_check_thread(thread_ref_t thread)
{
	checker_t *checker = THREAD_ARG(thread);
	enum connect_result status;

	if (thread->type == THREAD_READ_TIMEOUT) {
		if (checker->is_up &&
		    (global_data->checker_log_all_failures || checker->log_all_failures))
			log_message(LOG_INFO, "ICMP connection to address %s timeout.", FMT_CHK(checker));
		status = connect_error;
	} else
		status = checker->co->dst.ss_family == AF_INET ?
				recv_it(thread->u.f.fd, checker) : recv6_it(thread->u.f.fd, checker);

	/*
	 * If status = connect_success, then we start udp check with the record of icmp failed times.
	 * Otherwise we will do the icmp connect again until it reaches the unhealthy threshold.
	 * we handle fd uniform.
	 */
	if (status == connect_success)
		icmp_epilog(thread, true);
	else if (status == connect_error) {
		if (checker->is_up &&
		    thread->type != THREAD_READ_TIMEOUT &&
		    (global_data->checker_log_all_failures || checker->log_all_failures))
			log_message(LOG_INFO, "ICMP connection to %s of %s failed."
				,FMT_CHK(checker), FMT_VS(checker->vs));
		icmp_epilog(thread, false);
	}
}

static void
icmp_ping_thread(thread_ref_t thread)
{
	checker_t *checker = THREAD_ARG(thread);
	conn_opts_t *co = checker->co;
	enum connect_result status;

	if (thread->type == THREAD_READ_TIMEOUT) {
		/* Send next ICMP echo request */
		if (co->dst.ss_family == AF_INET)
			status = ping_it(thread->u.f.fd, checker, co);
		else
			status = ping6_it(thread->u.f.fd, checker, co);

		/* handle icmp send status & register check worker thread */
		if (udp_icmp_check_state(thread->u.f.fd, status, thread, icmp_check_thread, co->connection_to))
			icmp_epilog(thread, false);
	} else if (thread->type == THREAD_READY_READ_FD) {
		/* This shouldn't happen */
		ssize_t len;
		char recv_buf[sizeof (struct icmp6_hdr) + ICMP_BUFSIZE] __attribute__((aligned(__alignof__(struct icmp6_hdr))));

		len = recv(thread->u.f.fd, recv_buf, sizeof(recv_buf), 0);

		if (len > 0)
			log_message(LOG_INFO, "unexpected ping check receive packet");
		thread_add_read_sands(thread->master, icmp_ping_thread, checker, thread->u.f.fd, &thread->sands, THREAD_DESTROY_CLOSE_FD);
	} else {
		/* THREAD_READ_ERROR */
		thread_close_fd(thread);
		thread_add_timer_sands(thread->master, icmp_connect_thread, checker, &thread->sands);
	}
}

static void
icmp_connect_thread(thread_ref_t thread)
{
	checker_t *checker = THREAD_ARG(thread);
	ping_check_t *ping_checker = CHECKER_ARG(checker);
	conn_opts_t *co = checker->co;
	int fd;
	int size = SOCK_RECV_BUFF;

	if (!checker->enabled) {
		thread_add_timer(thread->master, icmp_connect_thread, checker,
				checker->delay_loop);
		return;
	}

	/*
	 * If we config a real server in several virtual server, the icmp_ratelimit should be cancelled.
	 * echo 0 > /proc/sys/net/ipv4/icmp_ratelimit
	 */

	/*
	 * Using SOCK_DGRAM with IPPROTO_ICMP/ICMPV6 means that the kernel will insert the ICMP
	 * id value. In order to keep the same id value for each ICMP echo request, we need to keep
	 * the socket open.
	 *
	 * The alternative is to use SOCK_RAW, but then we can't ensure the uniqueness of the id
	 * field.
	 */
	if ((fd = socket(co->dst.ss_family, SOCK_DGRAM | SOCK_CLOEXEC | SOCK_NONBLOCK,
			 co->dst.ss_family == AF_INET ? IPPROTO_ICMP : IPPROTO_ICMPV6)) == -1) {
		log_message(LOG_INFO, "ICMP%s connect fail to create socket. Rescheduling.",
				co->dst.ss_family == AF_INET ? "" : "v6");
		thread_add_timer(thread->master, icmp_connect_thread, checker,
				checker->delay_loop);
		return;
	}

	if (setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &size, sizeof(size)))
		log_message(LOG_INFO, "setsockopt SO_RCVBUF for socket %d failed (%d) - %m", fd, errno);

	ping_checker->seq_no = 0;

	thread_add_read(thread->master, icmp_ping_thread, checker, fd, 0, THREAD_DESTROY_CLOSE_FD);
}

#ifdef THREAD_DUMP
void
register_check_ping_addresses(void)
{
	register_thread_address("icmp_check_thread", icmp_check_thread);
	register_thread_address("icmp_connect_thread", icmp_connect_thread);
	register_thread_address("icmp_ping_thread", icmp_ping_thread);
}
#endif
