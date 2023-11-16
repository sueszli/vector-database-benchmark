/*
 * Soft:        Keepalived is a failover program for the LVS project
 *              <www.linuxvirtualserver.org>. It monitor & manipulate
 *              a loadbalanced server pool using multi-layer checks.
 *
 * Part:        SMTP CHECK. Check an SMTP-server.
 *
 * Authors:     Jeremy Rumpf, <jrumpf@heavyload.net>
 *              Alexandre Cassen, <acassen@linux-vs.org>
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
 * Copyright (C) 2001-2017 Alexandre Cassen, <acassen@gmail.com>
 */

#include "config.h"

#include <errno.h>
#include <unistd.h>
#include <stdio.h>
#include <ctype.h>

#include "check_smtp.h"
#include "logger.h"
#include "ipwrapper.h"
#include "utils.h"
#include "parser.h"
#include "layer4.h"
#include "smtp.h"
#ifdef THREAD_DUMP
#include "scheduler.h"
#endif
#include "check_parser.h"


/* Specifying host blocks within the SMTP checker is deprecated, but currently
 * still supported. All code to support it is in WITH_HOST_ENTRIES conditional
 * compilation, so it is easy to remove all the code eventually. */
#define WITH_HOST_ENTRIES

#ifdef WITH_HOST_ENTRIES
static LIST_HEAD_INITIALIZE(host_list); /* ref_co_t */
typedef struct _ref_co {
	conn_opts_t	*co;

	/* Linked list member */
	list_head_t	e_list;
} ref_co_t;
static checker_t *current_checker_host;
#endif

static void smtp_connect_thread(thread_ref_t);
static void smtp_start_check_thread(thread_ref_t);
static void smtp_engine_thread(thread_ref_t);

/* Used as a callback from the checker api, queue_checker(),
 * to free up a checker entry and all its associated data.
 */
static void
free_smtp_check(checker_t *checker)
{
	smtp_checker_t *smtp_checker = checker->data;

	FREE_PTR(checker->co);
	FREE_CONST(smtp_checker->helo_name);
	FREE(smtp_checker);
	FREE(checker);
}

/*
 * Callback for whenever we've been requested to dump our
 * configuration.
 */
static void
dump_smtp_check(FILE *fp, const checker_t *checker)
{
	const smtp_checker_t *smtp_checker = checker->data;

	conf_write(fp, "   Keepalive method = SMTP_CHECK");
	conf_write(fp, "   helo = %s", smtp_checker->helo_name);
}

static bool
compare_smtp_check(const checker_t *old_c, checker_t *new_c)
{
	const smtp_checker_t *old = old_c->data;
	const smtp_checker_t *new = new_c->data;

	if (strcmp(old->helo_name, new->helo_name) != 0)
		return false;
	if (!compare_conn_opts(old_c->co, new_c->co))
		return false;

	return true;
}

static const checker_funcs_t smtp_checker_funcs = { CHECKER_SMTP, free_smtp_check, dump_smtp_check, compare_smtp_check, NULL };

/*
 * Callback for whenever an SMTP_CHECK keyword is encountered
 * in the config file.
 */
static void
smtp_check_handler(__attribute__((unused)) const vector_t *strvec)
{
	smtp_checker_t *smtp_checker;
	conn_opts_t *co;

	PMALLOC(smtp_checker);
	PMALLOC(co);
	co->connection_to = UINT_MAX;

	/* Have the checker queue code put our checker into the real server's checkers_queue list. */
	queue_checker(&smtp_checker_funcs, smtp_start_check_thread, smtp_checker, co, true);

	/* We need to be able to check if anything has been set */
	co->dst.ss_family = AF_UNSPEC;
	PTR_CAST(struct sockaddr_in, &co->dst)->sin_port = 0;
}

static void
smtp_check_end_handler(void)
{
	smtp_checker_t *smtp_checker = current_checker->data;
	checker_t *checker = current_checker;
#ifdef WITH_HOST_ENTRIES
	smtp_checker_t *new_smtp_checker;
	conn_opts_t *co;
	ref_co_t *rco, *rco_tmp;
	list_head_t sav_rs_list;
#endif

	if (!smtp_checker->helo_name)
		smtp_checker->helo_name = STRDUP(SMTP_DEFAULT_HELO);

#ifdef WITH_HOST_ENTRIES
	if (!list_empty(&host_list))
		log_message(LOG_INFO, "The SMTP_CHECK host block is deprecated. Please define additional checkers.");
#endif

	/* If any connection component has been configured, or there are no (deprecated) host entries,
	 * we want to use any information provided, using defaults as necessary. */
#ifdef WITH_HOST_ENTRIES
	/* Have any of the connection parameters been set, or are there no hosts? */
	if (current_checker->co->dst.ss_family != AF_UNSPEC ||
	    PTR_CAST(struct sockaddr_in, &current_checker->co->dst)->sin_port ||
	    current_checker->co->bindto.ss_family != AF_UNSPEC ||
	    PTR_CAST(struct sockaddr_in, &current_checker->co->bindto)->sin_port ||
	    current_checker->co->bind_if[0] ||
	    list_empty(&host_list) ||
#ifdef _WITH_SO_MARK_
	    current_checker->co->fwmark ||
#endif
	    current_checker->co->connection_to != UINT_MAX)
#endif
	{
		/* Set any necessary defaults. NOTE: we are relying on
		 * struct sockaddr_in and sockaddr_in6 port offsets being the same. */
		uint16_t saved_port = PTR_CAST(struct sockaddr_in, &current_checker->co->dst)->sin_port;
		if (current_checker->co->dst.ss_family == AF_UNSPEC) {
			current_checker->co->dst = current_rs->addr;
			if (saved_port)
				checker_set_dst_port(&current_checker->co->dst, saved_port);
		}
		if (!saved_port)
			checker_set_dst_port(&current_checker->co->dst, PTR_CAST(struct sockaddr_in, &current_rs->addr)->sin_port);

		if (!check_conn_opts(current_checker->co)) {
			dequeue_new_checker();
			return;
		}
	}
#ifdef WITH_HOST_ENTRIES
	else {
		/* No connection options have been specified, but there
		 * is at least one host entry. Use that host entry's
		 * connection options for the main checker. */
		FREE(current_checker->co);

		rco = list_first_entry(&host_list, ref_co_t, e_list);
		current_checker->co = rco->co;
		list_del_init(&rco->e_list);
		FREE(rco);
	}
#endif
 
	/* Set the connection timeout if not set */
	unsigned conn_to = current_rs->connection_to;
	if (conn_to == UINT_MAX)
		conn_to = current_vs->connection_to;

	if (current_checker->co->connection_to == UINT_MAX)
		current_checker->co->connection_to = conn_to;

#ifdef WITH_HOST_ENTRIES
	/* Create a new checker for each host on the host list */
	list_for_each_entry_safe(rco, rco_tmp, &host_list, e_list) {
		co = rco->co;
		PMALLOC(new_smtp_checker);
		*new_smtp_checker = *smtp_checker;

		if (co->connection_to == UINT_MAX)
			co->connection_to = conn_to;

		new_smtp_checker->helo_name = STRDUP(smtp_checker->helo_name);

		queue_checker(&smtp_checker_funcs, smtp_start_check_thread,
					      new_smtp_checker, NULL, true);

		/* Copy the checker info, but preserve the list_head entry, th
		 * co pointer and the pointer to new_smtp_checker. */
		sav_rs_list = current_checker->rs_list;
		*current_checker = *checker;
		current_checker->rs_list = sav_rs_list;
		current_checker->co = co;
		current_checker->data = new_smtp_checker;

		/* queue the checker */
		list_add_tail(&current_checker->rs_list, &checker->rs->checkers_list);

		list_del_init(&rco->e_list);
		FREE(rco);
	}
#endif
}

#ifdef WITH_HOST_ENTRIES
/* Callback for "host" keyword */
static void
smtp_host_handler(__attribute__((unused)) const vector_t *strvec)
{
	PMALLOC(current_checker_host);
	PMALLOC(current_checker_host->co);

	/* Default to the RS */
	current_checker_host->co->dst = current_rs->addr;
}

static void
smtp_host_end_handler(void)
{
	ref_co_t *rco;

	if (!check_conn_opts(current_checker_host->co))
		FREE(current_checker_host->co);
	else {
		PMALLOC(rco);
		INIT_LIST_HEAD(&rco->e_list);
		rco->co = current_checker_host->co;

		list_add_tail(&rco->e_list, &host_list);
	}

	FREE(current_checker_host);
}
#endif

/* "helo_name" keyword */
static void
smtp_helo_name_handler(const vector_t *strvec)
{
	smtp_checker_t *smtp_checker = current_checker->data;

	if (vector_size(strvec) < 2) {
		report_config_error(CONFIG_GENERAL_ERROR, "SMTP_CHECK helo name missing");
		return;
	}

	if (smtp_checker->helo_name) {
		report_config_error(CONFIG_GENERAL_ERROR, "SMTP_CHECK helo name already specified");
		FREE_CONST(smtp_checker->helo_name);
	}

	smtp_checker->helo_name = set_value(strvec);
}

/* Config callback installer */
void
install_smtp_check_keyword(void)
{
	vpp_t check_ptr;
#ifdef WITH_HOST_ENTRIES
	vpp_t check_ptr1;
#endif

	/*
	 * Notify the config log parser that we need to be notified via
	 * callbacks when the following keywords are encountered in the
	 * keepalive.conf file.
	 */
	install_keyword("SMTP_CHECK", &smtp_check_handler);
	check_ptr = install_sublevel(VPP &current_checker);
	install_keyword("helo_name", &smtp_helo_name_handler);

	install_checker_common_keywords(true);

	/*
	 * The host list feature is deprecated. It makes config fussy by
	 * adding another nesting level and is excessive since it is possible
	 * to attach multiple checkers to a RS.
	 * So these keywords below are kept for compatibility with users'
	 * existing configs.
	 */
#ifdef WITH_HOST_ENTRIES
	install_keyword("host", &smtp_host_handler);
	check_ptr1 = install_sublevel(VPP &current_checker_host);
	install_checker_common_keywords(true);
	install_level_end_handler(smtp_host_end_handler);
	install_sublevel_end(check_ptr1);
#endif

	install_level_end_handler(&smtp_check_end_handler);
	install_sublevel_end(check_ptr);
}

/*
 * Final handler. Determines if we need a retry or not.
 * Also has to make a decision if we need to bring the resulting
 * service down in case of error.
 */
static int __attribute__ ((format (printf, 2, 3)))
smtp_final(thread_ref_t thread, const char *format, ...)
{
	checker_t *checker = THREAD_ARG(thread);
	char error_buff[512];
	char smtp_buff[542];
	va_list varg_list;
	bool checker_was_up;
	bool rs_was_alive;

	/* Error or no error we should always have to close the socket */
	if (thread->type != THREAD_READY_TIMER)
		thread_close_fd(thread);

	if (format) {
		/* Always syslog the error when the real server is up */
		if ((checker->is_up || !checker->has_run) &&
		    (global_data->checker_log_all_failures ||
		     checker->log_all_failures ||
		     checker->retry_it >= checker->retry)) {
			/* prepend format with the "SMTP_CHECK " string */
			strcpy_safe(error_buff, "SMTP_CHECK ");
			strncat(error_buff, format, sizeof(error_buff) - 11 - 1);

			va_start(varg_list, format);
			vlog_message(LOG_INFO, error_buff, varg_list);
			va_end(varg_list);
		}

		/*
		 * If we still have retries left, try this host again by
		 * scheduling the main thread to check it again after the
		 * configured backoff delay. Otherwise down the RS.
		 */
		if (++checker->retry_it <= checker->retry) {
			thread_add_timer(thread->master, smtp_connect_thread, checker,
					 checker->delay_before_retry);
			return 0;
		}

		/*
		 * No more retries, pull the real server from the virtual server.
		 * Only smtp_alert if it wasn't previously down. It should
		 * be noted that smtp_alert makes a copy of the string arguments, so
		 * we don't have to keep them statically allocated.
		 */
		if (checker->is_up || !checker->has_run) {
			checker_was_up = checker->is_up;
			rs_was_alive = checker->rs->alive;
			update_svr_checker_state(DOWN, checker);
			if (checker->rs->smtp_alert && checker_was_up &&
			    (rs_was_alive != checker->rs->alive || !global_data->no_checker_emails)) {
				if (format != NULL) {
					snprintf(error_buff, sizeof(error_buff), "=> CHECK failed on service : %s <=", format);
					va_start(varg_list, format);
					vsnprintf(smtp_buff, sizeof(smtp_buff), error_buff, varg_list);
					va_end(varg_list);
				} else
					strncpy(smtp_buff, "=> CHECK failed on service <=", sizeof(smtp_buff));

				smtp_buff[sizeof(smtp_buff) - 1] = '\0';
				smtp_alert(SMTP_MSG_RS, checker, NULL, smtp_buff);
			}
		}

		/* Reschedule the main thread using the configured delay loop */
		thread_add_timer(thread->master, smtp_start_check_thread, checker, checker->delay_loop);

		return 0;
	}

	/*
	 * Ok this host was successful, increment to the next host in the list
	 * and reset the retry_it counter. We'll then reschedule the main thread again.
	 * If host_ptr exceeds the end of the list, smtp_connect_main_thread will
	 * take note and bring up the real server as well as inject the delay_loop.
	 */
	checker->retry_it = 0;

	/*
	 * Set the internal host pointer to the host that we'll be
	 * working on. If it's NULL, we've successfully tested all hosts.
	 * We'll bring the service up (if it's not already), reset the host list,
	 * and insert the delay loop. When we get scheduled again the host list
	 * will be reset and we will continue on checking them one by one.
	 */
	if (!checker->is_up || !checker->has_run) {
		log_message(LOG_INFO, "Remote SMTP server %s succeed on service."
				    , FMT_CHK(checker));

		checker_was_up = checker->is_up;
		rs_was_alive = checker->rs->alive;
		update_svr_checker_state(UP, checker);
		if (checker->rs->smtp_alert && !checker_was_up &&
		    (rs_was_alive != checker->rs->alive || !global_data->no_checker_emails))
			smtp_alert(SMTP_MSG_RS, checker, NULL,
				   "=> CHECK succeed on service <=");
	}

	checker->has_run = true;

	thread_add_timer(thread->master, smtp_start_check_thread, checker, checker->delay_loop);

	return 0;
}

/*
 * One thing to note here is we do a very cheap check for a newline.
 * We could receive two lines (with two newline characters) in a
 * single packet, but we don't care. We are only looking at the
 * SMTP response codes at the beginning anyway.
 */
static void
smtp_get_line_cb(thread_ref_t thread)
{
	checker_t *checker = THREAD_ARG(thread);
	smtp_checker_t *smtp_checker = CHECKER_ARG(checker);
	conn_opts_t *smtp_host = checker->co;
	ssize_t r;
	char *nl;

	/* Handle read timeout */
	if (thread->type == THREAD_READ_TIMEOUT) {
		smtp_final(thread, "Read timeout from server %s"
				    , FMT_SMTP_RS(smtp_host));
		return;
	}

	/* wrap the buffer, if full, by clearing it */
	if (smtp_checker->buff_ctr >= SMTP_BUFF_MAX - 1) {
		log_message(LOG_INFO, "SMTP_CHECK Buffer overflow reading from server %s. "
				      "Increase SMTP_BUFF_MAX in check_smtp.h"
				    , FMT_SMTP_RS(smtp_host));
		smtp_checker->buff_ctr = 0;
	}

	/* read the data */
	r = read(thread->u.f.fd, smtp_checker->buff + smtp_checker->buff_ctr,
		 SMTP_BUFF_MAX - smtp_checker->buff_ctr - 1);

	if (r == -1 && (check_EAGAIN(errno) || check_EINTR(errno))) {
		thread_add_read(thread->master, smtp_get_line_cb, checker,
				thread->u.f.fd, smtp_host->connection_to, THREAD_DESTROY_CLOSE_FD);
		return;
	}

	/*
	 * If the connection was closed or there was
	 * some sort of error, notify smtp_final()
	 */
	if (r <= 0) {
		smtp_final(thread, "Read failure from server %s"
				     , FMT_SMTP_RS(smtp_host));
		return;
	}

	smtp_checker->buff_ctr += (size_t)r;
	smtp_checker->buff[smtp_checker->buff_ctr] = '\0';

	/* check if we have a newline, if so, callback */
	if ((nl = strchr(smtp_checker->buff, '\n'))) {
		*nl = '\0';

#ifdef _CHECKER_DEBUG_
		if (do_checker_debug)
			log_message(LOG_DEBUG, "SMTP_CHECK %s < %s" , FMT_SMTP_RS(smtp_host) , smtp_checker->buff);
#endif

		smtp_engine_thread(thread);

		return;
	}

	/*
	 * Last case, we haven't read enough data yet
	 * to pull a newline. Schedule ourselves for
	 * another round.
	 */
	thread_add_read(thread->master, smtp_get_line_cb, checker,
			thread->u.f.fd, smtp_host->connection_to, THREAD_DESTROY_CLOSE_FD);
}

/*
 * Ok a caller has asked us to asyncronously schedule a single line
 * to be received from the server. They have also passed us a call back
 * function that we'll call once we have the newline. If something bad
 * happens, the caller assumes we'll pass the error off to smtp_final(),
 * which will either down the real server or schedule a retry. The
 * function smtp_get_line_cb is what does the dirty work since the
 * scheduler can only accept a single *thread argument.
 */
static void
smtp_get_line(thread_ref_t thread)
{
	checker_t *checker = THREAD_ARG(thread);
	smtp_checker_t *smtp_checker = CHECKER_ARG(checker);
	conn_opts_t *smtp_host = checker->co;

	/* clear the buffer */
	smtp_checker->buff_ctr = 0;

	/* schedule the I/O with our helper function  */
	thread_add_read(thread->master, smtp_get_line_cb, checker,
		thread->u.f.fd, smtp_host->connection_to, THREAD_DESTROY_CLOSE_FD);
	thread_del_write(thread);
}

/*
 * The scheduler function that puts the data out on the wire.
 * All our data will fit into one packet, so we only check if
 * the current write would block or not. If it wants to block,
 * we'll return to the scheduler and try again later.
 */
static void
smtp_put_line_cb(thread_ref_t thread)
{
	checker_t *checker = THREAD_ARG(thread);
	smtp_checker_t *smtp_checker = CHECKER_ARG(checker);
	conn_opts_t *smtp_host = checker->co;
	ssize_t w;

	/* Handle read timeout */
	if (thread->type == THREAD_WRITE_TIMEOUT) {
		smtp_final(thread, "Write timeout to server %s"
				     , FMT_SMTP_RS(smtp_host));
		return;
	}

	/* write the data */
	w = write(thread->u.f.fd, smtp_checker->buff, smtp_checker->buff_ctr);

	if (w == -1 && (check_EAGAIN(errno) || check_EINTR(errno))) {
		thread_add_write(thread->master, smtp_put_line_cb, checker,
				 thread->u.f.fd, smtp_host->connection_to, THREAD_DESTROY_CLOSE_FD);
		return;
	}

#ifdef _CHECKER_DEBUG_
	if (do_checker_debug)
		log_message(LOG_DEBUG, "SMTP_CHECK %s > %s" , FMT_SMTP_RS(smtp_host) , smtp_checker->buff);
#endif

	/*
	 * If the connection was closed or there was
	 * some sort of error, notify smtp_final()
	 */
	if (w <= 0) {
		smtp_final(thread, "Write failure to server %s"
				     , FMT_SMTP_RS(smtp_host));
		return;
	}

	/* Execute the callback */
	smtp_engine_thread(thread);
}

/*
 * This is the same as smtp_get_line() except that we're sending a
 * line of data instead of receiving one.
 */
static void
smtp_put_line(thread_ref_t thread)
{
	checker_t *checker = THREAD_ARG(thread);
	smtp_checker_t *smtp_checker = CHECKER_ARG(checker);

	smtp_checker->buff_ctr = strlen(smtp_checker->buff);

	/* schedule the I/O with our helper function  */
	smtp_put_line_cb(thread);

	return;
}

/*
 * Ok, our goal here is to snag the status code out of the
 * buffer and return it as an integer. If it's not legible,
 * return -1.
 */
static int
smtp_get_status(smtp_checker_t *smtp_checker)
{
	char *buff = smtp_checker->buff;
	int status;
	char *endptr;

	status = strtoul(buff, &endptr, 10);
	if (endptr - buff != 3 ||
	    (*endptr && *endptr != ' '))
		return -1;

	return status;
}

/*
 * We have a connected socket and are ready to begin
 * the conversation. This function schedules itself to
 * be called via callbacks and tracking state in
 * smtp_checker->state. Upon first calling, smtp_checker->state
 * should be set to SMTP_START.
 */
static void
smtp_engine_thread(thread_ref_t thread)
{
	checker_t *checker = THREAD_ARG(thread);
	smtp_checker_t *smtp_checker = CHECKER_ARG(checker);
	conn_opts_t *smtp_host = checker->co;

	switch (smtp_checker->state) {

		/* First step, schedule to receive the greeting banner */
		case SMTP_START:
			/*
			 * Ok, if smtp_get_line schedules us back, we will
			 * have data to analyze. Otherwise, smtp_get_line
			 * will defer directly to smtp_final.
			 */
			smtp_checker->state = SMTP_HAVE_BANNER;
			smtp_get_line(thread);
			break;

		/* Second step, analyze banner, send HELO */
		case SMTP_HAVE_BANNER:
			/* Check for "220 some.mailserver.com" in the greeting */
			if (smtp_get_status(smtp_checker) != 220) {
				smtp_final(thread, "Bad greeting banner from server %s"
						     , FMT_SMTP_RS(smtp_host));
			} else {
				/*
				 * Schedule to send the HELO, smtp_put_line will
				 * defer directly to smtp_final on error.
				 */
				smtp_checker->state = SMTP_SENT_HELO;
				snprintf(smtp_checker->buff, SMTP_BUFF_MAX, "HELO %s\r\n",
					 smtp_checker->helo_name);
				smtp_put_line(thread);
			}
			break;

		/* Third step, schedule to read the HELO response */
		case SMTP_SENT_HELO:
			smtp_checker->state = SMTP_RECV_HELO;
			smtp_get_line(thread);
			break;

		/* Fourth step, analyze HELO return, send QUIT */
		case SMTP_RECV_HELO:
			/* Check for "250 Please to meet you..." */
			if (smtp_get_status(smtp_checker) != 250) {
				smtp_final(thread, "Bad HELO response from server %s"
						     , FMT_SMTP_RS(smtp_host));
			} else {
				smtp_checker->state = SMTP_SENT_QUIT;
				snprintf(smtp_checker->buff, SMTP_BUFF_MAX, "QUIT\r\n");
				smtp_put_line(thread);
			}
			break;

		/* Fifth step, schedule to receive QUIT confirmation */
		case SMTP_SENT_QUIT:
			smtp_checker->state = SMTP_RECV_QUIT;
			smtp_get_line(thread);
			break;

		/* Sixth step, wrap up success to smtp_final */
		case SMTP_RECV_QUIT:
			smtp_final(thread, NULL);
			break;

		default:
			/* We shouldn't be here */
			smtp_final(thread, "Unknown smtp engine state encountered");
			break;
	}
}

/*
 * Second step in the process. Here we'll see if the connection
 * to the host we're checking was successful or not.
 */
static void
smtp_check_thread(thread_ref_t thread)
{
	checker_t *checker = THREAD_ARG(thread);
	smtp_checker_t *smtp_checker = CHECKER_ARG(checker);
	conn_opts_t *smtp_host = checker->co;
	int status;

	status = tcp_socket_state(thread, smtp_check_thread, 0);
	switch (status) {
		case connect_error:
			smtp_final(thread, "Error connecting to server %s"
					     , FMT_SMTP_RS(smtp_host));
			break;

		case connect_timeout:
			smtp_final(thread, "Connection timeout to server %s"
					     , FMT_SMTP_RS(smtp_host));
			break;

		case connect_fail:
			smtp_final(thread, "Could not connect to server %s"
					     , FMT_SMTP_RS(smtp_host));
			break;

		case connect_success:
#ifdef _CHECKER_DEBUG_
			if (do_checker_debug)
				log_message(LOG_DEBUG, "SMTP_CHECK Remote SMTP server %s connected",
						     FMT_SMTP_RS(smtp_host));
#endif

			/* Enter the engine at SMTP_START */
			smtp_checker->state = SMTP_START;
			smtp_engine_thread(thread);
			break;

		default:
			/* we shouldn't be here */
			smtp_final(thread, "Unknown connection error to server %s"
					     , FMT_SMTP_RS(smtp_host));
			break;
	}
}

/*
 * This is the main thread, where all the action starts.
 * When the check daemon comes up, it goes down each real
 * server's checkers_queue and launches a thread for each
 * checker that got registered. This is the callback/event
 * function for that initial thread.
 *
 * It should be noted that we ARE responsible for scheduling
 * ourselves to run again. It doesn't have to be right here,
 * but eventually has to happen.
 */
static void
smtp_connect_thread(thread_ref_t thread)
{
	checker_t *checker = THREAD_ARG(thread);
	conn_opts_t *smtp_host;
	enum connect_result status;
	int sd;

	/* Let's review our data structures.
	 *
	 * Thread is the structure used by the sceduler
	 * for scheduling many types of events. thread->arg in this
	 * case points to a checker structure. The checker
	 * structure holds data about the vs and rs configurations
	 * as well as the delay loop, etc. Each real server
	 * defined in the keepalived.conf will more than likely have
	 * a checker structure assigned to it. Each checker structure
	 * has a data element that is meant to hold per checker
	 * configurations. So thread->arg(checker)->data points to
	 * a smtp_checker structure. In the smtp_checker structure
	 * we hold global configuration data for the smtp check.
	 *
	 * So this whole thing looks like this:
	 * thread->arg(checker)->data(smtp_checker)->host(smtp_host)
	 *
	 * To make life simple, we'll break the structures out so
	 * that "checker" always points to the current checker structure,
	 * "smtp_checker" points to the current smtp_checker structure.
	 */

	/*
	 * If we're disabled, we'll do nothing at all.
	 * But we still have to register ourselves again so
	 * we don't fall of the face of the earth.
	 */
	if (!checker->enabled) {
		thread_add_timer(thread->master, smtp_start_check_thread, checker,
				 checker->delay_loop);
		return;
	}

	smtp_host = checker->co;

	/* Create the socket, failing here should be an oddity */
	if ((sd = socket(smtp_host->dst.ss_family, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, IPPROTO_TCP)) == -1) {
		log_message(LOG_INFO, "SMTP_CHECK connection failed to create socket. Rescheduling.");
		thread_add_timer(thread->master, smtp_start_check_thread, checker,
				 checker->delay_loop);
		return;
	}

	status = tcp_bind_connect(sd, smtp_host);

	/* handle tcp connection status & register callback the next step in the process */
	if (tcp_connection_state(sd, status, thread, smtp_check_thread, smtp_host->connection_to, 0)) {
		if (status == connect_fail) {
			close(sd);
			smtp_final(thread, "Network unreachable for server %s - real server %s",
					   inet_sockaddrtos(&checker->co->dst),
					   inet_sockaddrtopair(&checker->rs->addr));
		} else {
			close(sd);
			log_message(LOG_INFO, "SMTP_CHECK socket bind failed. Rescheduling.");
			thread_add_timer(thread->master, smtp_start_check_thread, checker,
				checker->delay_loop);
		}
	}
}

static void
smtp_start_check_thread(thread_ref_t thread)
{
	checker_t *checker = THREAD_ARG(thread);

	checker->retry_it = 0;

	smtp_connect_thread(thread);
}

#ifdef THREAD_DUMP
void
register_check_smtp_addresses(void)
{
	register_thread_address("smtp_start_check_thread", smtp_start_check_thread);
	register_thread_address("smtp_check_thread", smtp_check_thread);
	register_thread_address("smtp_connect_thread", smtp_connect_thread);
	register_thread_address("smtp_get_line_cb", smtp_get_line_cb);
	register_thread_address("smtp_put_line_cb", smtp_put_line_cb);
}
#endif
