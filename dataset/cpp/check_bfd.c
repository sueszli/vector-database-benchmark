/*
 * Soft:        Keepalived is a failover program for the LVS project
 *              <www.linuxvirtualserver.org>. It monitor & manipulate
 *              a loadbalanced server pool using multi-layer checks.
 *
 * Part:        BFD CHECK. Perform a system call to run an extra
 *              system prog or script.
 *
 * Authors:     Quentin Armitage <quentin@armitage.org.uk>
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

#include <unistd.h>
#include <stdio.h>
#include <inttypes.h>

#include "main.h"
#include "check_api.h"
#include "ipwrapper.h"
#include "logger.h"
#include "smtp.h"
#include "parser.h"
#include "global_data.h"
#include "global_parser.h"
#include "bfd.h"
#include "bfd_event.h"
#include "bfd_daemon.h"
#include "bitops.h"
#ifdef THREAD_DUMP
#include "scheduler.h"
#endif
#include "check_parser.h"


/* local data */
static thread_ref_t bfd_thread;

static void bfd_check_thread(thread_ref_t);

/* Configuration stream handling */
static void
free_bfd_check(checker_t *checker)
{
	bfd_checker_t *bfd_checker = checker->data;

	FREE(bfd_checker);
	FREE(checker);
}

static void
dump_bfd_check(FILE *fp, const checker_t *checker)
{
	const bfd_checker_t *bfd_checker = checker->data;

	conf_write(fp, "   Keepalive method = BFD_CHECK");
	conf_write(fp, "   Name = %s", bfd_checker->bfd->bname);
	conf_write(fp, "   Alpha is %s", checker->alpha ? "ON" : "OFF");

//	conf_write(fp, "   weight = %d", bfd_checker->weight);
}

/* real server on a BFD's list */
void
free_checker_tracked_bfd_list(list_head_t *l)
{
	cref_tracked_bfd_t *tbfd, *tbfd_tmp;

	list_for_each_entry_safe(tbfd, tbfd_tmp, l, e_list)
		FREE(tbfd);
}

void
free_bfds_rs_list(list_head_t *l)
{
	tracking_obj_t *top, *top_tmp;

	list_for_each_entry_safe(top, top_tmp, l, e_list)
		FREE(top);
}

static void
dump_bfds_rs(FILE *fp, const checker_t *checker)
{
	conf_write(fp, "   %s", FMT_RS(checker->rs, checker->vs));
}
void
dump_bfds_rs_list(FILE *fp, const list_head_t *l)
{
	tracking_obj_t *top;

	list_for_each_entry(top, l, e_list)
		dump_bfds_rs(fp, top->obj.checker);
}

static bool
compare_bfd_check(const checker_t *old_c, checker_t *new_c)
{
	const bfd_checker_t *old = old_c->data;
	const bfd_checker_t *new = new_c->data;

	if (strcmp(old->bfd->bname, new->bfd->bname))
		return false;

	return true;
}

static checker_tracked_bfd_t * __attribute__ ((pure))
find_checker_tracked_bfd_by_name(char *name)
{
	checker_tracked_bfd_t *cbfd;

	list_for_each_entry(cbfd, &check_data->track_bfds, e_list) {
		if (!strcmp(cbfd->bname, name))
			return cbfd;
	}

	return NULL;
}

static const checker_funcs_t bfd_checker_funcs = { CHECKER_BFD, free_bfd_check, dump_bfd_check, compare_bfd_check, NULL };

static void
bfd_check_handler(__attribute__((unused)) const vector_t *strvec)
{
	bfd_checker_t *new_bfd_checker;

	PMALLOC(new_bfd_checker);
	INIT_LIST_HEAD(&new_bfd_checker->e_list);

	/* queue new checker */
	queue_checker(&bfd_checker_funcs, NULL, new_bfd_checker, NULL, false);
}

static void
bfd_name_handler(const vector_t *strvec)
{
	checker_tracked_bfd_t *ctbfd;
	cref_tracked_bfd_t *tbfd;
	bfd_checker_t *bfdc;
	bool config_error = true;
	char *name;

	bfdc = current_checker->data;

	if (vector_size(strvec) >= 2)
		name = vector_slot(strvec, 1);

	if (vector_size(strvec) != 2)
		report_config_error(CONFIG_GENERAL_ERROR, "(%s) BFD_CHECK - No or too many names specified"
							  " - skipping checker"
							, FMT_RS(current_checker->rs, current_checker->vs));
	else if (!(ctbfd = find_checker_tracked_bfd_by_name(name)))
		report_config_error(CONFIG_GENERAL_ERROR, "(%s) BFD_CHECK - BFD %s not configured"
							, FMT_RS(current_checker->rs, current_checker->vs)
							, name);
	else if (bfdc->bfd)
		report_config_error(CONFIG_GENERAL_ERROR, "(%s) BFD_CHECK - BFD %s already specified as %s"
							, FMT_RS(current_checker->rs, current_checker->vs)
							, name
							, bfdc->bfd->bname);
	else if (strlen(name) >= BFD_INAME_MAX)
		report_config_error(CONFIG_GENERAL_ERROR, "(%s) BFD_CHECK - BFD name %s too long"
							, FMT_RS(current_checker->rs, current_checker->vs)
							, name);
	else
		config_error = false;

	/* Now check we are not already monitoring it */
	if (!config_error) {
		list_for_each_entry(tbfd, &current_checker->rs->tracked_bfds, e_list) {
			if (ctbfd == tbfd->bfd) {
				report_config_error(CONFIG_GENERAL_ERROR, "(%s) BFD_CHECK - RS already monitoring %s"
									, FMT_RS(current_checker->rs, current_checker->vs)
									, strvec_slot(strvec, 1));
				config_error = true;
				break;
			}
		}
	}

	if (config_error) {
		dequeue_new_checker();
		return;
	}

	bfdc->bfd = ctbfd;
}

static void
bfd_alpha_handler(const vector_t *strvec)
{
	int res;

	if (vector_size(strvec) >= 2) {
		res = check_true_false(strvec_slot(strvec, 1));
		if (res == -1) {
			report_config_error(CONFIG_GENERAL_ERROR, "Invalid alpha parameter %s", strvec_slot(strvec, 1));
			return;
		}
	}
	else
		res = true;

	current_checker->alpha = res;
}

static void
bfd_end_handler(void)
{
	bfd_checker_t *bfdc;
	tracking_obj_t *top;
	cref_tracked_bfd_t *tbfd;

	bfdc = current_checker->data;

	if (!bfdc->bfd) {
		report_config_error(CONFIG_GENERAL_ERROR, "(%s) No name has been specified for BFD_CHECKER - skipping"
							, FMT_RS(current_checker->rs, current_checker->vs));
		dequeue_new_checker();
		return;
	}

	/* Add the bfd to the RS's list */
	PMALLOC(tbfd);
	INIT_LIST_HEAD(&tbfd->e_list);
	tbfd->bfd = bfdc->bfd;
	list_add_tail(&tbfd->e_list, &current_checker->rs->tracked_bfds);

	/* Add the checker to the BFD */
	PMALLOC(top);
	INIT_LIST_HEAD(&top->e_list);
	top->type = TRACK_CHECKER;
	top->obj.checker = current_checker;
	list_add_tail(&top->e_list, &bfdc->bfd->tracking_rs);
}

void
install_bfd_check_keyword(void)
{
	vpp_t check_ptr;

	install_keyword("BFD_CHECK", &bfd_check_handler);
	check_ptr = install_sublevel(VPP &current_checker);
	install_keyword("name", &bfd_name_handler);
	install_keyword("alpha", &bfd_alpha_handler);
	install_level_end_handler(&bfd_end_handler);
	install_sublevel_end(check_ptr);
}

static void
bfd_check_handle_event(bfd_event_t * evt)
{
	struct timeval cur_time;
	struct timeval timer_tmp;
	uint32_t delivery_time;
	checker_tracked_bfd_t *cbfd;
	tracking_obj_t *top;
	checker_t *checker;
	char message[80];
	bool checker_was_up;
	bool rs_was_alive;

	if (__test_bit(LOG_DETAIL_BIT, &debug)) {
		cur_time = timer_now();
		timersub(&cur_time, &evt->sent_time, &timer_tmp);
		delivery_time = timer_long(timer_tmp);
		log_message(LOG_INFO, "Received BFD event: instance %s is in"
			    " state %s (delivered in %" PRIu32 " usec)",
			    evt->iname, BFD_STATE_STR(evt->state), delivery_time);
	}

	list_for_each_entry(cbfd, &check_data->track_bfds, e_list) {
		if (strcmp(cbfd->bname, evt->iname))
			continue;

		/* We can't assume the state of the bfd instance up state
		 * matches the checker up state due to the potential of
		 * alpha state for some checkers and not others */
		list_for_each_entry(top, &cbfd->tracking_rs, e_list) {
			checker = top->obj.checker;
			if ((evt->state == BFD_STATE_UP) == checker->is_up &&
			    checker->has_run)
				continue;

			log_message(LOG_INFO, "BFD check of [%s] RS(%s) is %s"
					    , evt->iname, FMT_RS(checker->rs, checker->vs), evt->state == BFD_STATE_UP ? "UP" : "DOWN");

			checker_was_up = checker->is_up;
			rs_was_alive = checker->rs->alive;
			update_svr_checker_state(evt->state == BFD_STATE_UP ? UP : DOWN, checker);
			if (checker->rs->smtp_alert &&
			    (rs_was_alive != checker->rs->alive || !global_data->no_checker_emails) &&
			    (evt->state == BFD_STATE_UP) != checker_was_up) {
				snprintf(message, sizeof(message), "=> BFD CHECK %s %s on service <=", evt->iname, evt->state == BFD_STATE_UP ? "succeeded" : "failed");
				smtp_alert(SMTP_MSG_RS, checker, NULL, message);
			}
		}
		break;
	}
}

static void
bfd_check_thread(thread_ref_t thread)
{
	bfd_event_t evt;

	if (thread->type == THREAD_READ_ERROR) {
		thread_close_fd(thread);
		return;
	}

	bfd_thread = thread_add_read(master, bfd_check_thread, NULL,
				     thread->u.f.fd, TIMER_NEVER, 0);

	if (thread->type != THREAD_READY_READ_FD)
		return;

	while (read(thread->u.f.fd, &evt, sizeof(bfd_event_t)) != -1)
		bfd_check_handle_event(&evt);
}

void
start_bfd_monitoring(thread_master_t *thread_master)
{
	thread_add_read(thread_master, bfd_check_thread, NULL, bfd_checker_event_pipe[0], TIMER_NEVER, 0);
}

void
checker_bfd_dispatcher_release(void)
{
	thread_cancel(bfd_thread);
	bfd_thread = NULL;
}

#ifdef THREAD_DUMP
void
register_check_bfd_addresses(void)
{
	register_thread_address("bfd_check_thread", bfd_check_thread);
}
#endif
