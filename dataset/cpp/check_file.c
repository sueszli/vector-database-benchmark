/*
 * Soft:        Keepalived is a failover program for the LVS project
 *              <www.linuxvirtualserver.org>. It monitor & manipulate
 *              a loadbalanced server pool using multi-layer checks.
 *
 * Part:        FILE CHECK. Monitor contents if a file
 *
 * Authors:     Quentin Armitage, <quentin@armitage.org.uk>
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
 * Copyright (C) 2020-2020 Alexandre Cassen, <acassen@gmail.com>
 */

#include "config.h"

#include <stdio.h>

#include "check_file.h"
#include "check_data.h"
#include "track_file.h"
#include "parser.h"
#include "logger.h"
#include "check_data.h"
#include "logger.h"
#include "main.h"
#include "check_parser.h"


static tracked_file_monitor_t *current_tfile;

static void
free_file_check(checker_t *checker)
{
	FREE(checker);
}

static void
dump_file_check(FILE *fp, const checker_t *checker)
{
	tracked_file_t *tfp = checker->data;

	conf_write(fp, "   Keepalive method = FILE_CHECK");
	conf_write(fp, "     Tracked file = %s", tfp->fname);
	conf_write(fp, "     Reloaded = %s", tfp->reloaded ? "Yes" : "No");
}

static void
track_file_handler(const vector_t *strvec)
{
	tracked_file_t *vsf;

	vsf = find_tracked_file_by_name(strvec_slot(strvec, 1), &check_data->track_files);
	if (!vsf) {
		report_config_error(CONFIG_GENERAL_ERROR, "track_file %s not found", strvec_slot(strvec, 1));
		return;
	}

	current_tfile->file = vsf;
}

static void
file_check_handler(__attribute__((unused)) const vector_t *strvec)
{
	PMALLOC(current_tfile);
	current_tfile->weight = IPVS_WEIGHT_FAULT;
	INIT_LIST_HEAD(&current_tfile->e_list);
}

static void
track_file_weight_handler(const vector_t *strvec)
{
	int weight;
	bool reverse = false;

	if (vector_size(strvec) < 2) {
		report_config_error(CONFIG_GENERAL_ERROR, "track file weight missing");
		return;
	}

	if (!read_int_strvec(strvec, 1, &weight, -IPVS_WEIGHT_LIMIT, IPVS_WEIGHT_LIMIT, true)) {
		report_config_error(CONFIG_GENERAL_ERROR, "weight for track file must be in "
				 "[%d..%d] inclusive. Ignoring...", -IPVS_WEIGHT_LIMIT, IPVS_WEIGHT_LIMIT);
		return;
	}

	if (vector_size(strvec) >= 3) {
		if (!strcmp(strvec_slot(strvec, 2), "reverse"))
			reverse = true;
		else if (!strcmp(strvec_slot(strvec, 2), "noreverse"))
			reverse = false;
		else {
			report_config_error(CONFIG_GENERAL_ERROR, "unknown track file weight option %s - ignoring",
					 strvec_slot(strvec, 2));
			return;
		}
	}

	current_tfile->weight = weight;
	current_tfile->weight_reverse = reverse;
}

static void
file_end_handler(void)
{
	if (!current_tfile->file) {
		report_config_error(CONFIG_GENERAL_ERROR, "FILE_CHECK has no track_file specified - ignoring");
		free_track_file_monitor(current_tfile);
		return;
	}

	if (current_tfile->weight == IPVS_WEIGHT_FAULT) {
		current_tfile->weight = current_tfile->file->weight;
		current_tfile->weight_reverse = current_tfile->file->weight_reverse;
	}

	list_add_tail(&current_tfile->e_list, &current_rs->track_files);

	current_tfile = NULL;
}

void
install_file_check_keyword(void)
{
	vpp_t check_ptr;

	install_keyword("FILE_CHECK", &file_check_handler);
	check_ptr = install_sublevel(VPP &current_tfile);
	install_keyword("track_file", &track_file_handler);
	install_keyword("weight", &track_file_weight_handler);
	install_level_end_handler(&file_end_handler);
	install_sublevel_end(check_ptr);
}

static const checker_funcs_t file_checker_funcs = { CHECKER_FILE, free_file_check, dump_file_check, NULL, NULL };

void
add_rs_to_track_files(void)
{
	virtual_server_t *vs;
	real_server_t *rs;
	tracked_file_monitor_t *tfl;

	list_for_each_entry(vs, &check_data->vs, e_list) {
		list_for_each_entry(rs, &vs->rs, e_list) {
			list_for_each_entry(tfl, &rs->track_files, e_list) {
				/* queue new checker - we don't have a compare function since we don't
				 * update file checkers that way on a reload. */
				queue_checker(&file_checker_funcs, NULL, tfl->file, NULL, false);
				current_checker->vs = vs;
				current_checker->rs = rs;

				/* There is no concept of the checker running, but we will have
				 * checked the file, so mark it as run. */
				current_checker->has_run = true;

				/* Clear Alpha mode - we know the state of the checker immediately */
				current_checker->alpha = false;

				add_obj_to_track_file(current_checker, tfl, FMT_RS(rs, vs), dump_tracking_rs);
			}
		}
	}
}

void
set_track_file_checkers_down(void)
{
	tracked_file_t *tfl;
	tracking_obj_t *top;
	int status;

	list_for_each_entry(tfl, &check_data->track_files, e_list) {
		if (tfl->last_status) {
			list_for_each_entry(top, &tfl->tracking_obj, e_list) {
				checker_t *checker = top->obj.checker;

				if (!top->weight ||
				    (int64_t)tfl->last_status * top->weight * top->weight_multiplier <= IPVS_WEIGHT_FAULT) {
					if (reload) {
						/* This is pretty horrible. At some stage this should
						 * be tidied up so that it works without having to
						 * fudge the values to make update_track_file_status()
						 * work for us. */
						status = tfl->last_status;
						tfl->last_status = 0;
						process_update_checker_track_file_status(tfl, !status != (top->weight_multiplier == 1) ? IPVS_WEIGHT_FAULT: 0, top);
						tfl->last_status = status;
					} else
						checker->is_up = false;
				} else if ((int64_t)tfl->last_status * top->weight * top->weight_multiplier <= IPVS_WEIGHT_FAULT && !reload)
					checker->is_up = false;
			}
		}
	}
}

void
set_track_file_weights(void)
{
	tracked_file_t *tfl;
	tracking_obj_t *top;

	list_for_each_entry(tfl, &check_data->track_files, e_list) {
		if (tfl->last_status) {
			list_for_each_entry(top, &tfl->tracking_obj, e_list) {
				checker_t *checker = top->obj.checker;

				if (top->weight)
					checker->cur_weight = (int64_t)tfl->last_status * top->weight * top->weight_multiplier;
			}
		}
	}
}

#ifdef THREAD_DUMP
void
register_check_file_addresses(void)
{
	register_track_file_inotify_addresses();
}
#endif
