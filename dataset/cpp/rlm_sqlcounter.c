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
 * $Id: 5595ab14f85a15583d49c4b958cccbfd2140ff15 $
 * @file rlm_sqlcounter.c
 * @brief Tracks data usage and other counters using SQL.
 *
 * @copyright 2001,2006 The FreeRADIUS server project
 * @copyright 2001 Alan DeKok (aland@freeradius.org)
 */
RCSID("$Id: 5595ab14f85a15583d49c4b958cccbfd2140ff15 $")

#define LOG_PREFIX "sqlcounter"

#include <freeradius-devel/server/base.h>
#include <freeradius-devel/server/module_rlm.h>
#include <freeradius-devel/util/debug.h>

#include <ctype.h>

#define MAX_QUERY_LEN 1024

/*
 *	Note: When your counter spans more than 1 period (ie 3 months
 *	or 2 weeks), this module probably does NOT do what you want! It
 *	calculates the range of dates to count across by first calculating
 *	the End of the Current period and then subtracting the number of
 *	periods you specify from that to determine the beginning of the
 *	range.
 *
 *	For example, if you specify a 3 month counter and today is June 15th,
 *	the end of the current period is June 30. Subtracting 3 months from
 *	that gives April 1st. So, the counter will sum radacct entries from
 *	April 1st to June 30. Then, next month, it will sum entries from
 *	May 1st to July 31st.
 *
 *	To fix this behavior, we need to add some way of storing the Next
 *	Reset Time.
 */

/*
 *	Define a structure for our module configuration.
 *
 *	These variables do not need to be in a structure, but it's
 *	a lot cleaner to do so, and a pointer to the structure can
 *	be used as the instance handle.
 */
typedef struct {
	tmpl_t	*start_attr;		//!< &control.${.:instance}-Start
	tmpl_t	*end_attr;		//!< &control.${.:instance}-End

	tmpl_t	*paircmp_attr;		//!< Daily-Session-Time.
	tmpl_t	*limit_attr;  		//!< Max-Daily-Session.
	tmpl_t	*reply_attr;  		//!< Session-Timeout.
	tmpl_t	*key;  			//!< User-Name

	char const	*sqlmod_inst;	//!< Instance of SQL module to use, usually just 'sql'.
	char const	*query;		//!< SQL query to retrieve current session time.
	char const	*reset;  	//!< Daily, weekly, monthly, never or user defined.

	fr_time_t	reset_time;
	fr_time_t	last_reset;
} rlm_sqlcounter_t;

static const CONF_PARSER module_config[] = {
	{ FR_CONF_OFFSET("sql_module_instance", FR_TYPE_STRING | FR_TYPE_REQUIRED, rlm_sqlcounter_t, sqlmod_inst) },


	{ FR_CONF_OFFSET("query", FR_TYPE_STRING | FR_TYPE_XLAT | FR_TYPE_REQUIRED, rlm_sqlcounter_t, query) },
	{ FR_CONF_OFFSET("reset", FR_TYPE_STRING | FR_TYPE_REQUIRED, rlm_sqlcounter_t, reset) },

	{ FR_CONF_OFFSET("key", FR_TYPE_TMPL | FR_TYPE_NOT_EMPTY, rlm_sqlcounter_t, key), .dflt = "%{%{Stripped-User-Name}:-%{User-Name}}", .quote = T_DOUBLE_QUOTED_STRING },

	{ FR_CONF_OFFSET("reset_period_start_name", FR_TYPE_TMPL | FR_TYPE_ATTRIBUTE, rlm_sqlcounter_t, start_attr),
	  .dflt = "&control.${.:instance}-Start" },
	{ FR_CONF_OFFSET("reset_period_end_name", FR_TYPE_TMPL | FR_TYPE_ATTRIBUTE, rlm_sqlcounter_t, end_attr),
	  .dflt = "&control.${.:instance}-End" },

	/* Just used to register a paircmp against */
	{ FR_CONF_OFFSET("counter_name", FR_TYPE_TMPL | FR_TYPE_ATTRIBUTE | FR_TYPE_REQUIRED, rlm_sqlcounter_t, paircmp_attr) },
	{ FR_CONF_OFFSET("check_name", FR_TYPE_TMPL | FR_TYPE_ATTRIBUTE | FR_TYPE_REQUIRED, rlm_sqlcounter_t, limit_attr) },

	/* Attribute to write remaining session to */
	{ FR_CONF_OFFSET("reply_name", FR_TYPE_TMPL | FR_TYPE_ATTRIBUTE, rlm_sqlcounter_t, reply_attr) },
	CONF_PARSER_TERMINATOR
};

static fr_dict_t const *dict_freeradius;
static fr_dict_t const *dict_radius;

extern fr_dict_autoload_t rlm_sqlcounter_dict[];
fr_dict_autoload_t rlm_sqlcounter_dict[] = {
	{ .out = &dict_freeradius, .proto = "freeradius" },
	{ .out = &dict_radius, .proto = "radius" },
	{ NULL }
};

static fr_dict_attr_t const *attr_reply_message;
static fr_dict_attr_t const *attr_session_timeout;

extern fr_dict_attr_autoload_t rlm_sqlcounter_dict_attr[];
fr_dict_attr_autoload_t rlm_sqlcounter_dict_attr[] = {
	{ .out = &attr_reply_message, .name = "Reply-Message", .type = FR_TYPE_STRING, .dict = &dict_radius },
	{ .out = &attr_session_timeout, .name = "Session-Timeout", .type = FR_TYPE_UINT32, .dict = &dict_radius },
	{ NULL }
};

static int find_next_reset(rlm_sqlcounter_t *inst, fr_time_t now)
{
	int		ret = 0;
	size_t		len;
	unsigned int	num = 1;
	char		last = '\0';
	struct tm	*tm, s_tm;
	time_t		time_s = fr_time_to_sec(now);

	tm = localtime_r(&time_s, &s_tm);
	tm->tm_sec = tm->tm_min = 0;

	fr_assert(inst->reset != NULL);

	if (isdigit((uint8_t) inst->reset[0])){
		len = strlen(inst->reset);
		if (len == 0)
			return -1;
		last = inst->reset[len - 1];
		if (!isalpha((uint8_t) last))
			last = 'd';
		num = atoi(inst->reset);
		DEBUG("num=%d, last=%c",num,last);
	}
	if (strcmp(inst->reset, "hourly") == 0 || last == 'h') {
		/*
		 *  Round up to the next nearest hour.
		 */
		tm->tm_hour += num;
		inst->reset_time = fr_time_from_sec(mktime(tm));
	} else if (strcmp(inst->reset, "daily") == 0 || last == 'd') {
		/*
		 *  Round up to the next nearest day.
		 */
		tm->tm_hour = 0;
		tm->tm_mday += num;
		inst->reset_time = fr_time_from_sec(mktime(tm));
	} else if (strcmp(inst->reset, "weekly") == 0 || last == 'w') {
		/*
		 *  Round up to the next nearest week.
		 */
		tm->tm_hour = 0;
		tm->tm_mday += (7 - tm->tm_wday) +(7*(num-1));
		inst->reset_time = fr_time_from_sec(mktime(tm));
	} else if (strcmp(inst->reset, "monthly") == 0 || last == 'm') {
		tm->tm_hour = 0;
		tm->tm_mday = 1;
		tm->tm_mon += num;
		inst->reset_time = fr_time_from_sec(mktime(tm));
	} else if (strcmp(inst->reset, "never") == 0) {
		inst->reset_time = fr_time_wrap(0);
	} else {
		return -1;
	}

	DEBUG2("Current Time: %pV, Next reset %pV", fr_box_time(now), fr_box_time(inst->reset_time));

	return ret;
}


/*  I don't believe that this routine handles Daylight Saving Time adjustments
    properly.  Any suggestions?
*/
static int find_prev_reset(rlm_sqlcounter_t *inst, fr_time_t now)
{
	int		ret = 0;
	size_t		len;
	unsigned	int num = 1;
	char		last = '\0';
	struct		tm *tm, s_tm;
	time_t		time_s = fr_time_to_sec(now);

	tm = localtime_r(&time_s, &s_tm);
	tm->tm_sec = tm->tm_min = 0;

	fr_assert(inst->reset != NULL);

	if (isdigit((uint8_t) inst->reset[0])){
		len = strlen(inst->reset);
		if (len == 0)
			return -1;
		last = inst->reset[len - 1];
		if (!isalpha((uint8_t) last))
			last = 'd';
		num = atoi(inst->reset);
		DEBUG("num=%d, last=%c", num, last);
	}
	if (strcmp(inst->reset, "hourly") == 0 || last == 'h') {
		/*
		 *  Round down to the prev nearest hour.
		 */
		tm->tm_hour -= num - 1;
		inst->last_reset = fr_time_from_sec(mktime(tm));
	} else if (strcmp(inst->reset, "daily") == 0 || last == 'd') {
		/*
		 *  Round down to the prev nearest day.
		 */
		tm->tm_hour = 0;
		tm->tm_mday -= num - 1;
		inst->last_reset = fr_time_from_sec(mktime(tm));
	} else if (strcmp(inst->reset, "weekly") == 0 || last == 'w') {
		/*
		 *  Round down to the prev nearest week.
		 */
		tm->tm_hour = 0;
		tm->tm_mday -= tm->tm_wday +(7*(num-1));
		inst->last_reset = fr_time_from_sec(mktime(tm));
	} else if (strcmp(inst->reset, "monthly") == 0 || last == 'm') {
		tm->tm_hour = 0;
		tm->tm_mday = 1;
		tm->tm_mon -= num - 1;
		inst->last_reset = fr_time_from_sec(mktime(tm));
	} else if (strcmp(inst->reset, "never") == 0) {
		inst->reset_time = fr_time_wrap(0);
	} else {
		return -1;
	}

	DEBUG2("Current Time: %pV, Prev reset %pV", fr_box_time(now), fr_box_time(inst->last_reset));

	return ret;
}


/*
 *	Find the named user in this modules database.  Create the set
 *	of attribute-value pairs to check and reply with for this user
 *	from the database. The authentication code only needs to check
 *	the password, the rest is done here.
 */
static unlang_action_t CC_HINT(nonnull) mod_authorize(rlm_rcode_t *p_result, module_ctx_t const *mctx, request_t *request)
{
	rlm_sqlcounter_t	*inst = talloc_get_type_abort(mctx->inst->data, rlm_sqlcounter_t);
	uint64_t		counter, res;
	fr_pair_t		*limit, *vp;
	fr_pair_t		*reply_item;
	char			msg[128];
	int			ret;
	size_t len;
	char *expanded = NULL;
	char query[MAX_QUERY_LEN];

	/*
	 *	Before doing anything else, see if we have to reset
	 *	the counters.
	 */
	if (fr_time_eq(inst->reset_time, fr_time_wrap(0)) &&
	    (fr_time_lteq(inst->reset_time, request->packet->timestamp))) {
		/*
		 *	Re-set the next time and prev_time for this counters range
		 */
		inst->last_reset = inst->reset_time;
		find_next_reset(inst, request->packet->timestamp);
	}

	if (tmpl_find_vp(&limit, request, inst->limit_attr) < 0) {
		RWDEBUG2("Couldn't find %s, doing nothing...", inst->limit_attr->name);
		RETURN_MODULE_NOOP;
	}

	if (tmpl_find_or_add_vp(&vp, request, inst->start_attr) < 0) {
		RWDEBUG2("Couldn't find %s, doing nothing...", inst->start_attr->name);
		RETURN_MODULE_NOOP;
	}
	vp->vp_uint64 = fr_time_to_sec(inst->last_reset);

	if (tmpl_find_or_add_vp(&vp, request, inst->end_attr) < 0) {
		RWDEBUG2("Couldn't find %s, doing nothing...", inst->end_attr->name);
		RETURN_MODULE_NOOP;
	}
	vp->vp_uint64 = fr_time_to_sec(inst->reset_time);

	/* Then combine that with the name of the module were using to do the query */
	len = snprintf(query, sizeof(query), "%%{%s:%s}", inst->sqlmod_inst, inst->query);
	if (len >= (sizeof(query) - 1)) {
		REDEBUG("Insufficient query buffer space");

		RETURN_MODULE_FAIL;
	}

	/* Finally, xlat resulting SQL query */
	if (xlat_aeval(request, &expanded, request, query, NULL, NULL) < 0) {
		RETURN_MODULE_FAIL;
	}

	if (sscanf(expanded, "%" PRIu64, &counter) != 1) {
		RDEBUG2("No integer found in result string \"%s\".  May be first session, setting counter to 0",
			expanded);
		counter = 0;
	}

	talloc_free(expanded);

	/*
	 *	Check if check item > counter
	 */
	if (limit->vp_uint64 <= counter) {
		/* User is denied access, send back a reply message */
		snprintf(msg, sizeof(msg), "Your maximum %s usage time has been reached", inst->reset);

		MEM(pair_update_reply(&vp, attr_reply_message) >= 0);
		fr_pair_value_strdup(vp, msg, false);

		REDEBUG2("Maximum %s usage time reached", inst->reset);
		REDEBUG2("Rejecting user, %s value (%" PRIu64 ") is less than counter value (%" PRIu64 ")",
			 inst->limit_attr->name, limit->vp_uint64, counter);

		RETURN_MODULE_REJECT;
	}

	res = limit->vp_uint64 - counter;
	RDEBUG2("Allowing user, %s value (%" PRIu64 ") is greater than counter value (%" PRIu64 ")",
		inst->limit_attr->name, limit->vp_uint64, counter);

	/*
	 *	Add the counter to the control list
	 */
	MEM(pair_update_control(&vp, tmpl_attr_tail_da(inst->paircmp_attr)) >= 0);
	vp->vp_uint64 = counter;

	/*
	 *	We are assuming that simultaneous-use=1. But
	 *	even if that does not happen then our user
	 *	could login at max for 2*max-usage-time Is
	 *	that acceptable?
	 */
	if (inst->reply_attr) {
		/*
		 *	If we are near a reset then add the next
		 *	limit, so that the user will not need to login
		 *	again.  Do this only for Session-Timeout.
		 */
		if ((tmpl_attr_tail_da(inst->reply_attr) == attr_session_timeout) &&
		    fr_time_gt(inst->reset_time, fr_time_wrap(0)) &&
		    ((int64_t)res >= fr_time_delta_to_sec(fr_time_sub(inst->reset_time, request->packet->timestamp)))) {
			fr_time_delta_t to_reset = fr_time_sub(inst->reset_time, request->packet->timestamp);

			RDEBUG2("Time remaining (%pV) is greater than time to reset (%" PRIu64 "s).  "
				"Adding %pV to reply value",
				fr_box_time_delta(to_reset), res, fr_box_time_delta(to_reset));
			res = fr_time_delta_to_sec(to_reset) + limit->vp_uint64;

			/*
			 *	Limit the reply attribute to the minimum of the existing value, or this new one.
			 *
			 *	Duplicate code because Session-Timeout is uint32, not uint64
			 */
			ret = tmpl_find_or_add_vp(&reply_item, request, inst->reply_attr);
			switch (ret) {
			case 1:		/* new */
				break;

			case 0:		/* found */
				if (reply_item->vp_uint32 <= res) {
					RDEBUG2("Leaving existing %s value of %u", inst->reply_attr->name,
						reply_item->vp_uint32);
					RETURN_MODULE_OK;
				}
				break;

			case -1:	/* alloc failed */
				REDEBUG("Error allocating attribute %s", inst->reply_attr->name);
				RETURN_MODULE_FAIL;

			default:	/* request or list unavailable */
				RDEBUG2("List or request context not available for %s, skipping...", inst->reply_attr->name);
				RETURN_MODULE_OK;
			}

			if (res > UINT32_MAX) res = UINT32_MAX;
			reply_item->vp_uint32 = res;

		} else {
			/*
			 *	Limit the reply attribute to the minimum of the existing value, or this new one.
			 */
			ret = tmpl_find_or_add_vp(&reply_item, request, inst->reply_attr);
			switch (ret) {
			case 1:		/* new */
				break;

			case 0:		/* found */
				if (reply_item->vp_uint64 <= res) {
					RDEBUG2("Leaving existing %s value of %" PRIu64, inst->reply_attr->name,
						reply_item->vp_uint64);
					RETURN_MODULE_OK;
				}
				break;

			case -1:	/* alloc failed */
				REDEBUG("Error allocating attribute %s", inst->reply_attr->name);
				RETURN_MODULE_FAIL;

			default:	/* request or list unavailable */
				RDEBUG2("List or request context not available for %s, skipping...", inst->reply_attr->name);
				RETURN_MODULE_OK;
			}
			reply_item->vp_uint64 = res;
		}
		RDEBUG2("&%pP", reply_item);

		RETURN_MODULE_UPDATED;
	}

	RETURN_MODULE_OK;
}

/*
 *	Do any per-module initialization that is separate to each
 *	configured instance of the module.  e.g. set up connections
 *	to external databases, read configuration files, set up
 *	dictionary entries, etc.
 *
 *	If configuration information is given in the config section
 *	that must be referenced in later calls, store a handle to it
 *	in *instance otherwise put a null pointer there.
 */
static int mod_instantiate(module_inst_ctx_t const *mctx)
{
	rlm_sqlcounter_t	*inst = talloc_get_type_abort(mctx->inst->data, rlm_sqlcounter_t);
	CONF_SECTION    	*conf = mctx->inst->conf;

	fr_assert(inst->query && *inst->query);

	inst->reset_time = fr_time_wrap(0);

	if (find_next_reset(inst, fr_time()) == -1) {
		cf_log_err(conf, "Invalid reset '%s'", inst->reset);
		return -1;
	}

	/*
	 *  Discover the beginning of the current time period.
	 */
	inst->last_reset = fr_time_wrap(0);

	if (find_prev_reset(inst, fr_time()) < 0) {
		cf_log_err(conf, "Invalid reset '%s'", inst->reset);
		return -1;
	}

	return 0;
}

static int mod_bootstrap(module_inst_ctx_t const *mctx)
{
	rlm_sqlcounter_t	*inst = talloc_get_type_abort(mctx->inst->data, rlm_sqlcounter_t);
	CONF_SECTION    	*conf = mctx->inst->conf;
	fr_dict_attr_flags_t	flags;

	/*
	 *	Create a new attribute for the counter.
	 */
	fr_assert(inst->paircmp_attr);
	fr_assert(inst->limit_attr);

	memset(&flags, 0, sizeof(flags));
	if (tmpl_attr_tail_unresolved_add(fr_dict_unconst(dict_freeradius), inst->start_attr, FR_TYPE_UINT64, &flags) < 0) {
		cf_log_perr(conf, "Failed defining reset_period_start attribute");
		return -1;
	}

	if (tmpl_attr_tail_unresolved_add(fr_dict_unconst(dict_freeradius), inst->end_attr, FR_TYPE_UINT64, &flags) < 0) {
		cf_log_perr(conf, "Failed defining reset_end_start attribute");
		return -1;
	}

	if (tmpl_attr_tail_unresolved_add(fr_dict_unconst(dict_freeradius), inst->paircmp_attr, FR_TYPE_UINT64, &flags) < 0) {
		cf_log_perr(conf, "Failed defining counter attribute");
		return -1;
	}

	if (tmpl_attr_tail_unresolved_add(fr_dict_unconst(dict_freeradius), inst->limit_attr, FR_TYPE_UINT64, &flags) < 0) {
		cf_log_perr(conf, "Failed defining check attribute");
		return -1;
	}

	if (tmpl_attr_tail_da(inst->paircmp_attr)->type != FR_TYPE_UINT64) {
		cf_log_err(conf, "Counter attribute %s MUST be uint64", tmpl_attr_tail_da(inst->paircmp_attr)->name);
		return -1;
	}

	if (tmpl_attr_tail_da(inst->limit_attr)->type != FR_TYPE_UINT64) {
		cf_log_err(conf, "Check attribute %s MUST be uint64", tmpl_attr_tail_da(inst->limit_attr)->name);
		return -1;
	}

	return 0;
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
extern module_rlm_t rlm_sqlcounter;
module_rlm_t rlm_sqlcounter = {
	.common = {
		.magic		= MODULE_MAGIC_INIT,
		.name		= "sqlcounter",
		.type		= MODULE_TYPE_THREAD_SAFE,
		.inst_size	= sizeof(rlm_sqlcounter_t),
		.config		= module_config,
		.bootstrap	= mod_bootstrap,
		.instantiate	= mod_instantiate,
	},
	.method_names = (module_method_name_t[]){
		{ .name1 = CF_IDENT_ANY,	.name2 = CF_IDENT_ANY,		.method = mod_authorize},
		MODULE_NAME_TERMINATOR
	}
};
