/*
 * Copyright (C) 2009-2011 Nippon Telegraph and Telephone Corporation.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License version
 * 2 as published by the Free Software Foundation.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/time.h>
#include <fcntl.h>

#include "sheepdog_proto.h"
#include "sheep.h"
#include "dog.h"
#include "util.h"
#include "sockfd_cache.h"

#define EPOLL_SIZE 4096

static const char program_name[] = "dog";
struct node_id sd_nid = {
	/* default sdhost is "127.0.0.1" */
	.addr = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127, 0, 0, 1 },
	.port = SD_LISTEN_PORT,
};
bool highlight = true;
bool raw_output;
bool verbose;
bool elapsed_time;

static const struct sd_option dog_options[] = {

	/* common options for all dog commands */
	{'a', "address", true, "specify the daemon address (default: localhost)"},
	{'p', "port", true, "specify the daemon port"},
	{'r', "raw", false, "raw output mode: omit headers, separate fields with\n"
	 "                          single spaces and print all sizes in decimal bytes"},
	{'v', "verbose", false, "print more information than default"},
	{'h', "help", false, "display this help and exit"},
	{'T', "time", false, "show elapsed time"},

	{ 0, NULL, false, NULL },
};

static void usage(const struct command *commands, int status);

uint32_t sd_epoch;

int sd_nodes_nr;
struct rb_root sd_vroot = RB_ROOT;
struct rb_root sd_nroot = RB_ROOT;
int sd_zones_nr;
/* a number of zones never exceeds a number of nodes */
static uint32_t sd_zones[SD_MAX_NODES];

int update_node_list(int max_nodes)
{
	int ret;
	unsigned int size;
	struct sd_node *buf = NULL;
	struct sd_node *ent;
	struct sd_req hdr;
	struct sd_rsp *rsp = (struct sd_rsp *)&hdr;
	struct epoch_log *logs = NULL;
	int log_length;

	size = sizeof(*ent) * max_nodes;
	buf = xzalloc(size);
	sd_init_req(&hdr, SD_OP_GET_NODE_LIST);

	hdr.data_length = size;

	ret = dog_exec_req(&sd_nid, &hdr, buf);
	if (ret < 0)
		goto out;

	if (rsp->result != SD_RES_SUCCESS) {
		sd_err("Failed to update node list: %s",
		       sd_strerror(rsp->result));
		ret = -1;
		goto out;
	}

	size = rsp->data_length;
	sd_nodes_nr = size / sizeof(*ent);
	if (sd_nodes_nr == 0) {
		sd_err("There are no active sheep daemons");
		exit(EXIT_FAILURE);
	}

	/* FIXME */
	if (sd_nodes_nr > max_nodes) {
		ret = -1;
		goto out;
	}

	for (int i = 0; i < sd_nodes_nr; i++) {
		struct sd_node *n = xmalloc(sizeof(*n));
		int j;

		*n = buf[i];
		rb_insert(&sd_nroot, n, rb, node_cmp);

		for (j = 0; j < sd_zones_nr; j++) {
			if (sd_zones[j] == n->zone)
				break;
		}

		if (j == sd_zones_nr)
			sd_zones[sd_zones_nr++] = n->zone;
	}
	/* check whether cluster use diskmode */
	log_length = sizeof(struct epoch_log);
	logs = xmalloc(log_length);
	memset(logs, 0, log_length);

	sd_init_req(&hdr, SD_OP_STAT_CLUSTER);
	hdr.data_length = log_length;

	ret = dog_exec_req(&sd_nid, &hdr, logs);
	if (ret < 0)
		goto out;

	if (logs->flags & SD_CLUSTER_FLAG_DISKMODE)
		disks_to_vnodes(&sd_nroot, &sd_vroot);
	else
		nodes_to_vnodes(&sd_nroot, &sd_vroot);

	sd_epoch = hdr.epoch;
out:
	if (buf)
		free(buf);
	free(logs);
	return ret;
}

static int (*command_parser)(int, const char *);
static int (*command_fn)(int, char **);
static const char *command_opts;
static const char *command_arg;
static const char *command_desc;
static const struct sd_option *command_options;

static const struct sd_option *find_opt(int ch)
{
	const struct sd_option *opt;

	/* search for common options */
	sd_for_each_option(opt, dog_options) {
		if (opt->ch == ch)
			return opt;
	}

	/* search for self options */
	if (command_options) {
		sd_for_each_option(opt, command_options) {
			if (opt->ch == ch)
				return opt;
		}
	}

	sd_err("Internal error");
	exit(EXIT_SYSFAIL);
}

static void init_commands(const struct command **commands)
{
	static struct command *cmds;
	struct command command_list[] = {
		vdi_command,
		node_command,
		cluster_command,
#ifdef HAVE_TRACE
		trace_command,
#endif
#ifdef HAVE_NFS
		nfs_command,
#endif
		upgrade_command,
		benchmark_command,
		{NULL,}
	};

	if (!cmds) {
		cmds = (struct command *)xmalloc(sizeof(command_list));
		memcpy(cmds, command_list, sizeof(command_list));
	}

	*commands = cmds;
	return;
}

static const struct subcommand *find_subcmd(const char *cmd, const char *subcmd)
{
	int i, j;
	const struct command *commands;
	const struct subcommand *sub;

	init_commands(&commands);

	for (i = 0; commands[i].name; i++) {
		if (!strcmp(commands[i].name, cmd)) {
			sub = commands[i].sub;
			for (j = 0; sub[j].name; j++) {
				if (!strcmp(sub[j].name, subcmd))
					return &sub[j];
			}
		}
	}

	return NULL;
}

static unsigned long setup_commands(const struct command *commands,
				    const char *cmd, const char *subcmd)
{
	int i;
	bool found = false;
	const struct subcommand *s;
	unsigned long flags = 0;

	for (i = 0; commands[i].name; i++) {
		if (!strcmp(commands[i].name, cmd)) {
			found = true;
			if (commands[i].parser)
				command_parser = commands[i].parser;
			break;
		}
	}

	if (!found) {
		if (cmd && strcmp(cmd, "help") && strcmp(cmd, "--help") &&
		    strcmp(cmd, "-h")) {
			sd_err("Invalid command '%s'", cmd);
			usage(commands, EXIT_USAGE);
		}
		usage(commands, 0);
	}

	for (s = commands[i].sub; subcmd && s->name; s++) {
		if (!strcmp(s->name, subcmd)) {
			command_fn = s->fn;
			command_opts = s->opts;
			command_arg = s->arg;
			command_desc = s->desc;
			command_options = s->options;
			flags = s->flags;
			break;
		}
	}

	if (!command_fn) {
		if (subcmd && strcmp(subcmd, "help") &&
		    strcmp(subcmd, "--help") && strcmp(subcmd, "-h"))
			sd_err("Invalid command '%s %s'", cmd, subcmd);
		sd_err("Available %s commands:", cmd);
		for (s = commands[i].sub; s->name; s++)
			sd_err("  %s %s", cmd, s->name);
		exit(EXIT_USAGE);
	}

	return flags;
}

static void usage(const struct command *commands, int status)
{
	int i;
	const struct subcommand *s;
	char name[64];

	if (status)
		sd_err("Try '%s --help' for more information.", program_name);
	else {
		printf("Sheepdog administrator utility (version %s)\n",
		       PACKAGE_VERSION);
		printf("Usage: %s <command> <subcommand> [options]\n", program_name);
		printf("\nAvailable commands:\n");
		for (i = 0; commands[i].name; i++) {
			for (s = commands[i].sub; s->name; s++) {
				snprintf(name, sizeof(name), "%s %s",
					 commands[i].name, s->name);
				printf("  %-24s%s\n", name, s->desc);
			}
		}
		printf("\n");
		printf("For more information, run "
		       "'%s <command> <subcommand> --help'.\n", program_name);
	}
	exit(status);
}

void subcommand_usage(char *cmd, char *subcmd, int status)
{
	int i, n, len = strlen(command_opts);
	const struct sd_option *sd_opt;
	const struct subcommand *sub, *subsub;
	char name[64];

	printf("Usage: %s %s %s", program_name, cmd, subcmd);

	if (0 <= subcmd_depth) {
		for (i = 0; i < subcmd_depth + 1; i++)
			printf(" %s", subcmd_stack[i]->name);

		subsub = subcmd_stack[i - 1]->sub;
	} else {
		sub = find_subcmd(cmd, subcmd);
		subsub = sub->sub;
	}

	if (subsub) {
		n = 0;
		while (subsub[n].name)
			n++;
		if (n == 1)
			printf(" %s", subsub[0].name);
		else if (n > 1) {
			printf(" {%s", subsub[0].name);
			for (i = 1; i < n; i++)
				printf("|%s", subsub[i].name);
			printf("}");
		}
	}

	for (i = 0; i < len; i++) {
		sd_opt = find_opt(command_opts[i]);
		if (sd_opt->has_arg)
			printf(" [-%c %s]", sd_opt->ch, sd_opt->name);
		else
			printf(" [-%c]", sd_opt->ch);
	}
	if (command_arg)
		printf(" %s", command_arg);

	printf("\n");
	if (subsub) {
		printf("Available subcommands:\n");
		for (i = 0; subsub[i].name; i++)
			printf("  %-24s%s\n", subsub[i].name, subsub[i].desc);

	}

	printf("Options:\n");
	for (i = 0; i < len; i++) {
		sd_opt = find_opt(command_opts[i]);
		snprintf(name, sizeof(name), "-%c, --%s",
			 sd_opt->ch, sd_opt->name);
		printf("  %-24s%s\n", name, sd_opt->desc);
	}

	exit(status);
}

static const struct sd_option *build_sd_options(const char *opts)
{
	static struct sd_option sd_opts[256], *p;
	int i, len = strlen(opts);

	p = sd_opts;
	for (i = 0; i < len; i++)
		*p++ = *find_opt(opts[i]);
	memset(p, 0, sizeof(struct sd_option));

	return sd_opts;
}

static void crash_handler(int signo, siginfo_t *info, void *context)
{
	sd_err("dog exits unexpectedly (%s).", strsignal(signo));

	sd_backtrace();

	/*
	 * OOM raises SIGABRT in xmalloc but the administrator expects
	 * that dog exits with EXIT_SYSFAIL.  We have to give up
	 * dumping a core file in this case.
	 */
	if (signo == SIGABRT)
		exit(EXIT_SYSFAIL);

	reraise_crash_signal(signo, EXIT_SYSFAIL);
}

static size_t get_nr_nodes(void)
{
	return sd_nodes_nr;
}

static void log_dog_operation(int argc, char **argv)
{
	int length = 0, printed = 0;
	char *msg;
	const char *custom_log_path;

	if (!getenv("SHEEPDOG_DOG_LOG"))
		/* don't log operation of dog */
		return;

	for (int i = 0; i < argc; i++)
		length += 1 + strlen(argv[i]); /* 1 is for space */

	length++; /* 1 is for '\0' */
	msg = xcalloc(length, sizeof(char));

	for (int i = 0; i < argc; i++)
		printed += snprintf(msg + printed, length - printed,
				    " %s", argv[i]);

	custom_log_path = getenv("SHEEPDOG_DOG_LOG_PATH");
	if (custom_log_path) {
		struct timeval tv;
		struct tm tm;
		char time_str[256];
		int fd;

		fd = open(custom_log_path, O_WRONLY | O_APPEND | O_CREAT,
			  S_IRUSR | S_IWUSR);
		if (fd < 0) {
			fprintf(stderr, "error at opening log file of dog"
				"(%s): %m\n", custom_log_path);
			goto out;
		}

		gettimeofday(&tv, NULL);
		localtime_r(&tv.tv_sec, &tm);
		strftime(time_str, sizeof(time_str),
			 "%Y %b %2d %H:%M:%S ", &tm);

		dprintf(fd, "%s: %s\n", time_str, msg);
		close(fd);
	} else {
		/* if the path is not specified, we use standard syslog */

		openlog("sheepdog admin operation", LOG_PID, LOG_USER);
		syslog(LOG_INFO, "%s\n", msg);
		closelog();
	}

out:
	free(msg);
}

int main(int argc, char **argv)
{
	int ch, longindex, ret;
	unsigned long flags;
	struct option *long_options;
	const struct command *commands;
	const char *short_options;
	char *p, *env;
	const struct sd_option *sd_opts;
	uint8_t sdhost[16];
	int sdport;
	struct timespec start, end;

	start = get_time_tick();

	log_dog_operation(argc, argv);

	install_crash_handler(crash_handler);

	init_commands(&commands);

	if (argc < 2)
		usage(commands, 0);

	flags = setup_commands(commands, argv[1], argv[2]);

	optind = 3;

	sd_opts = build_sd_options(command_opts);
	long_options = build_long_options(sd_opts);
	short_options = build_short_options(sd_opts);

	env = getenv("SHEEPDOG_DOG_ADDR");
	if (env) {
		if (!str_to_addr(env, sdhost)) {
			sd_err("Invalid ip address %s", env);
			return EXIT_FAILURE;
		}
		memcpy(sd_nid.addr, sdhost, sizeof(sdhost));
	}

	env = getenv("SHEEPDOG_DOG_PORT");
	if (env) {
		sdport = strtol(env, &p, 10);
		if (env == p || sdport < 1 || sdport > UINT16_MAX
		    || !is_numeric(env)) {
			sd_err("Invalid port number '%s'", env);
			exit(EXIT_USAGE);
		}
		sd_nid.port = sdport;
	}

	while ((ch = getopt_long(argc, argv, short_options, long_options,
				&longindex)) >= 0) {

		switch (ch) {
		case 'a':
			if (!str_to_addr(optarg, sdhost)) {
				sd_err("Invalid ip address %s", optarg);
				return EXIT_FAILURE;
			}
			memcpy(sd_nid.addr, sdhost, sizeof(sdhost));
			break;
		case 'p':
			sdport = strtol(optarg, &p, 10);
			if (optarg == p || sdport < 1 || sdport > UINT16_MAX
					|| !is_numeric(optarg)) {
				sd_err("Invalid port number '%s'", optarg);
				exit(EXIT_USAGE);
			}
			sd_nid.port = sdport;
			break;
		case 'r':
			raw_output = true;
			break;
		case 'v':
			verbose = true;
			break;
		case 'h':
			subcommand_usage(argv[1], argv[2], EXIT_SUCCESS);
			break;
		case 'T':
			elapsed_time = true;
			break;
		case '?':
			usage(commands, EXIT_USAGE);
			break;
		default:
			if (command_parser)
				command_parser(ch, optarg);
			else
				usage(commands, EXIT_USAGE);
			break;
		}
	}

	if (sd_inode_actor_init(dog_bnode_writer, dog_bnode_reader) < 0)
		exit(EXIT_SYSFAIL);

	if (!is_stdout_console() || raw_output)
		highlight = false;

	if (flags & CMD_NEED_NODELIST) {
		ret = update_node_list(SD_MAX_NODES);
		if (ret < 0) {
			sd_err("Failed to get node list");
			exit(EXIT_SYSFAIL);
		}
	}

	if (flags & CMD_NEED_ARG && argc == optind)
		subcommand_usage(argv[1], argv[2], EXIT_USAGE);

	if (init_event(EPOLL_SIZE) < 0)
		exit(EXIT_SYSFAIL);

	if (wq_trace_init() < 0)
		exit(EXIT_SYSFAIL);

	if (init_work_queue(get_nr_nodes) != 0) {
		sd_err("Failed to init work queue");
		exit(EXIT_SYSFAIL);
	}

	if (sockfd_init()) {
		sd_err("sockfd_init() failed");
		exit(EXIT_SYSFAIL);
	}

	ret = command_fn(argc, argv);
	if (ret == EXIT_USAGE)
		subcommand_usage(argv[1], argv[2], EXIT_USAGE);

	if (elapsed_time) {
		end = get_time_tick();
		printf("\nElapsed time: %.3lf seconds\n",
				get_time_interval(&start, &end));
	}

	return ret;
}
