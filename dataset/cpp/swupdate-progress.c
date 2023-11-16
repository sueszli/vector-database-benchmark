/*
 * (C) Copyright 2016
 * Stefano Babic, DENX Software Engineering, sbabic@denx.de.
 *
 * SPDX-License-Identifier:     GPL-2.0-only
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/select.h>
#include <sys/reboot.h>
#include <linux/reboot.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <pthread.h>
#include <getopt.h>

#include <progress_ipc.h>

#define PSPLASH_MSG_SIZE	64

#define RESET		0
#define BRIGHT 		1
#define DIM		2
#define UNDERLINE 	3
#define BLINK		4
#define REVERSE		7
#define HIDDEN		8

#define BLACK 		0
#define RED		1
#define GREEN		2
#define YELLOW		3
#define BLUE		4
#define MAGENTA		5
#define CYAN		6
#define	WHITE		7

static bool silent = false;

static void resetterm(void)
{
	if (!silent)
		fprintf(stdout, "%c[%dm", 0x1B, RESET);
}

static void textcolor(int attr, int fg, int bg)
{
	if (!silent)
		fprintf(stdout, "%c[%d;%d;%dm", 0x1B, attr, fg + 30, bg + 40);
}

static struct option long_options[] = {
	{"help", no_argument, NULL, 'h'},
	{"psplash", no_argument, NULL, 'p'},
	{"reboot", no_argument, NULL, 'r'},
	{"wait", no_argument, NULL, 'w'},
	{"color", no_argument, NULL, 'c'},
	{"socket", required_argument, NULL, 's'},
	{"exec", required_argument, NULL, 'e'},
	{"quiet", no_argument, NULL, 'q'},
	{NULL, 0, NULL, 0}
};

static void usage(char *programname)
{
	fprintf(stdout, "%s (compiled %s)\n", programname, __DATE__);
	fprintf(stdout, "Usage %s [OPTION]\n",
			programname);
	fprintf(stdout,
		" -c, --color             : Use colors to show results\n"
		" -e, --exec <script>     : call the script with the result of update\n"
		" -r, --reboot            : reboot after a successful update\n"
		" -w, --wait              : wait for a connection with SWUpdate\n"
		" -p, --psplash           : send info to the psplash process\n"
		" -s, --socket <path>     : path to progress IPC socket\n"
		" -h, --help              : print this help and exit\n"
		" -q, --quiet             : do not print progress bar\n"
		);
}

static int psplash_init(char *pipe)
{
	int psplash_pipe_fd;
	int pid_psplash;

	if ((psplash_pipe_fd = open(pipe, O_WRONLY | O_NONBLOCK)) == -1) {
		/* Try to run psplash in background */
		pid_psplash = fork();
		if (pid_psplash < 0)
			return 0;
		else if (pid_psplash == 0) {
			execl("/usr/bin/psplash", "psplash", (char *)0);
			exit(1);
		} else {
			sleep(1);
			if ((psplash_pipe_fd = open(pipe, O_WRONLY | O_NONBLOCK)) == -1) {
				return 0;
			}
		}
	}

	close(psplash_pipe_fd);

	return 1;
}

static void psplash_write_fifo(char *pipe, char *buf)
{
	int   psplash_pipe_fd, ret;

	if ((psplash_pipe_fd = open(pipe, O_WRONLY | O_NONBLOCK)) == -1) {
		fprintf(stderr, "Error unable to open psplash pipe, closing...\n");
		return;
	}

	ret = write(psplash_pipe_fd, buf, strlen(buf) + 1);
	if (ret < 0) {
		fprintf(stderr, "PSPLASH not available anymore");
	}

	close(psplash_pipe_fd);
}

static void psplash_progress(char *pipe, struct progress_msg *pmsg)
{
	char *buf;

	buf = malloc(PSPLASH_MSG_SIZE);

	if (!buf)
		return;

	switch (pmsg->status) {
	case SUCCESS:
	case FAILURE:
		snprintf(buf, PSPLASH_MSG_SIZE - 1, "MSG %s",
			 pmsg->status == SUCCESS ? "SUCCESS" : "FAILURE");
		psplash_write_fifo(pipe, buf);

		sleep(5);

		snprintf(buf, PSPLASH_MSG_SIZE - 1, "QUIT");
		psplash_write_fifo(pipe, buf);
		free(buf);
		return;
		break;
	case DONE:
		free(buf);
		return;
		break;
	default:
		break;
	}

	snprintf(buf, PSPLASH_MSG_SIZE - 1, "MSG step %d of %d",
		       	pmsg->cur_step, pmsg->nsteps);
	psplash_write_fifo(pipe, buf);

	usleep(100);

	snprintf(buf, PSPLASH_MSG_SIZE - 1, "PROGRESS %d", pmsg->cur_percent);
	psplash_write_fifo(pipe, buf);

	free(buf);
}

static void fill_progress_bar(char *bar, size_t size, unsigned int percent)
{
	/* the max len for a bar is size-1 sue to string terminator */
	unsigned int filled_len, remain;

	if (percent > 100)
		percent = 100;
	filled_len = (size - 1) * percent / 100;
	memset(bar, 0, size);

	memset(bar,'=', filled_len);
	remain = (size - 1) - filled_len;
	memset(&bar[filled_len], '-', remain);
}

static void reboot_device(void)
{
	sleep(5);
	sync();
	if (reboot(LINUX_REBOOT_CMD_RESTART) < 0) { /* Should never happen. */
		fprintf(stdout, "Please reset the board.\n");
	}
}

static void run_post_script(char *script, struct progress_msg *msg)
{
	char *cmd;
	if (asprintf(&cmd, "%s %s", script,
		     msg->status == SUCCESS ?
		     "SUCCESS" : "FAILURE") == -1) {
		fprintf(stderr, "OOM calling post-exec script\n");
		return;
	}
	int ret = system(cmd);
	if (ret) {
		fprintf(stdout, "Executed %s with error : %d\n", cmd, ret);
	}
	free(cmd);
}

int main(int argc, char **argv)
{
	int connfd;
	struct progress_msg msg;
	const char *rundir;
	char psplash_pipe_path[256];
	int psplash_ok = 0;
	unsigned int curstep = 0;
	unsigned int percent = 0;
	const int bar_len = 60;
	char bar[bar_len+1];
	int opt_c = 0;
	int opt_w = 0;
	int opt_r = 0;
	int opt_p = 0;
	int c;
	char *script = NULL;
	bool wait_update = true;
	bool disable_reboot = false;

	/* Process options with getopt */
	while ((c = getopt_long(argc, argv, "cwprhs:e:q",
				long_options, NULL)) != EOF) {
		switch (c) {
		case 'c':
			opt_c = 1;
			break;
		case 'w':
			opt_w = 1;
			break;
		case 'p':
			opt_p = 1;
			break;
		case 'r':
			opt_r = 1;
			break;
		case 's':
			SOCKET_PROGRESS_PATH = strdup(optarg);
			break;
		case 'e':
			script = strdup(optarg);
			break;
		case 'h':
			usage(argv[0]);
			exit(0);
			break;
		case 'q':
			silent = true;
			break;
		default:
			usage(argv[0]);
			exit(1);
			break;
		}
	}
		
	if (opt_p) {
		rundir = getenv("PSPLASH_FIFO_DIR");
		if(!rundir){
			rundir = getenv("RUNTIME_DIRECTORY");
		}
		if (!rundir)
			rundir = "/run";
		snprintf(psplash_pipe_path, sizeof(psplash_pipe_path), "%s/psplash_fifo", rundir);
	}
	connfd = -1;
	while (1) {
		if (connfd < 0) {
			connfd = progress_ipc_connect(opt_w);
		}

		/*
		 * if still fails, try later
		 */
		if (connfd < 0) {
			sleep(1);
			continue;
		}

		if (progress_ipc_receive(&connfd, &msg) <= 0) {
			continue;
		}

		/*
		 * Something happens, show the info
		 */
		if (wait_update) {
			if (msg.status == START || msg.status == RUN) {
				fprintf(stdout, "\n\nUpdate started !\n");
				fprintf(stdout, "Interface: ");
				switch (msg.source) {
				case SOURCE_UNKNOWN:
					fprintf(stdout, "UNKNOWN\n\n");
					break;
				case SOURCE_WEBSERVER:
					fprintf(stdout, "WEBSERVER\n\n");
					break;
				case SOURCE_SURICATTA:
					fprintf(stdout, "BACKEND\n\n");
					break;
				case SOURCE_DOWNLOADER:
					fprintf(stdout, "DOWNLOADER\n\n");
					break;
				case SOURCE_CHUNKS_DOWNLOADER:
					fprintf(stdout, "CHUNKS DOWNLOADER\n\n");
					break;
				case SOURCE_LOCAL:
					fprintf(stdout, "LOCAL\n\n");
					break;
				}
				/*
				 * Reset per update variables
				 */
				curstep = 0;
				disable_reboot = false;
				wait_update = false;
			}
		}

		/*
		 * Be sure that string in message are Null terminated
		 */
		if (msg.infolen > 0) {
			char *reboot_mode;
			int n, cause;

			if (msg.infolen >= sizeof(msg.info) - 1) {
				msg.infolen = sizeof(msg.info) - 1;
			}
			msg.info[msg.infolen] = '\0';
			fprintf(stdout, "INFO : %s\n", msg.info);

			/*
			 * Check for no-reboot mode
			 * Just do a simple parsing for now. If more messages
			 * will be added, JSON lib should be linked.
			 */
			n = sscanf(msg.info, "{\"%d\": { \"reboot-mode\" : \"%m[-a-z]\"}}",
				   &cause, &reboot_mode);
			if (n == 2) {
				if (cause == CAUSE_REBOOT_MODE) {
					if (!strcmp(reboot_mode, "no-reboot")) {
						disable_reboot = true;
					}
				}
				free(reboot_mode);
			}
		}
		msg.cur_image[sizeof(msg.cur_image) - 1] = '\0';

		if (!psplash_ok && opt_p) {
			psplash_ok = psplash_init(psplash_pipe_path);
		}

		if (!wait_update) {

			if (msg.cur_step > 0) {
				if ((msg.cur_step != curstep) && (curstep != 0)){
					if (!silent) {
						fprintf(stdout, "\n");
						fflush(stdout);
					}
				}
				fill_progress_bar(bar, sizeof(bar), msg.cur_percent);

				if (!silent) {
					fprintf(stdout, "[ %.*s ] %d of %d %d%% (%s), dwl %d%% of %llu bytes\r",
						bar_len,
						bar,
						msg.cur_step, msg.nsteps, msg.cur_percent,
						msg.cur_image, msg.dwl_percent, msg.dwl_bytes);
					fflush(stdout);
				}

				if (psplash_ok && ((msg.cur_step != curstep) || (msg.cur_percent != percent))) {
					psplash_progress(psplash_pipe_path, &msg);
				}
				curstep = msg.cur_step;
				percent = msg.cur_percent;
			}
		}

		switch (msg.status) {
		case SUCCESS:
		case FAILURE:
			if (opt_c) {
				if (msg.status == FAILURE)
					textcolor(BLINK, RED, BLACK);
				else
					textcolor(BRIGHT, GREEN, BLACK);
			}

			fprintf(stdout, "\n%s !\n", msg.status == SUCCESS
							  ? "SUCCESS"
							  : "FAILURE");
			if (script) {
				run_post_script(script, &msg);
			}
			resetterm();
		#if !defined(CONFIG_SURICATTA_WFX)
			if (psplash_ok)
				psplash_progress(psplash_pipe_path, &msg);
			psplash_ok = 0;
			if ((msg.status == SUCCESS) && (msg.cur_step > 0) && opt_r && !disable_reboot) {
				reboot_device();
			}
		#else
			if (msg.status == SUCCESS) {
				fprintf(stdout, "\nWaiting for activation.\n");
				char *buf = alloca(PSPLASH_MSG_SIZE);
				snprintf(buf, PSPLASH_MSG_SIZE - 1, "MSG Waiting for activation.");
				psplash_write_fifo(psplash_pipe_path, buf);
			}
			if (msg.status == FAILURE) {
				if (psplash_ok)
					psplash_progress(psplash_pipe_path, &msg);
				psplash_ok = 0;
			}
		#endif
			wait_update = true;
			break;
		case DONE:
			fprintf(stdout, "\nDONE.\n\n");
			break;
		case PROGRESS:
			#if defined(CONFIG_SURICATTA_WFX)
			if (strcasestr(msg.info, "\"state\": \"ACTIVATING\"") &&
			    strcasestr(msg.info, "\"progress\": 100")) {
				msg.status = SUCCESS;
				if (psplash_ok)
					psplash_progress(psplash_pipe_path, &msg);
				psplash_ok = 0;
				if (opt_r && !disable_reboot && strcasestr(msg.info, "firmware")) {
					reboot_device();
					break;
				}
				fprintf(stdout, "Don't know how to activate this update, doing nothing.\n");
			}
			#endif
			break;
		default:
			break;
		}
	}
}
