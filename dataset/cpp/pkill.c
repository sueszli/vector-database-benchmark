/*	$OpenBSD: pkill.c,v 1.17 2008/06/26 05:42:21 ray Exp $	*/
/*	$NetBSD: pkill.c,v 1.5 2002/10/27 11:49:34 kleink Exp $	*/

/*-
 * Copyright (c) 2002 The NetBSD Foundation, Inc.
 * All rights reserved.
 *
 * This code is derived from software contributed to The NetBSD Foundation
 * by Andrew Doran.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE NETBSD FOUNDATION, INC. AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef lint
static const char rcsid[] = "$OpenBSD: pkill.c,v 1.17 2008/06/26 05:42:21 ray Exp $";
#endif /* !lint */

#include <sys/types.h>
#include <sys/param.h>
#include <sys/sysctl.h>
#include <sys/proc.h>
#include <sys/queue.h>
#include <sys/stat.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <regex.h>
#include <ctype.h>
#include <kvm.h>
#include <err.h>
#include <pwd.h>
#include <grp.h>
#include <errno.h>

#define	STATUS_MATCH	0
#define	STATUS_NOMATCH	1
#define	STATUS_BADUSAGE	2
#define	STATUS_ERROR	3

enum listtype {
	LT_GENERIC,
	LT_USER,
	LT_GROUP,
	LT_TTY,
	LT_PGRP,
	LT_SID
};

struct list {
	SLIST_ENTRY(list) li_chain;
	long	li_number;
};

SLIST_HEAD(listhead, list);

struct kinfo_proc2	*plist;
char	*selected;
char	*delim = "\n";
int	nproc;
int	pgrep;
int	signum = SIGTERM;
int	newest;
int	oldest;
int	inverse;
int	longfmt;
int	matchargs;
int	fullmatch;
kvm_t	*kd;
pid_t	mypid;

struct listhead euidlist = SLIST_HEAD_INITIALIZER(list);
struct listhead ruidlist = SLIST_HEAD_INITIALIZER(list);
struct listhead rgidlist = SLIST_HEAD_INITIALIZER(list);
struct listhead pgrplist = SLIST_HEAD_INITIALIZER(list);
struct listhead ppidlist = SLIST_HEAD_INITIALIZER(list);
struct listhead tdevlist = SLIST_HEAD_INITIALIZER(list);
struct listhead sidlist = SLIST_HEAD_INITIALIZER(list);

int	main(int, char **);
void	usage(void);
int	killact(struct kinfo_proc2 *, int);
int	grepact(struct kinfo_proc2 *, int);
void	makelist(struct listhead *, enum listtype, char *);

extern char *__progname;

int
main(int argc, char **argv)
{
	extern char *optarg;
	extern int optind;
	char buf[_POSIX2_LINE_MAX], *mstr, **pargv, *p, *q;
	int i, j, ch, bestidx, rv, criteria;
	int (*action)(struct kinfo_proc2 *, int);
	struct kinfo_proc2 *kp;
	struct list *li;
	u_int32_t bestsec, bestusec;
	regex_t reg;
	regmatch_t regmatch;

	if (strcmp(__progname, "pgrep") == 0) {
		action = grepact;
		pgrep = 1;
	} else {
		action = killact;
		p = argv[1];

		if (argc > 1 && p[0] == '-') {
			p++;
			i = (int)strtol(p, &q, 10);
			if (*q == '\0') {
				signum = i;
				argv++;
				argc--;
			} else {
				if (strncasecmp(p, "sig", 3) == 0)
					p += 3;
				for (i = 1; i < NSIG; i++)
					if (strcasecmp(sys_signame[i], p) == 0)
						break;
				if (i != NSIG) {
					signum = i;
					argv++;
					argc--;
				}
			}
		}
	}

	criteria = 0;

	while ((ch = getopt(argc, argv, "G:P:U:d:fg:lnos:t:u:vx")) != -1)
		switch (ch) {
		case 'G':
			makelist(&rgidlist, LT_GROUP, optarg);
			criteria = 1;
			break;
		case 'P':
			makelist(&ppidlist, LT_GENERIC, optarg);
			criteria = 1;
			break;
		case 'U':
			makelist(&ruidlist, LT_USER, optarg);
			criteria = 1;
			break;
		case 'd':
			if (!pgrep)
				usage();
			delim = optarg;
			break;
		case 'f':
			matchargs = 1;
			break;
		case 'g':
			makelist(&pgrplist, LT_PGRP, optarg);
			criteria = 1;
			break;
		case 'l':
			if (!pgrep)
				usage();
			longfmt = 1;
			break;
		case 'n':
			newest = 1;
			criteria = 1;
			break;
		case 'o':
			oldest = 1;
			criteria = 1;
			break;
		case 's':
			makelist(&sidlist, LT_SID, optarg);
			criteria = 1;
			break;
		case 't':
			makelist(&tdevlist, LT_TTY, optarg);
			criteria = 1;
			break;
		case 'u':
			makelist(&euidlist, LT_USER, optarg);
			criteria = 1;
			break;
		case 'v':
			inverse = 1;
			break;
		case 'x':
			fullmatch = 1;
			break;
		default:
			usage();
			/* NOTREACHED */
		}

	argc -= optind;
	argv += optind;
	if (argc != 0)
		criteria = 1;
	if (!criteria || (newest && oldest))
		usage();

	mypid = getpid();

	/*
	 * Retrieve the list of running processes from the kernel.
	 */
	kd = kvm_openfiles(NULL, NULL, NULL, KVM_NO_FILES, buf);
	if (kd == NULL)
		errx(STATUS_ERROR, "kvm_openfiles(): %s", buf);

	plist = kvm_getproc2(kd, KERN_PROC_ALL, 0, sizeof(*plist), &nproc);
	if (plist == NULL)
		errx(STATUS_ERROR, "kvm_getproc2() failed");

	/*
	 * Allocate memory which will be used to keep track of the
	 * selection.
	 */
	if ((selected = malloc(nproc)) == NULL)
		errx(STATUS_ERROR, "memory allocation failure");
	memset(selected, 0, nproc);

	/*
	 * Refine the selection.
	 */
	for (; *argv != NULL; argv++) {
		if ((rv = regcomp(&reg, *argv, REG_EXTENDED)) != 0) {
			regerror(rv, &reg, buf, sizeof(buf));
			errx(STATUS_BADUSAGE, "bad expression: %s", buf);
		}

		for (i = 0, kp = plist; i < nproc; i++, kp++) {
			if ((kp->p_flag & P_SYSTEM) != 0 || kp->p_pid == mypid)
				continue;

			if (matchargs) {
				if ((pargv = kvm_getargv2(kd, kp, 0)) == NULL)
					continue;

				j = 0;
				while (j < sizeof(buf) && *pargv != NULL) {
					int ret;

					ret = snprintf(buf + j, sizeof(buf) - j,
					    pargv[1] != NULL ? "%s " : "%s",
					    pargv[0]);
					if (ret >= sizeof(buf) - j)
						j += sizeof(buf) - j - 1;
					else if (ret > 0)
						j += ret;
					pargv++;
				}

				mstr = buf;
			} else
				mstr = kp->p_comm;

			rv = regexec(&reg, mstr, 1, &regmatch, 0);
			if (rv == 0) {
				if (fullmatch) {
					if (regmatch.rm_so == 0 &&
					    regmatch.rm_eo == strlen(mstr))
						selected[i] = 1;
				} else
					selected[i] = 1;
			} else if (rv != REG_NOMATCH) {
				regerror(rv, &reg, buf, sizeof(buf));
				errx(STATUS_ERROR, "regexec(): %s", buf);
			}
		}

		regfree(&reg);
	}

	for (i = 0, kp = plist; i < nproc; i++, kp++) {
		if ((kp->p_flag & P_SYSTEM) != 0 || kp->p_pid == mypid)
			continue;

		SLIST_FOREACH(li, &ruidlist, li_chain)
			if (kp->p_ruid == (uid_t)li->li_number)
				break;
		if (SLIST_FIRST(&ruidlist) != NULL && li == NULL) {
			selected[i] = 0;
			continue;
		}

		SLIST_FOREACH(li, &rgidlist, li_chain)
			if (kp->p_rgid == (gid_t)li->li_number)
				break;
		if (SLIST_FIRST(&rgidlist) != NULL && li == NULL) {
			selected[i] = 0;
			continue;
		}

		SLIST_FOREACH(li, &euidlist, li_chain)
			if (kp->p_uid == (uid_t)li->li_number)
				break;
		if (SLIST_FIRST(&euidlist) != NULL && li == NULL) {
			selected[i] = 0;
			continue;
		}

		SLIST_FOREACH(li, &ppidlist, li_chain)
			if (kp->p_ppid == (uid_t)li->li_number)
				break;
		if (SLIST_FIRST(&ppidlist) != NULL && li == NULL) {
			selected[i] = 0;
			continue;
		}

		SLIST_FOREACH(li, &pgrplist, li_chain)
			if (kp->p__pgid == (uid_t)li->li_number)
				break;
		if (SLIST_FIRST(&pgrplist) != NULL && li == NULL) {
			selected[i] = 0;
			continue;
		}

		SLIST_FOREACH(li, &tdevlist, li_chain) {
			if (li->li_number == -1 &&
			    (kp->p_flag & P_CONTROLT) == 0)
				break;
			if (kp->p_tdev == (uid_t)li->li_number)
				break;
		}
		if (SLIST_FIRST(&tdevlist) != NULL && li == NULL) {
			selected[i] = 0;
			continue;
		}

		SLIST_FOREACH(li, &sidlist, li_chain)
			if (kp->p_sid == (uid_t)li->li_number)
				break;
		if (SLIST_FIRST(&sidlist) != NULL && li == NULL) {
			selected[i] = 0;
			continue;
		}

		if (argc == 0)
			selected[i] = 1;
	}

	if (newest || oldest) {
		bestidx = -1;

		if (newest)
			bestsec = bestusec = 0;
		else
			bestsec = bestusec = UINT32_MAX;

		for (i = 0, kp = plist; i < nproc; i++, kp++) {
			if (!selected[i])
				continue;

			if ((newest && (kp->p_ustart_sec > bestsec ||
			    (kp->p_ustart_sec == bestsec
			    && kp->p_ustart_usec > bestusec)))
			|| (oldest && (kp->p_ustart_sec < bestsec ||
                            (kp->p_ustart_sec == bestsec
                            && kp->p_ustart_usec < bestusec)))) {

				bestsec = kp->p_ustart_sec;
				bestusec = kp->p_ustart_usec;
				bestidx = i;
			}
		}

		memset(selected, 0, nproc);
		if (bestidx != -1)
			selected[bestidx] = 1;
	}

	/*
	 * Take the appropriate action for each matched process, if any.
	 */
	rv = STATUS_NOMATCH;
	for (i = 0, j = 0, kp = plist; i < nproc; i++, kp++) {
		if ((kp->p_flag & P_SYSTEM) != 0 || kp->p_pid == mypid)
			continue;
		if (selected[i]) {
			if (inverse)
				continue;
		} else if (!inverse)
			continue;

		if ((*action)(kp, j++) == -1)
			rv = STATUS_ERROR;
		else if (rv != STATUS_ERROR)
			rv = STATUS_MATCH;
	}
	if (pgrep && j)
		putchar('\n');

	exit(rv);
}

void
usage(void)
{
	const char *ustr;

	if (pgrep)
		ustr = "[-flnovx] [-d delim]";
	else
		ustr = "[-signal] [-fnovx]";

	fprintf(stderr, "usage: %s %s [-G gid] [-g pgrp] [-P ppid] [-s sid] "
	    "[-t tty]\n\t[-U uid] [-u euid] [pattern ...]\n", __progname, ustr);

	exit(STATUS_ERROR);
}

int
killact(struct kinfo_proc2 *kp, int dummy)
{

	if (kill(kp->p_pid, signum) == -1 && errno != ESRCH) {
		warn("signalling pid %d", (int)kp->p_pid);
		return (-1);
	}
	return (0);
}

int
grepact(struct kinfo_proc2 *kp, int printdelim)
{
	char **argv;

	if (printdelim)
		fputs(delim, stdout);
	if (longfmt && matchargs) {
		if ((argv = kvm_getargv2(kd, kp, 0)) == NULL)
			return (-1);

		printf("%d ", (int)kp->p_pid);
		for (; *argv != NULL; argv++) {
			printf("%s", *argv);
			if (argv[1] != NULL)
				putchar(' ');
		}
	} else if (longfmt)
		printf("%d %s", (int)kp->p_pid, kp->p_comm);
	else
		printf("%d", (int)kp->p_pid);

	return (0);
}

void
makelist(struct listhead *head, enum listtype type, char *src)
{
	struct list *li;
	struct passwd *pw;
	struct group *gr;
	struct stat st;
	char *sp, *p, buf[MAXPATHLEN];
	int empty;

	empty = 1;

	while ((sp = strsep(&src, ",")) != NULL) {
		if (*sp == '\0')
			usage();

		if ((li = malloc(sizeof(*li))) == NULL)
			errx(STATUS_ERROR, "memory allocation failure");
		SLIST_INSERT_HEAD(head, li, li_chain);
		empty = 0;

		li->li_number = (uid_t)strtol(sp, &p, 0);
		if (*p == '\0') {
			switch (type) {
			case LT_PGRP:
				if (li->li_number == 0)
					li->li_number = getpgrp();
				break;
			case LT_SID:
				if (li->li_number == 0)
					li->li_number = getsid(mypid);
				break;
			case LT_TTY:
				usage();
			default:
				break;
			}
			continue;
		}

		switch (type) {
		case LT_USER:
			if ((pw = getpwnam(sp)) == NULL)
				errx(STATUS_BADUSAGE, "unknown user `%s'", sp);
			li->li_number = pw->pw_uid;
			break;
		case LT_GROUP:
			if ((gr = getgrnam(sp)) == NULL)
				errx(STATUS_BADUSAGE, "unknown group `%s'", sp);
			li->li_number = gr->gr_gid;
			break;
		case LT_TTY:
			if (strcmp(sp, "-") == 0) {
				li->li_number = -1;
				break;
			} else if (strcmp(sp, "co") == 0)
				p = "console";
			else if (strncmp(sp, "tty", 3) == 0)
				p = sp;
			else
				p = NULL;

			if (p == NULL)
				snprintf(buf, sizeof(buf), "/dev/tty%s", sp);
			else
				snprintf(buf, sizeof(buf), "/dev/%s", p);

			if (stat(buf, &st) < 0) {
				if (errno == ENOENT)
					errx(STATUS_BADUSAGE,
					    "no such tty: `%s'", sp);
				err(STATUS_ERROR, "stat(%s)", sp);
			}

			if (!S_ISCHR(st.st_mode))
				errx(STATUS_BADUSAGE, "not a tty: `%s'", sp);

			li->li_number = st.st_rdev;
			break;
		default:
			usage();
		}
	}

	if (empty)
		usage();
}
