/*
 * This file is part of the UCB release of Plan 9. It is subject to the license
 * terms in the LICENSE file found in the top-level directory of this
 * distribution and at http://akaros.cs.berkeley.edu/files/Plan9License. No
 * part of the UCB release of Plan 9, including this file, may be copied,
 * modified, propagated, or distributed except according to the terms contained
 * in the LICENSE file.
 */
/* posix */
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <assert.h>
#include <ctype.h>

/* bsd extensions */
#include <sys/uio.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/un.h>
#include <arpa/inet.h>

#include <sys/plan9_helpers.h>

/* Puts the path of the conversation file for 'name' for the Rock in retval,
 * which must be a char [Ctlsize]. */
void _sock_get_conv_filename(Rock *r, const char *name, char *retloc)
{
	char *p;

	strlcpy(retloc, r->ctl, Ctlsize);
	p = strrchr(retloc, '/');
	assert(p);
	p++;
	*p = 0;
	strlcat(retloc, name, Ctlsize);
}

void
_sock_ingetaddr(Rock *r, struct sockaddr_in *ip, socklen_t *alen,
                const char *a)
{
	int n, fd;
	char *p;
	char name[Ctlsize];
	int open_flags;

	/* get remote address */
	_sock_get_conv_filename(r, a, name);
	open_flags = O_RDONLY;
	open_flags |= (r->sopts & SOCK_CLOEXEC ? O_CLOEXEC : 0);
	fd = open(name, open_flags);
	if (fd >= 0) {
		n = read(fd, name, sizeof(name) - 1);
		if (n > 0) {
			name[n] = 0;
			p = strchr(name, '!');
			if (p) {
				*p++ = 0;
				ip->sin_family = AF_INET;
				ip->sin_port = htons(atoi(p));
				ip->sin_addr.s_addr = inet_addr(name);
				if (alen)
					*alen = sizeof(struct sockaddr_in);
			}
		}
		close(fd);
	}

}

/*
 * return ndb attribute type of an ip name
 */
int _sock_ipattr(const char *name)
{
	const char *p;
	int dot = 0;
	int alpha = 0;

	for (p = name; *p; p++) {
		if (isdigit(*p))
			continue;
		else if (isalpha(*p) || *p == '-')
			alpha = 1;
		else if (*p == '.')
			dot = 1;
		else
			return Tsys;
	}

	if (alpha) {
		if (dot)
			return Tdom;
		else
			return Tsys;
	}

	if (dot)
		return Tip;
	else
		return Tsys;
}

/* we can't avoid overrunning npath because we don't know how big it is. */
void _sock_srvname(char *npath, char *path)
{
	char *p;

	strcpy(npath, "/srv/UD.");
	p = strrchr(path, '/');
	if (p == 0)
		p = path;
	else
		p++;
	strcat(npath, p);
}

int _sock_srv(char *path, int fd)
{
	int sfd;
	char msg[8 + 256 + 1];

	/* change the path to something in srv */
	_sock_srvname(msg, path);

	/* remove any previous instance */
	unlink(msg);

	/* put the fd in /srv and then close it */
	sfd = creat(msg, 0666);
	if (sfd < 0) {
		close(fd);
		return -1;
	}
	snprintf(msg, sizeof(msg), "%d", fd);
	if (write(sfd, msg, strlen(msg)) < 0) {
		close(sfd);
		close(fd);
		return -1;
	}
	close(sfd);
	close(fd);
	return 0;
}

#warning "Not threadsafe!"
Rock *_sock_rock;

Rock *_sock_findrock(int fd, struct stat *dp)
{
	Rock *r;
	struct stat d;

	/* Skip the fstat if there are no socket rocks */
	if (!_sock_rock)
		return 0;
	/* If they pass us a struct stat, then they already did an fstat */
	if (dp == 0) {
		dp = &d;
		fstat(fd, dp);
	}
	for (r = _sock_rock; r; r = r->next) {
		if (r->inode == dp->st_ino && r->dev == dp->st_dev)
			break;
	}
	return r;
}

Rock *_sock_newrock(int fd)
{
	Rock *r;
	struct stat d;

	fstat(fd, &d);
	r = _sock_findrock(fd, &d);
	if (r == 0) {
		r = malloc(sizeof(Rock));
		if (r == 0)
			return 0;
		r->dev = d.st_dev;
		r->inode = d.st_ino;
		/* TODO: this is not thread-safe! */
		r->next = _sock_rock;
		_sock_rock = r;
	}
	assert(r->dev == d.st_dev);
	assert(r->inode == d.st_ino);
	r->domain = 0;
	r->stype = 0;
	r->sopts = 0;
	r->protocol = 0;
	memset(&r->addr, 0, sizeof(r->addr_stor));
	r->reserved = 0;
	memset(&r->raddr, 0, sizeof(r->raddr_stor));
	r->ctl[0] = '\0';
	r->ctl_fd = -1;
	r->other = -1;
	r->has_listen_fd = FALSE;
	r->listen_fd = -1;
	return r;
}

void _sock_fd_closed(int fd)
{
	Rock *r = _sock_findrock(fd, 0);

	if (!r)
		return;
	if (r->ctl_fd >= 0)
		close(r->ctl_fd);
	if (r->has_listen_fd) {
		close(r->listen_fd);
		/* This shouldn't matter - the rock is being closed anyways. */
		r->has_listen_fd = FALSE;
	}
}

/* For a ctlfd and a few other settings, it opens and returns the corresponding
 * datafd.  This will close cfd on error, or store it in the rock o/w. */
int _sock_data(int cfd, const char *net, int domain, int type, int protocol,
               Rock **rp)
{
	int n, fd;
	Rock *r;
	char name[Ctlsize];
	int open_flags;

	/* get the data file name */
	n = read(cfd, name, sizeof(name) - 1);
	if (n < 0) {
		close(cfd);
		errno = ENOBUFS;
		return -1;
	}
	name[n] = 0;
	n = strtoul(name, 0, 0);
	snprintf(name, sizeof(name), "/net/%s/%d/data", net, n);

	/* open data file */
	open_flags = O_RDWR;
	open_flags |= (type & SOCK_NONBLOCK ? O_NONBLOCK : 0);
	open_flags |= (type & SOCK_CLOEXEC ? O_CLOEXEC : 0);
	fd = open(name, open_flags);
	if (fd < 0) {
		close(cfd);
		errno = ENOBUFS;
		return -1;
	}

	/* hide stuff under the rock */
	snprintf(name, sizeof(name), "/net/%s/%d/ctl", net, n);
	r = _sock_newrock(fd);
	if (r == 0) {
		close(cfd);
		close(fd);
		errno = ENOBUFS;
		return -1;
	}
	if (rp)
		*rp = r;
	memset(&r->raddr, 0, sizeof(r->raddr_stor));
	memset(&r->addr, 0, sizeof(r->addr_stor));
	r->domain = domain;
	r->stype = _sock_strip_opts(type);
	r->sopts = _sock_get_opts(type);
	r->protocol = protocol;
	strcpy(r->ctl, name);
	r->ctl_fd = cfd;
	return fd;
}

/* Takes network-byte ordered IPv4 addr and writes it into buf, in the plan 9 IP
 * addr format */
void naddr_to_plan9addr(uint32_t sin_addr, uint8_t *buf)
{
	uint8_t *sin_bytes = (uint8_t *)&sin_addr;

	memset(buf, 0, 10);
	buf += 10;
	buf[0] = 0xff;
	buf[1] = 0xff;
	buf += 2;
	buf[0] = sin_bytes[0];	/* e.g. 192 */
	buf[1] = sin_bytes[1];	/* e.g. 168 */
	buf[2] = sin_bytes[2];	/* e.g.   0 */
	buf[3] = sin_bytes[3];	/* e.g.   1 */
}

/* does v4 only */
uint32_t plan9addr_to_naddr(uint8_t *buf)
{
	buf += 12;
	return (buf[3] << 24) | (buf[2] << 16) | (buf[1] << 8) | (buf[0] << 0);
}

/* Returns a rock* if the socket exists and is UDP */
Rock *udp_sock_get_rock(int fd)
{
	Rock *r = _sock_findrock(fd, 0);

	if (!r) {
		errno = ENOTSOCK;
		return 0;
	}
	if ((r->domain == PF_INET) && (r->stype == SOCK_DGRAM))
		return r;

	return 0;
}

/* In Linux, socket options are multiplexed in the socket type. */
int _sock_strip_opts(int type)
{
	return type & ~(SOCK_NONBLOCK | SOCK_CLOEXEC);
}

int _sock_get_opts(int type)
{
	return type & (SOCK_NONBLOCK | SOCK_CLOEXEC);
}

/* Opens the FD for "listen", and attaches it to the Rock.  When the dfd (and
 * thus the Rock) closes, we'll close the listen file too.  Returns the FD on
 * success, -1 on error.  This is racy, like a lot of other Rock stuff. */
static int _rock_open_listen_fd(Rock *r)
{
	char listen_file[Ctlsize];
	int ret;
	int open_flags;

	_sock_get_conv_filename(r, "listen", listen_file);
	open_flags = O_PATH;
	open_flags |= (r->sopts & SOCK_CLOEXEC ? O_CLOEXEC : 0);
	ret = open(listen_file, open_flags);
	/* Probably a bug in the rock code (or the kernel!) if we couldn't walk
	 * to our listen. */
	assert(ret >= 0);
	r->listen_fd = ret;
	r->has_listen_fd = TRUE;

	return ret;
}

/* Used by user/iplib (e.g. epoll).  Returns the FDs for the ctl_fd and for the
 * listen file, opened O_PATH, for this conversation.  Returns -1 on no FDs. */
void _sock_lookup_rock_fds(int sock_fd, bool can_open_listen_fd,
                           int *listen_fd_r, int *ctl_fd_r)
{
	Rock *r = _sock_findrock(sock_fd, 0);

	*listen_fd_r = -1;
	*ctl_fd_r = -1;
	if (!r || r->domain == PF_UNIX)
		return;
	if (!r->has_listen_fd && can_open_listen_fd)
		_rock_open_listen_fd(r);
	*listen_fd_r = r->listen_fd;	/* might still be -1.  that's OK. */
	*ctl_fd_r = r->ctl_fd;
}

/* Used by fcntl for F_SETFL. */
void _sock_mirror_fcntl(int sock_fd, int cmd, long arg)
{
	Rock *r = _sock_findrock(sock_fd, 0);

	if (!r || r->domain == PF_UNIX)
		return;
	if (r->ctl_fd >= 0)
		syscall(SYS_fcntl, r->ctl_fd, cmd, arg);
	if (r->has_listen_fd)
		syscall(SYS_fcntl, r->listen_fd, cmd, arg);
}

/* Given an FD, opens the FD with the name 'sibling' in the same directory.
 * e.g., you have a data, you open a ctl.  Don't use this with cloned FDs (i.e.
 * open clone, get a ctl back) until we fix 9p and fd2path.
 *
 * Careful, this will always open O_CLOEXEC.  The rationale is that the callers
 * of this are low-level libraries that quickly close the FD, before any
 * non-malicious exec. */
int get_sibling_fd(int fd, const char *sibling)
{
	char path[MAX_PATH_LEN];
	char *graft;

	if (syscall(SYS_fd2path, fd, path, sizeof(path)) < 0)
		return -1;
	graft = strrchr(path, '/');
	if (!graft)
		return -1;
	graft++;
	*graft = 0;
	snprintf(graft, sizeof(path) - strlen(path), sibling);
	return open(path, O_RDWR | O_CLOEXEC);
}

/* Writes num to FD in ASCII in hex format. */
int write_hex_to_fd(int fd, uint64_t num)
{
	int ret;
	char cmd[50];
	char *ptr;

	ptr = u64_to_str(num, cmd, sizeof(cmd));
	if (!ptr)
		return -1;
	ret = write(fd, ptr, sizeof(cmd) - (ptr - cmd));
	if (ret <= 0)
		return -1;
	return 0;
}

/* Returns a char representing the lowest 4 bits of x */
static char num_to_nibble(unsigned int x)
{
	return "0123456789abcdef"[x & 0xf];
}

/* Converts num to a string, in hex, using buf as storage.  Returns a pointer to
 * the string from within your buf, or 0 on failure. */
char *u64_to_str(uint64_t num, char *buf, size_t len)
{
	char *ptr;
	size_t nr_nibbles = sizeof(num) * 8 / 4;

	/* 3: 0, x, and \0 */
	if (len < nr_nibbles + 3)
		return 0;
	ptr = &buf[len - 1];
	/* Build the string backwards */
	*ptr = '\0';
	for (int i = 0; i < nr_nibbles; i++) {
		ptr--;
		*ptr = num_to_nibble(num);
		num >>= 4;
	}
	ptr--;
	*ptr = 'x';
	ptr--;
	*ptr = '0';
	return ptr;
}
