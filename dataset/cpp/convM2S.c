/* 
 * This file is part of the UCB release of Plan 9. It is subject to the license
 * terms in the LICENSE file found in the top-level directory of this
 * distribution and at http://akaros.cs.berkeley.edu/files/Plan9License. No
 * part of the UCB release of Plan 9, including this file, may be copied,
 * modified, propagated, or distributed except according to the terms contained
 * in the LICENSE file.
 */

#include <string.h>
#include <fcall.h>

static uint8_t *gstring(uint8_t * p, uint8_t * ep, char **s)
{
	unsigned int n;

	if (p + BIT16SZ > ep)
		return NULL;
	n = GBIT16(p);
	p += BIT16SZ - 1;
	if (p + n + 1 > ep)
		return NULL;
	/* move it down, on top of count, to make room for '\0' */
	memmove(p, p + 1, n);
	p[n] = '\0';
	*s = (char *)p;
	p += n + 1;
	return p;
}

static uint8_t *gqid(uint8_t * p, uint8_t * ep, struct qid *q)
{
	if (p + QIDSZ > ep)
		return NULL;
	q->type = GBIT8(p);
	p += BIT8SZ;
	q->vers = GBIT32(p);
	p += BIT32SZ;
	q->path = GBIT64(p);
	p += BIT64SZ;
	return p;
}

void init_empty_dir(struct dir *d)
{
	d->type = ~0;
	d->dev = ~0;
	d->qid.path = ~0;
	d->qid.vers = ~0;
	d->qid.type = ~0;
	d->mode = ~0;
	d->atime = ~0;
	d->mtime = ~0;
	d->length = ~0;
	d->name = "";
	d->uid = "";
	d->gid = "";
	d->muid = "";
}

/*
 * no syntactic checks.
 * three causes for error:
 *  1. message size field is incorrect
 *  2. input buffer too short for its own data (counts too long, etc.)
 *  3. too many names or qids
 * gqid() and gstring() return NULL if they would reach beyond buffer.
 * main switch statement checks range and also can fall through
 * to test at end of routine.
 */
unsigned int convM2S(uint8_t * ap, unsigned int nap, struct fcall *f)
{
	uint8_t *p, *ep;
	unsigned int i, size;

	p = ap;
	ep = p + nap;

	if (p + BIT32SZ + BIT8SZ + BIT16SZ > ep)
		return 0;
	size = GBIT32(p);
	p += BIT32SZ;

	if (size < BIT32SZ + BIT8SZ + BIT16SZ)
		return 0;

	f->type = GBIT8(p);
	p += BIT8SZ;
	f->tag = GBIT16(p);
	p += BIT16SZ;

	switch (f->type) {
	default:
		return 0;

	case Tversion:
		if (p + BIT32SZ > ep)
			return 0;
		f->msize = GBIT32(p);
		p += BIT32SZ;
		p = gstring(p, ep, &f->version);
		break;

	case Tflush:
		if (p + BIT16SZ > ep)
			return 0;
		f->oldtag = GBIT16(p);
		p += BIT16SZ;
		break;

	case Tauth:
		if (p + BIT32SZ > ep)
			return 0;
		f->afid = GBIT32(p);
		p += BIT32SZ;
		p = gstring(p, ep, &f->uname);
		if (p == NULL)
			break;
		p = gstring(p, ep, &f->aname);
		if (p == NULL)
			break;
		break;

	case Tattach:
		if (p + BIT32SZ > ep)
			return 0;
		f->fid = GBIT32(p);
		p += BIT32SZ;
		if (p + BIT32SZ > ep)
			return 0;
		f->afid = GBIT32(p);
		p += BIT32SZ;
		p = gstring(p, ep, &f->uname);
		if (p == NULL)
			break;
		p = gstring(p, ep, &f->aname);
		if (p == NULL)
			break;
		break;

	case Twalk:
		if (p + BIT32SZ + BIT32SZ + BIT16SZ > ep)
			return 0;
		f->fid = GBIT32(p);
		p += BIT32SZ;
		f->newfid = GBIT32(p);
		p += BIT32SZ;
		f->nwname = GBIT16(p);
		p += BIT16SZ;
		if (f->nwname > MAXWELEM)
			return 0;
		for (i = 0; i < f->nwname; i++) {
			p = gstring(p, ep, &f->wname[i]);
			if (p == NULL)
				break;
		}
		break;

	case Topen:
		if (p + BIT32SZ + BIT8SZ > ep)
			return 0;
		f->fid = GBIT32(p);
		p += BIT32SZ;
		f->mode = GBIT8(p);
		p += BIT8SZ;
		break;

	case Tcreate:
		if (p + BIT32SZ > ep)
			return 0;
		f->fid = GBIT32(p);
		p += BIT32SZ;
		p = gstring(p, ep, &f->name);
		if (p == NULL)
			break;
		if (p + BIT32SZ + BIT8SZ > ep)
			return 0;
		f->perm = GBIT32(p);
		p += BIT32SZ;
		f->mode = GBIT8(p);
		p += BIT8SZ;
		break;

	case Tread:
		if (p + BIT32SZ + BIT64SZ + BIT32SZ > ep)
			return 0;
		f->fid = GBIT32(p);
		p += BIT32SZ;
		f->offset = GBIT64(p);
		p += BIT64SZ;
		f->count = GBIT32(p);
		p += BIT32SZ;
		break;

	case Twrite:
		if (p + BIT32SZ + BIT64SZ + BIT32SZ > ep)
			return 0;
		f->fid = GBIT32(p);
		p += BIT32SZ;
		f->offset = GBIT64(p);
		p += BIT64SZ;
		f->count = GBIT32(p);
		p += BIT32SZ;
		if (p + f->count > ep)
			return 0;
		f->data = (char *)p;
		p += f->count;
		break;

	case Tclunk:
	case Tremove:
		if (p + BIT32SZ > ep)
			return 0;
		f->fid = GBIT32(p);
		p += BIT32SZ;
		break;

	case Tstat:
		if (p + BIT32SZ > ep)
			return 0;
		f->fid = GBIT32(p);
		p += BIT32SZ;
		break;

	case Twstat:
		if (p + BIT32SZ + BIT16SZ > ep)
			return 0;
		f->fid = GBIT32(p);
		p += BIT32SZ;
		f->nstat = GBIT16(p);
		p += BIT16SZ;
		if (p + f->nstat > ep)
			return 0;
		f->stat = p;
		p += f->nstat;
		break;



	case Rversion:
		if (p + BIT32SZ > ep)
			return 0;
		f->msize = GBIT32(p);
		p += BIT32SZ;
		p = gstring(p, ep, &f->version);
		break;

	case Rerror:
		p = gstring(p, ep, &f->ename);
		break;

	case Rflush:
		break;

	case Rauth:
		p = gqid(p, ep, &f->aqid);
		if (p == NULL)
			break;
		break;

	case Rattach:
		p = gqid(p, ep, &f->qid);
		if (p == NULL)
			break;
		break;

	case Rwalk:
		if (p + BIT16SZ > ep)
			return 0;
		f->nwqid = GBIT16(p);
		p += BIT16SZ;
		if (f->nwqid > MAXWELEM)
			return 0;
		for (i = 0; i < f->nwqid; i++) {
			p = gqid(p, ep, &f->wqid[i]);
			if (p == NULL)
				break;
		}
		break;

	case Ropen:
	case Rcreate:
		p = gqid(p, ep, &f->qid);
		if (p == NULL)
			break;
		if (p + BIT32SZ > ep)
			return 0;
		f->iounit = GBIT32(p);
		p += BIT32SZ;
		break;

	case Rread:
		if (p + BIT32SZ > ep)
			return 0;
		f->count = GBIT32(p);
		p += BIT32SZ;
		if (p + f->count > ep)
			return 0;
		f->data = (char *)p;
		p += f->count;
		break;

	case Rwrite:
		if (p + BIT32SZ > ep)
			return 0;
		f->count = GBIT32(p);
		p += BIT32SZ;
		break;

	case Rclunk:
	case Rremove:
		break;

	case Rstat:
		if (p + BIT16SZ > ep)
			return 0;
		f->nstat = GBIT16(p);
		p += BIT16SZ;
		if (p + f->nstat > ep)
			return 0;
		f->stat = p;
		p += f->nstat;
		break;

	case Rwstat:
		break;
	}

	if (p == NULL || p > ep)
		return 0;
	if (ap + size == p)
		return size;
	return 0;
}
