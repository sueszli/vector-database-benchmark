/*
 * Copyright (c) 2018, IBM
 * Author(s): Ricardo Koller
 *
 * Permission to use, copy, modify, and/or distribute this software for
 * any purpose with or without fee is hereby granted, provided that the
 * above copyright notice and this permission notice appear in all
 * copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
 * WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE
 * AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL
 * DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA
 * OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
 * TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE.
 */

/*-
 * Copyright (c) 2003 The NetBSD Foundation, Inc.
 * All rights reserved.
 *
 * This code is derived from software contributed to The NetBSD Foundation
 * by Konrad E. Schroder <perseant@hhhh.org>.
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
/*-
 * Copyright (c) 1991, 1993
 *	The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#define _GNU_SOURCE
#include <err.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <sys/mman.h>

#include "config.h"
#include "lfs.h"
#include "lfs_accessors.h"

#define SF_IMMUTABLE 0x00020000 /* file may not be changed */

#define HIGHEST_USED_INO ULFS_ROOTINO

u_int32_t lfs_sb_cksum32(struct dlfs *fs);
u_int32_t cksum(void *str, size_t len);

/* size args */
#define NSUPERBLOCKS LFS_MAXNUMSB
#ifndef IFILE_MAP_SZ
#define IFILE_MAP_SZ	32
#endif

#define MAX_INODES (((IFILE_MAP_SZ * DFL_LFSBLOCK) / sizeof(IFILE32)) - IFILE_MAP_SZ + 1)

/*
 * calculate the maximum file size allowed with the specified block shift.
 */
#define NPTR32 ((1 << DFL_LFSBLOCK_SHIFT) / sizeof(int32_t))
#define MAXFILESIZE32                                                          \
	((ULFS_NDADDR + NPTR32 + NPTR32 * NPTR32 + NPTR32 * NPTR32 * NPTR32)   \
	 << DFL_LFSBLOCK_SHIFT)

#define NPTR64 ((1 << DFL_LFSBLOCK_SHIFT) / sizeof(int64_t))
#define MAXFILESIZE64                                                          \
	((ULFS_NDADDR + NPTR64 + NPTR64 * NPTR64 + NPTR64 * NPTR64 * NPTR64)   \
	 << DFL_LFSBLOCK_SHIFT)

#define LOG2(X)                                                                \
	((unsigned)(8 * sizeof(unsigned long long) - __builtin_clzll((X)) - 1))

#define SECTOR_TO_BYTES(_S) (DEV_BSIZE * (_S))
#define FSBLOCK_TO_BYTES(_S) (DFL_LFSBLOCK * (uint64_t)(_S))
#define SEGS_TO_FSBLOCKS(_S) (((_S) * (uint64_t)DFL_LFSSEG) / DFL_LFSBLOCK)

#define DIV_UP(_x, _y) (((_x) + (_y)-1) / (_y))
#define MIN(_x, _y) (((_x) < (_y)) ? (_x) : (_y))

static const struct dlfs dlfs32_default = {
    .dlfs_magic = LFS_MAGIC,
    .dlfs_version = LFS_VERSION,
    .dlfs_ssize = DFL_LFSSEG,
    .dlfs_bsize = DFL_LFSBLOCK,
    .dlfs_fsize = DFL_LFSFRAG,
    .dlfs_frag = DFL_LFSBLOCK / DFL_LFSFRAG,

    /* TODO: this is never changed */
    .dlfs_freehd = HIGHEST_USED_INO + 1,
    .dlfs_uinodes = 0,
    .dlfs_idaddr = 0,
    .dlfs_ifile = LFS_IFILE_INUM,
    .dlfs_offset = 0,
    .dlfs_lastpseg = 0,
    .dlfs_nextseg = 0,
    .dlfs_curseg = 0,
    .dlfs_nfiles = 0,

    /* not very efficient, but makes things easier */
    .dlfs_inopf = 1,
    .dlfs_minfree = MINFREE,
    .dlfs_maxfilesize = MAXFILESIZE32,
    .dlfs_fsbpseg = DFL_LFSSEG / DFL_LFSFRAG,
    .dlfs_inopb = 1,
    .dlfs_ifpb = DFL_LFSBLOCK / sizeof(IFILE32),
    .dlfs_sepb = DFL_LFSBLOCK / sizeof(SEGUSE),
    .dlfs_nindir = DFL_LFSBLOCK / sizeof(int32_t),
    .dlfs_nspf = DFL_LFSBLOCK / 512,
    .dlfs_cleansz = 1,
    .dlfs_segmask = DFL_LFSSEG_MASK,
    .dlfs_segshift = DFL_LFSSEG_SHIFT,
    .dlfs_bshift = DFL_LFSBLOCK_SHIFT,
    .dlfs_ffshift = DFL_LFS_FFSHIFT,
    .dlfs_fbshift = DFL_LFS_FBSHIFT,
    .dlfs_bmask = DFL_LFSBLOCK_MASK,
    .dlfs_ffmask = DFL_LFS_FFMASK,
    .dlfs_fbmask = DFL_LFS_FBMASK,
    .dlfs_blktodb = LOG2(DFL_LFSBLOCK / DEV_BSIZE),
    .dlfs_sushift = 0,
    .dlfs_maxsymlinklen = LFS32_MAXSYMLINKLEN,
    .dlfs_sboffs = {0},
    .dlfs_fsmnt = {0},
    .dlfs_pflags = LFS_PF_CLEAN,
    .dlfs_dmeta = 0,
    .dlfs_sumsize = DFL_LFSFRAG,
    .dlfs_serial = 0,
    .dlfs_ibsize = DFL_LFSFRAG,
    .dlfs_s0addr = 0,
    .dlfs_tstamp = 0,
    .dlfs_inodefmt = LFS_44INODEFMT,
    .dlfs_interleave = 0,
    .dlfs_ident = 0,
    .dlfs_fsbtodb = LOG2(DFL_LFSBLOCK / DEV_BSIZE),

    .dlfs_pad = {0},
    .dlfs_cksum = 0};

#define SEGUSE_OFF(_sepb, _i)                                                  \
	(((_i) / _sepb) * DFL_LFSBLOCK +                                       \
	 sizeof(SEGUSE) * ((_i) - ((((_i) / _sepb) * _sepb))))

#define SEGUSE_GET(_fs, _i)                                                    \
	((SEGUSE *)&(_fs->ifile.segusage[SEGUSE_OFF(_fs->lfs.dlfs_sepb, (_i))]))

#define IFILE_OFF(_ifpb, _i)                                                   \
	(((_i) / _ifpb) * DFL_LFSBLOCK +                                       \
	 sizeof(IFILE32) * ((_i) - ((((_i) / _ifpb) * _ifpb))))

#define IFILE_GET(_fs, _i)                                                     \
	((IFILE32 *)&(_fs->ifile.ifiles[IFILE_OFF(_fs->lfs.dlfs_ifpb, (_i))]))

/* XXX: doesn't advance the log. Maybe it should? */
int write_log(struct fs *fs, void *data, uint64_t len, off_t lfs_off, int remap) {
	int ret;

	ret = pwrite64(fs->fd, data, len, lfs_off);
	if (ret == len)
		return 0;
	else if (ret == -1)
		return errno;
	else
		return -1;
}

/* Add a block into the data checksum */
void segment_add_datasum(struct segment *seg, char *block, uint32_t size) {
	uint32_t i;
	for (i = 0; i < size; i += DFL_LFSBLOCK) {
		/* The final checksum will be done with a sequence of the first
		 * byte of every block */
		assert(seg->cksum_idx < MAX_BLOCKS_PER_SEG);
		seg->data_for_cksum[seg->cksum_idx++] = block[i];
	}
}

int write_superblock(struct fs *fs) {
	uint32_t i;
	int ret;

	for (i = 0; i < NSUPERBLOCKS; i++) {
		fs->lfs.dlfs_cksum = lfs_sb_cksum32(&fs->lfs);
		ret = write_log(fs, &fs->lfs, sizeof(fs->lfs),
			FSBLOCK_TO_BYTES(fs->lfs.dlfs_sboffs[i]), 0);
		if (ret != 0)
			return ret;
		fs->lfs.dlfs_serial++;
	}
	return 0;
}

/* Advance the log by nr FS blocks. */
int _advance_log(struct fs *fs, uint32_t nr) {
	struct dlfs *lfs = &fs->lfs;

	if (lfs->dlfs_avail <= nr)
		return ENOSPC;

	/* Should not be used to make space for a superblock */
	lfs->dlfs_offset += nr;
	lfs->dlfs_lastpseg += nr;
	lfs->dlfs_avail -= nr;
	assert(lfs->dlfs_bfree - nr > 0);
	lfs->dlfs_bfree -= nr;

	return 0;
}

/*
 * This sets an initial version of the segment summary at the start of the
 * segment, and sets a block for a superblock if there is any.  The offset
 * should be at the beginning of the segment already.  The resulting offset
 * would be after the segment summary, and a superblock (if any).
 */
int start_segment(struct fs *fs, struct _ifile *ifile) {
	struct segsum32 *segsum = fs->seg.segsum;
	SEGUSE *segusage;
	int ret;

	assert(fs->lfs.dlfs_offset == 1 ||
		(fs->lfs.dlfs_offset % fs->lfs.dlfs_fsbpseg == 0));
	assert(segsum != NULL);

	fs->lfs.dlfs_nclean--;
	fs->lfs.dlfs_curseg += DFL_LFSSEG / DFL_LFSBLOCK;
	fs->lfs.dlfs_nextseg += DFL_LFSSEG / DFL_LFSBLOCK;
	assert(fs->lfs.dlfs_nextseg > fs->lfs.dlfs_curseg);
	fs->seg.seg_number++;

	if (fs->lfs.dlfs_curseg == 0)
		assert(fs->lfs.dlfs_offset == 1);
	else
		assert(fs->lfs.dlfs_offset % fs->lfs.dlfs_fsbpseg == 0);

	fs->seg.cksum_idx = 0;

	segusage = SEGUSE_GET(fs, fs->seg.seg_number);
	if (segusage->su_flags & SEGUSE_SUPERBLOCK) {
		/* The first block is for the superblock of the segment (if
		 * any) */
		segment_add_datasum(&fs->seg, (char *)&fs->lfs, DFL_LFSBLOCK);
		ret = _advance_log(fs, 1);
		if (ret != 0)
			return ret;
		segusage->su_flags = SEGUSE_SUPERBLOCK;
	} else {
		segusage->su_flags = 0;
	}

	fs->seg.fs = (struct lfs *)&fs->lfs;
	fs->seg.ninodes = 0;
	fs->seg.seg_bytes_left = fs->lfs.dlfs_ssize;
	fs->seg.sum_bytes_left = fs->lfs.dlfs_sumsize;
	fs->seg.disk_bno = fs->lfs.dlfs_offset;

	/*
	 * We create one segment summary per segment. In other words,
	 * one partial segment per segment.
	 */
	segsum->ss_magic = SS_MAGIC;
	segsum->ss_next = fs->lfs.dlfs_nextseg;
	/* TODO: make this random */
	segsum->ss_ident = 249755386;
	segsum->ss_nfinfo = 0;
	segsum->ss_ninos = 0;
	segsum->ss_flags = SS_RFW;
	segsum->ss_reclino = 0;
	segsum->ss_serial++;

	fs->seg.fip = (FINFO *)((uint64_t)segsum + sizeof(struct segsum32));

	segusage = SEGUSE_GET(fs, fs->seg.seg_number);
	segusage->su_flags |= SEGUSE_ACTIVE | SEGUSE_DIRTY;
	/* One seg. summary per segment. */
	segusage->su_nsums = 1;

	/* Make a hole for the segment summary. */
	/* TODO: make sure there is no superblock here. */
	ret = _advance_log(fs, fs->lfs.dlfs_sumsize / DFL_LFSBLOCK);
	if (ret != 0)
		return ret;
	assert(fs->seg.disk_bno < fs->lfs.dlfs_offset);
	fs->lfs.dlfs_dmeta++;

	assert(fs->lfs.dlfs_offset >= fs->lfs.dlfs_curseg);
	if (fs->lfs.dlfs_curseg == 0)
		assert((fs->lfs.dlfs_offset - 3) % fs->lfs.dlfs_fsbpseg == 0);
	else
		assert(((fs->lfs.dlfs_offset - 1) % fs->lfs.dlfs_fsbpseg == 0) ||
		       ((fs->lfs.dlfs_offset - 2) % fs->lfs.dlfs_fsbpseg == 0));

	return 0;
}

int write_segment_summary(struct fs *fs) {
	size_t sumstart = offsetof(SEGSUM32, ss_datasum);
	struct segsum32 *ssp;
	ssp = (struct segsum32 *)fs->seg.segsum;

	ssp->ss_create = time(0);
	ssp->ss_datasum = cksum(fs->seg.data_for_cksum,
					fs->seg.cksum_idx * sizeof(int32_t));
	ssp->ss_sumsum = cksum((char *)fs->seg.segsum + sumstart,
			       fs->lfs.dlfs_sumsize - sumstart);

	return write_log(fs, ssp, DFL_LFSBLOCK, FSBLOCK_TO_BYTES(fs->seg.disk_bno), 0);
}

/* Advance the log by nr FS blocks. */
int advance_log_by_one(struct fs *fs, struct _ifile *ifile) {
	int ret;

	assert(fs->lfs.dlfs_offset >= fs->lfs.dlfs_curseg);
	if ((fs->lfs.dlfs_offset - fs->lfs.dlfs_curseg + 1) <
	    fs->lfs.dlfs_fsbpseg) {
		ret = _advance_log(fs, 1);
		if (ret != 0)
			return ret;
	} else {
		assert(((fs->lfs.dlfs_offset + 1) % fs->lfs.dlfs_fsbpseg) == 0);
		write_segment_summary(fs);
		ret = _advance_log(fs, 1);
		if (ret != 0)
			return ret;
		assert(fs->lfs.dlfs_offset % fs->lfs.dlfs_fsbpseg == 0);
		ret = start_segment(fs, ifile);
		if (ret != 0)
			return ret;
	}
	assert(fs->lfs.dlfs_offset >= fs->lfs.dlfs_curseg);

	return 0;
}

/* Advance the log by nr FS blocks. */
int advance_log(struct fs *fs, struct _ifile *ifile, uint32_t nr) {
	uint32_t i, prev;
	int ret;

	assert(fs->lfs.dlfs_offset >= fs->lfs.dlfs_curseg);
	prev = fs->lfs.dlfs_offset;
	for (i = 0; i < nr; i++) {
		ret = advance_log_by_one(fs, ifile);
		if (ret != 0)
			return ret;
	}
	assert(fs->lfs.dlfs_offset >= fs->lfs.dlfs_curseg);
	assert(fs->lfs.dlfs_offset > prev);

	return 0;
}

int dir_add_entry(struct directory *dir, char *name, int inumber, int type) {
	int namlen = strnlen(name, LFS_MAXNAMLEN);
	int reclen = namlen + sizeof(struct lfs_dirheader32);

	/*
	 * The record length is always 4-byte aligned:
	 * The directory entry header structure (struct lfs_dirheader) is just
	 * the header information. A complete entry is this plus a null-
	 * terminated name following it, plus some amount of padding. The
	 * length of the name (not including the null terminator) is given by
	 * the namlen field of the header; the complete record length,
	 * including the null terminator and padding, is given by the reclen
	 * field of the header. The record length is always 4-byte aligned.
	 * (Even on 64-bit volumes, the record length is only 4-byte aligned,
	 * not 8-byte.)
	 */

	/* The "+ 1" is for the null terminator. */
	reclen = 4 * ((reclen + 3 + 1) / 4);

	assert(namlen < LFS_MAXNAMLEN);
	assert(reclen < LFS_DIRBLKSIZ);
	assert(dir->curr >= 0);
	assert(reclen % 4 == 0);

	if ((dir->curr % LFS_DIRBLKSIZ + reclen) > LFS_DIRBLKSIZ) {

		/* Round the curlen of the previous entry to LFS_DIRBLKSIZ. */
		if (dir->prev < dir->curr) {
			struct lfs_dirheader32 *prev =
			    (struct lfs_dirheader32 *)&dir->data[dir->prev];
			prev->dh_reclen = LFS_DIRBLKSIZ - (dir->prev % LFS_DIRBLKSIZ);

			assert(prev->dh_reclen <= LFS_DIRBLKSIZ);
			assert((dir->prev + prev->dh_reclen) % LFS_DIRBLKSIZ == 0);
			assert((prev->dh_reclen & 0x3) == 0);
		}

		/* Move this entry to the next BLK. */
		dir->curr += LFS_DIRBLKSIZ - (dir->curr % LFS_DIRBLKSIZ);
		assert(dir->curr % LFS_DIRBLKSIZ == 0);
	}

	if (dir->curr >= DIRSIZE)
		return ENFILE;

	dir->prev = dir->curr;
	struct lfs_dirheader32 d = {.dh_ino = inumber,
				    .dh_reclen = reclen,
				    .dh_type = type,
				    .dh_namlen = namlen};
	memcpy(&dir->data[dir->curr], &d, sizeof(d));
	dir->curr += sizeof(d);
	strcpy(&dir->data[dir->curr], name);
	dir->curr += reclen - sizeof(d);

	assert(dir->curr >= 0);
	if (dir->curr >= DIRSIZE)
		return ENFILE;

	return 0;
}

/* The last directory entry record len has to fill the remaining LFS_DIRBLKSIZ bytes. */
void dir_done(struct directory *dir) {
	assert(dir->curr > 0);
	assert(dir->curr < DIRSIZE);
	
	struct lfs_dirheader32 *prev =
		    (struct lfs_dirheader32 *)&dir->data[dir->prev];
	prev->dh_reclen = LFS_DIRBLKSIZ - (dir->prev % LFS_DIRBLKSIZ);
	dir->curr = dir->prev + prev->dh_reclen;

	assert(prev->dh_reclen <= LFS_DIRBLKSIZ);
	assert(dir->curr % LFS_DIRBLKSIZ == 0);
	assert(((dir->prev % LFS_DIRBLKSIZ) + prev->dh_reclen) % LFS_DIRBLKSIZ == 0);
	assert((prev->dh_reclen & 0x3) == 0);
}

/*
 * Writes one block for the dir data, and one block for the inode.
 */
int write_empty_root_dir(struct fs *fs) {
	struct directory dir;
	int ret;

	memset(&dir, 0, sizeof(struct directory));

	ret = dir_add_entry(&dir, ".", ULFS_ROOTINO, LFS_DT_DIR);
	if (ret != 0)
		return ret;

	ret = dir_add_entry(&dir, "..", ULFS_ROOTINO, LFS_DT_DIR);
	if (ret != 0)
		return ret;

	dir_done(&dir);

	assert(fs->lfs.dlfs_offset == 3);
	assert(dir.curr == LFS_DIRBLKSIZ);
	return write_file(fs, &dir.data[0], dir.curr, ULFS_ROOTINO,
			LFS_IFDIR | 0755, 2, 0);
}

void init_ifile(struct fs *fs) {
	struct dlfs *lfs = &fs->lfs;
	struct _ifile *ifile = &fs->ifile;
	uint32_t i;
	SEGUSE empty_segusage = {.su_nbytes = 0,
				 .su_olastmod = 0,
				 .su_nsums = 0,
				 .su_ninos = 0,
				 .su_flags = SEGUSE_EMPTY,
				 .su_lastmod = 0};

	/* XXX: Artifial limit on max inodes. */
	assert(sizeof(ifile->ifiles) <= DFL_LFSBLOCK);

	uint32_t nblocks =
	    lfs->dlfs_cleansz + lfs->dlfs_segtabsz + IFILE_MAP_SZ;
	ifile->data = calloc(DFL_LFSBLOCK, nblocks);
	assert(ifile->data);

	ifile->cleanerinfo = (struct _cleanerinfo32 *)ifile->data;
	ifile->segusage =
	    (char *)(ifile->data + (lfs->dlfs_cleansz * DFL_LFSBLOCK));
	ifile->ifiles = (char *)((uint64_t)ifile->segusage +
				    (lfs->dlfs_segtabsz * DFL_LFSBLOCK));

	memset(&ifile->ifiles[0], 0, sizeof(IFILE32));

	for (i = 1; i < MAX_INODES; i++) {
		int off = IFILE_OFF(lfs->dlfs_ifpb, i);
		IFILE32 *ifile_i = IFILE_GET(fs, i);
		assert((IFILE32 *)&ifile->ifiles[off] == ifile_i);
		assert(off < (IFILE_MAP_SZ * DFL_LFSBLOCK));
		ifile_i->if_version = 1;
		ifile_i->if_daddr = LFS_UNUSED_DADDR;
		ifile_i->if_nextfree = i + 1;
		ifile_i->if_atime_sec = 0;
		ifile_i->if_atime_nsec = 0;
	}
	assert(IFILE_OFF(lfs->dlfs_ifpb, lfs->dlfs_ifpb) == DFL_LFSBLOCK);
	assert(IFILE_OFF(lfs->dlfs_ifpb, lfs->dlfs_ifpb + 1) ==
			DFL_LFSBLOCK + sizeof(IFILE32));
	assert((IFILE32 *)&ifile->ifiles[DFL_LFSBLOCK] ==
			IFILE_GET(fs, lfs->dlfs_ifpb));
	assert(IFILE_OFF(lfs->dlfs_ifpb, 2 * lfs->dlfs_ifpb) == 2 * DFL_LFSBLOCK);
	assert(IFILE_OFF(lfs->dlfs_ifpb, 3 * lfs->dlfs_ifpb) == 3 * DFL_LFSBLOCK);
	assert(IFILE_OFF(lfs->dlfs_ifpb, 4 * lfs->dlfs_ifpb) == 4 * DFL_LFSBLOCK);
	assert((IFILE32 *)&ifile->ifiles[4 * DFL_LFSBLOCK] ==
			IFILE_GET(fs, 4 * lfs->dlfs_ifpb));
	if (IFILE_MAP_SZ > 4)
		assert(IFILE_GET(fs, 4 * lfs->dlfs_ifpb)->if_version == 1);

	ifile->cleanerinfo->free_head = 1;
	ifile->cleanerinfo->free_tail = MAX_INODES - 1;

	for (i = 0; i < fs->nsegs; i++) {
		int off = SEGUSE_OFF(lfs->dlfs_sepb, i);
		assert((SEGUSE *)&ifile->segusage[off] == SEGUSE_GET(fs, i));
		memcpy(&ifile->segusage[off], &empty_segusage,
		       sizeof(empty_segusage));
	}
}

void init_sboffs(struct fs *fs, struct _ifile *ifile) {
	struct dlfs *lfs = &fs->lfs;
	uint32_t i, j;
	uint32_t sb_interval; /* number of segs between super blocks */
	SEGUSE *segusage;

	if ((sb_interval = fs->nsegs / LFS_MAXNUMSB) < LFS_MIN_SBINTERVAL)
		sb_interval = LFS_MIN_SBINTERVAL;

	for (i = j = 0; i < fs->nsegs; i++) {
		if (i == 0) {
			segusage = SEGUSE_GET(fs, i);
			segusage->su_flags = SEGUSE_SUPERBLOCK;
			lfs->dlfs_sboffs[j] = 1;
			++j;
		}
		if (i > 0) {
			if ((i % sb_interval) == 0 && j < LFS_MAXNUMSB) {
				segusage = SEGUSE_GET(fs, i);
				segusage->su_flags = SEGUSE_SUPERBLOCK;
				lfs->dlfs_sboffs[j] = i * lfs->dlfs_fsbpseg;
				++j;
			}
		}
	}
}

void add_finfo_inode(struct fs *fs, uint64_t size, uint32_t inumber) {
	struct segment *seg = &fs->seg;
	uint32_t nblocks = (size + DFL_LFSBLOCK - 1) / DFL_LFSBLOCK;
	struct finfo32 *finfo = (struct finfo32 *)seg->fip;
	uint32_t i;

	finfo->fi_nblocks = nblocks;
	finfo->fi_version = 1;
	finfo->fi_ino = inumber;
	finfo->fi_lastlength = DFL_LFSBLOCK;
	seg->fip = (FINFO *)((uint64_t)seg->fip + sizeof(struct finfo32));
	IINFO32 *blocks = (IINFO32 *)seg->fip;
	for (i = 0; i < finfo->fi_nblocks; i++) {

		uint64_t tip = (uint64_t)seg->fip - (uint64_t)seg->segsum;
		if (tip > fs->lfs.dlfs_sumsize) {
			/*
			 * TODO: we should write the remaining blocks into the
			 * next segment.
			 */
			break;
		}

		blocks[i].ii_block = i;
		seg->fip = (FINFO *)((uint64_t)seg->fip + sizeof(IINFO32));
	}

	((struct segsum32 *)seg->segsum)->ss_ninos++;
	((struct segsum32 *)seg->segsum)->ss_nfinfo++;
}

/* Calculate the number of indirect blocks for a file of size (size) */
uint32_t num_iblocks(int32_t nblocks) {
	uint32_t res = 1;

	/* this can be negative (it's fine) */
	nblocks -= ULFS_NDADDR;

	if (nblocks > (NPTR32 * NPTR32 * NPTR32))
		res += DIV_UP(nblocks, NPTR32 * NPTR32 * NPTR32);
	if (nblocks > (NPTR32 * NPTR32))
		res += DIV_UP(nblocks, NPTR32 * NPTR32);
	if (nblocks > (NPTR32))
		res += DIV_UP(nblocks, NPTR32);
	if (nblocks > 0)
		res += 1;

	return res;
}

/*
 * Writes the block pointers and return the offset of the parent.
 */
int write_single_indirect(struct fs *fs, struct _ifile *ifile, int *blk_ptrs,
			uint32_t nblocks, int32_t *off,
			struct lfs32_dinode *inode) {
	SEGUSE *segusage;
	int ret;

	*off = fs->lfs.dlfs_offset;

	assert(nblocks <= NPTR32);

	ret = write_log(fs, blk_ptrs, DFL_LFSBLOCK, FSBLOCK_TO_BYTES(fs->lfs.dlfs_offset), 0);
	if (ret != 0)
		return ret;

	segment_add_datasum(&fs->seg, (char *)blk_ptrs, DFL_LFSBLOCK);
	segusage = SEGUSE_GET(fs, fs->seg.seg_number);
	segusage->su_nbytes += DFL_LFSBLOCK;
	// XXX: take care of failing advance_log
	ret = advance_log(fs, ifile, 1);
	if (ret != 0)
		return ret;
	inode->di_blocks++;

	return 0;
}

/*
 * Writes the block pointers and return the offset of the parent.
 */
int write_double_indirect(struct fs *fs, struct _ifile *ifile, int *blk_ptrs,
			  uint32_t nblocks, int32_t *off,
			  struct lfs32_dinode *inode) {
	int iblks[NPTR32];
	uint32_t i;
	assert(nblocks <= NPTR32 * NPTR32);
	SEGUSE *segusage;
	int ret;

	memset(iblks, 0, DFL_LFSBLOCK);

	for (i = 0; nblocks > 0; i++) {
		uint32_t _nblocks = MIN(nblocks, NPTR32);
		assert(i < NPTR32);
		ret = write_single_indirect(fs, ifile, blk_ptrs, _nblocks, &iblks[i], inode);
		if (ret != 0)
			return ret;
		nblocks -= _nblocks;
		blk_ptrs += _nblocks;
	}

	assert(nblocks == 0);

	*off = fs->lfs.dlfs_offset;

	ret = write_log(fs, iblks, DFL_LFSBLOCK,
			FSBLOCK_TO_BYTES(fs->lfs.dlfs_offset), 0);
	if (ret != 0)
		return ret;
	segment_add_datasum(&fs->seg, (char *)iblks, DFL_LFSBLOCK);
	segusage = SEGUSE_GET(fs, fs->seg.seg_number);
	segusage->su_nbytes += DFL_LFSBLOCK;
	// XXX: take care of failing advance_log
	ret = advance_log(fs, ifile, 1);
	if (ret != 0)
		return ret;
	inode->di_blocks++;

	return 0;
}

/*
 * Writes the block pointers and return the offset of the parent.
 */
int write_triple_indirect(struct fs *fs, struct _ifile *ifile, int *blk_ptrs,
			  uint32_t nblocks, int32_t *off,
			  struct lfs32_dinode *inode) {
	int iblks[NPTR32];
	uint32_t i;
	int ret;

	assert(nblocks <= NPTR32 * NPTR32 * NPTR32);
	SEGUSE *segusage;

	memset(iblks, 0, DFL_LFSBLOCK);

	for (i = 0; nblocks > 0; i++) {
		uint32_t _nblocks = MIN(nblocks, NPTR32 * NPTR32);
		assert(i < NPTR32);
		ret = write_double_indirect(fs, ifile, blk_ptrs, _nblocks, &iblks[i], inode);
		if (ret != 0)
			return ret;
		nblocks -= _nblocks;
		blk_ptrs += _nblocks;
	}

	assert(nblocks == 0);

	*off = fs->lfs.dlfs_offset;

	ret = write_log(fs, iblks,
			DFL_LFSBLOCK, FSBLOCK_TO_BYTES(fs->lfs.dlfs_offset), 0);
	if (ret != 0)
		return ret;
	segment_add_datasum(&fs->seg, (char *)iblks, DFL_LFSBLOCK);
	segusage = SEGUSE_GET(fs, fs->seg.seg_number);
	segusage->su_nbytes += DFL_LFSBLOCK;
	// XXX: take care of failing advance_log
	ret = advance_log(fs, ifile, 1);
	if (ret != 0)
		return ret;
	inode->di_blocks++;

	return 0;
}

int write_file(struct fs *fs, char *data, uint64_t size, int inumber, int mode,
		int nlink, int flags) {
	struct _ifile *ifile = &fs->ifile;
	int32_t nblocks = DIV_UP(size, DFL_LFSBLOCK);
	uint32_t i, j;
	int *blk_ptrs;
	int *indirect_blks = calloc(DFL_LFSBLOCK, num_iblocks(nblocks));
	int ret;

	assert(indirect_blks);
	SEGUSE *segusage;

	/*
	 * TODO: We can't enable this at the moment, because the segment size
	 * is limited to 1 block, and that's not enough for large files.
	 */
	add_finfo_inode(fs, size, inumber);
	assert(fs->lfs.dlfs_inopb == 1);
	fs->lfs.dlfs_dmeta++;

	assert(MAXFILESIZE32 > nblocks * DFL_LFSBLOCK);

	/* Write file inode */
	struct lfs32_dinode inode = {
	    .di_mode = mode,
	    .di_nlink = nlink,
	    .di_inumber = inumber,
	    .di_size = size,
	    .di_atime = time(0),
	    .di_atimensec = 0,
	    .di_mtime = time(0),
	    .di_mtimensec = 0,
	    .di_ctime = time(0),
	    .di_ctimensec = 0,
	    .di_db = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	    .di_ib = {0, 0, 0},
	    .di_flags = flags,
	    .di_blocks = nblocks,
	    .di_gen = 1,
	    .di_uid = 0,
	    .di_gid = 0,
	    .di_modrev = 0};

	ifile->cleanerinfo->free_head++;

	off_t pending;
	for (pending = size, i = 0; pending > 0;) {
		assert(i < nblocks);
		off_t avail_blocks, curr_nblocks, len;

		char *curr_blk = data + (DFL_LFSBLOCK * i);
		avail_blocks = fs->lfs.dlfs_fsbpseg;
		avail_blocks -= fs->lfs.dlfs_offset - fs->lfs.dlfs_curseg;
		assert(avail_blocks > 0 && avail_blocks < fs->lfs.dlfs_fsbpseg);

		len = MIN(pending, avail_blocks * DFL_LFSBLOCK);
		curr_nblocks = DIV_UP(len, DFL_LFSBLOCK);
		assert(len <= avail_blocks * DFL_LFSBLOCK && len > 0);
		assert(curr_nblocks <= avail_blocks && curr_nblocks > 0);

		segment_add_datasum(&fs->seg, curr_blk, len);

		write_log(fs, curr_blk, len,
			FSBLOCK_TO_BYTES(fs->lfs.dlfs_offset),
			mode & LFS_IFREG ? 1 : 0);

		for (j = 0; j < curr_nblocks; j++, i++) {
			if (i < ULFS_NDADDR) {
				inode.di_db[i] = fs->lfs.dlfs_offset + j;
			} else {
				indirect_blks[i - ULFS_NDADDR] = fs->lfs.dlfs_offset + j;
			}
		}

		segusage = SEGUSE_GET(fs, fs->seg.seg_number);
		segusage->su_nbytes += curr_nblocks * DFL_LFSBLOCK;
		ret = advance_log(fs, ifile, curr_nblocks);
		if (ret != 0)
			return ret;

		pending -= len;
	}

	nblocks -= MIN(nblocks, ULFS_NDADDR);
	assert(nblocks >= 0);
	blk_ptrs = indirect_blks;

	if (nblocks > 0) {
		uint32_t _nblocks = MIN(nblocks, NPTR32);
		ret = write_single_indirect(fs, ifile, blk_ptrs, _nblocks,
					&inode.di_ib[0], &inode);
		if (ret != 0)
			return ret;
		nblocks -= _nblocks;
		blk_ptrs += _nblocks;
	}

	if (nblocks > 0) {
		uint32_t _nblocks = MIN(nblocks, NPTR32 * NPTR32);
		ret = write_double_indirect(fs, ifile, blk_ptrs, _nblocks,
					&inode.di_ib[1], &inode);
		if (ret != 0)
			return ret;
		nblocks -= _nblocks;
		blk_ptrs += _nblocks;
	}

	if (nblocks > 0) {
		uint32_t _nblocks = MIN(nblocks, NPTR32 * NPTR32 * NPTR32);
		ret = write_triple_indirect(fs, ifile, blk_ptrs, _nblocks,
					&inode.di_ib[2], &inode);
		if (ret != 0)
			return ret;
		nblocks -= _nblocks;
		blk_ptrs += _nblocks;
	}

	assert(nblocks == 0);

	/* Write the inode */
	ret = write_log(fs, &inode, sizeof(inode),
			FSBLOCK_TO_BYTES(fs->lfs.dlfs_offset), 0);
	if (ret != 0)
		return ret;

	assert(inumber < MAX_INODES);
	
	IFILE32 *ifile_i = IFILE_GET(fs, inumber);
	/* we should be writing this for the first time */
	assert(ifile_i->if_daddr == LFS_UNUSED_DADDR);
	ifile_i->if_daddr = fs->lfs.dlfs_offset;
	ifile_i->if_nextfree = 0;
	segment_add_datasum(&fs->seg, (char *)&inode, DFL_LFSBLOCK);
	segusage = SEGUSE_GET(fs, fs->seg.seg_number);
	segusage->su_ninos += 1;
	segusage->su_nbytes += DFL_LFSBLOCK;
	ret = advance_log(fs, ifile, 1);
	if (ret != 0)
		return ret;

	if (inumber > fs->lfs.dlfs_freehd)
		fs->lfs.dlfs_freehd = inumber;

	free(indirect_blks);

	return 0;
}

/*
 * The difference with write_file is that for an ifile, the inode
 * is written first.
 */
int write_ifile_content(struct fs *fs, struct _ifile *ifile,
			 uint32_t nblocks) {
	uint32_t i;
	off_t inode_lbn;
	int indirect_blk[DFL_LFSBLOCK / sizeof(int)];
	int inumber = LFS_IFILE_INUM;
	int ret;

	add_finfo_inode(fs, nblocks * DFL_LFSBLOCK, inumber);
	assert(fs->lfs.dlfs_inopb == 1);
	fs->lfs.dlfs_dmeta++;

	/* TODO: only have single indirect disk blocks */
	assert(nblocks <= ULFS_NDADDR + NPTR32);
	assert(MAXFILESIZE32 > nblocks * DFL_LFSBLOCK);

	/* Write ifile inode */
	struct lfs32_dinode inode = {
	    .di_mode = LFS_IFREG | 0600,
	    .di_nlink = 1,
	    .di_inumber = LFS_IFILE_INUM,
	    .di_size = nblocks * DFL_LFSBLOCK,
	    .di_atime = time(0),
	    .di_atimensec = 0,
	    .di_mtime = time(0),
	    .di_mtimensec = 0,
	    .di_ctime = time(0),
	    .di_ctimensec = 0,
	    .di_db = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	    .di_ib = {0, 0, 0},
	    .di_flags = SF_IMMUTABLE,
	    .di_blocks = nblocks,
	    .di_gen = 1,
	    .di_uid = 0,
	    .di_gid = 0,
	    .di_modrev = 0};

	ifile->cleanerinfo->free_head++;

	IFILE32 *ifile_i = IFILE_GET(fs, inumber);
	ifile_i->if_daddr = fs->lfs.dlfs_offset;
	ifile_i->if_nextfree = 0;
	inode_lbn = fs->lfs.dlfs_offset;
	segment_add_datasum(&fs->seg, (char *)&inode, DFL_LFSBLOCK);

	/* This block is accounted for the inode. */
	ret = advance_log(fs, ifile, 1);
	if (ret != 0)
		return ret;

	for (i = 0; i < nblocks; i++) {
		char *curr_blk = ifile->data + (DFL_LFSBLOCK * i);
		segment_add_datasum(&fs->seg, curr_blk, DFL_LFSBLOCK);
		write_log(fs, curr_blk, DFL_LFSBLOCK, FSBLOCK_TO_BYTES(fs->lfs.dlfs_offset), 0);

		if (i < ULFS_NDADDR) {
			inode.di_db[i] = fs->lfs.dlfs_offset;
		} else {
			indirect_blk[i - ULFS_NDADDR] = fs->lfs.dlfs_offset;
		}
		/* Adding segusage[fs->seg.seg_number].su_nbytes here has no
		effect,
		as the segusage block is being written here as well.*/
		ret = advance_log(fs, ifile, 1);
		if (ret != 0)
			return ret;
	}

	nblocks -= MIN(nblocks, ULFS_NDADDR);

	if (nblocks > 0) {
		uint32_t _nblocks = MIN(nblocks, NPTR32);
		assert(_nblocks <= NPTR32);
		inode.di_ib[0] = fs->lfs.dlfs_offset;
		ret = write_log(fs, indirect_blk, DFL_LFSBLOCK,
				FSBLOCK_TO_BYTES(fs->lfs.dlfs_offset), 0);
		if (ret != 0)
			return ret;
		segment_add_datasum(&fs->seg, (char *)indirect_blk, DFL_LFSBLOCK);
		ret = advance_log(fs, ifile, 1);
		if (ret != 0)
			return ret;
		nblocks -= _nblocks;
		inode.di_blocks++;
	}
	assert(nblocks == 0);

	/* Write the inode (and indirect block) */
	ret = write_log(fs, &inode, sizeof(inode), FSBLOCK_TO_BYTES(inode_lbn), 0);
	if (ret != 0)
		return ret;

	return 0;
}

int write_ifile(struct fs *fs) {
	int nblocks = fs->lfs.dlfs_cleansz + fs->lfs.dlfs_segtabsz + IFILE_MAP_SZ;
	int all_blocks;
	struct _ifile *ifile = &fs->ifile;
	SEGUSE *segusage;
	int avail_blocks;
	int curr_seg;
	int ret;

	avail_blocks = fs->lfs.dlfs_fsbpseg;
	avail_blocks -= fs->lfs.dlfs_offset - fs->lfs.dlfs_curseg;
	assert(avail_blocks > 0 && avail_blocks < fs->lfs.dlfs_fsbpseg);

	/* Having the ifile span two segments is kind of tricky. So,
	 * if we can't fit it into the current segment, just advance
	 * to the next one. */
	if (nblocks > avail_blocks) {
		uint32_t curr = fs->seg.seg_number;
		while (fs->seg.seg_number == curr) {
			ret = advance_log_by_one(fs, ifile);
			if (ret != 0)
				return ret;
		}
	}

	segusage = SEGUSE_GET(fs, fs->seg.seg_number);
	segusage->su_ninos += 1;

	/* Every segment has a counter of used bytes (su_nbytes), which
	 * is written as part of the ifile. The ifile itself uses some bytes,
	 * so we have to update the counter before writing the ifile.
	 */
	all_blocks = nblocks + 1; /* + 1 for the inode */
	all_blocks += nblocks > ULFS_NDADDR ? 1 : 0; /* indirect block */
	for (curr_seg = fs->seg.seg_number; all_blocks > 0; curr_seg++) {
		segusage = SEGUSE_GET(fs, curr_seg);
		segusage->su_nbytes += DFL_LFSBLOCK * MIN(all_blocks,
						  fs->lfs.dlfs_fsbpseg - 1);
		all_blocks -= MIN(all_blocks, fs->lfs.dlfs_fsbpseg - 1);
	}

	/* point to ifile inode */
	fs->lfs.dlfs_idaddr = fs->lfs.dlfs_offset;

	IFILE32 *ifile_i = IFILE_GET(fs, LFS_IFILE_INUM);
	ifile_i->if_daddr = fs->lfs.dlfs_idaddr;
	ifile_i->if_nextfree = 0;

	/* IFILE/CLEANER INFO */
	ifile->cleanerinfo->clean = fs->lfs.dlfs_nclean;
	ifile->cleanerinfo->dirty = fs->lfs.dlfs_curseg + 1;
	ifile->cleanerinfo->bfree = fs->lfs.dlfs_bfree;
	ifile->cleanerinfo->avail = fs->lfs.dlfs_avail;
	assert(ifile->cleanerinfo->free_tail == (MAX_INODES - 1));
	assert(fs->lfs.dlfs_cleansz == 1);

	/* IFILE/SEGUSE */
	segusage = SEGUSE_GET(fs, fs->seg.seg_number);
	assert(segusage->su_nsums == 1);
	assert(segusage->su_lastmod == 0);
	assert((fs->nsegs * sizeof(SEGUSE)) <
	       (fs->lfs.dlfs_segtabsz * DFL_LFSBLOCK));

	/* IFILE/INODE MAP */
	return write_ifile_content(fs, ifile, nblocks);
}

int init_lfs(struct fs *fs, uint64_t nbytes) {
	uint64_t resvseg;
	struct dlfs *lfs = &fs->lfs;
	uint64_t nsegs;
	int ret;

	fs->lfs = dlfs32_default;

	fs->nbytes = nbytes;
	fs->nsegs = nsegs = ((fs->nbytes / DFL_LFSSEG) - 1);
	resvseg = (((nsegs / DFL_MIN_FREE_SEGS) / 2) + 1);

	lfs->dlfs_size = nbytes / DFL_LFSBLOCK;
	lfs->dlfs_dsize = ((uint64_t)(nsegs - nsegs / DFL_MIN_FREE_SEGS) *
			       (uint64_t)DFL_LFSSEG -
			   DFL_LFSBLOCK * (uint64_t)NSUPERBLOCKS) /
			  DFL_LFSBLOCK;
	lfs->dlfs_lastseg = (nbytes - 2 * (uint64_t)DFL_LFSSEG) / DFL_LFSBLOCK;
	lfs->dlfs_bfree = ((nsegs - nsegs / DFL_MIN_FREE_SEGS) * DFL_LFSSEG -
			   DFL_LFSBLOCK * NSUPERBLOCKS) /
			  DFL_LFSBLOCK;
	lfs->dlfs_avail =
	    SEGS_TO_FSBLOCKS((nbytes / (uint64_t)DFL_LFSSEG) - resvseg) -
	    NSUPERBLOCKS;
	lfs->dlfs_nseg = nsegs;
	lfs->dlfs_segtabsz = ((nsegs + DFL_LFSBLOCK / sizeof(SEGUSE) - 1) /
			      (DFL_LFSBLOCK / sizeof(SEGUSE)));

	/*
	 * write_ifile() currently doesn't support writing an ifile that spans
	 * more than one segment. Check that we won't get into that situation
	 * The "1 + 2" are for the inode, a segment summary, and a potential
	 * superblock.
	 */
	int nblocks = fs->lfs.dlfs_cleansz + fs->lfs.dlfs_segtabsz + 1 + 2;
	assert(nblocks < fs->lfs.dlfs_fsbpseg);

	if (lfs->dlfs_lastseg >= SEGS_TO_FSBLOCKS(nsegs))
		return ENOSPC;

	lfs->dlfs_nclean = nsegs;
	lfs->dlfs_minfreeseg = (nsegs / DFL_MIN_FREE_SEGS);
	lfs->dlfs_resvseg = resvseg;

	/* This mem is freed at exit time. */
	assert(lfs->dlfs_sumsize >= DFL_LFSBLOCK);
	assert(lfs->dlfs_sumsize % DFL_LFSBLOCK == 0);
	fs->seg.segsum = calloc(1, lfs->dlfs_sumsize);
	assert(fs->seg.segsum);

	/* XXX: These make things a lot simpler. */
	assert(DFL_LFSFRAG == DFL_LFSBLOCK);
	assert(fs->lfs.dlfs_fsbpseg > (2 + 6 + 2));
	assert(fs->lfs.dlfs_fsbpseg < MAX_BLOCKS_PER_SEG);
	assert(fs->lfs.dlfs_cleansz == 1);

	struct _ifile *ifile = &fs->ifile;

	init_ifile(fs);
	init_sboffs(fs, ifile);

	/* XXX: start_segment starts by advancing seg_number and dlfs_curseg */
	fs->lfs.dlfs_curseg = (-1) * (DFL_LFSSEG / DFL_LFSBLOCK);
	fs->lfs.dlfs_nextseg = 0;
	fs->seg.seg_number = -1;
	/* The first block is left empty */
	ret = _advance_log(fs, 1);
	if (ret != 0)
		return ret;

	assert(fs->lfs.dlfs_offset == 1);
	ret = start_segment(fs, ifile);
	if (ret != 0)
		return ret;

	return 0;
}

int finish_lfs(struct fs *fs)
{
	int ret;

	ret = write_ifile(fs);
	if (ret != 0)
		return ret;

	ret = write_superblock(fs);
	if (ret != 0)
		return ret;

	ret = write_segment_summary(fs);
	if (ret != 0)
		return ret;

	return 0;
}
