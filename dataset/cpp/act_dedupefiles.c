/* Deduplication of files with OS-specific copy-on-write mechanisms
 * This file is part of jdupes; see jdupes.c for license information */

#include "jdupes.h"

#ifdef ENABLE_DEDUPE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include "act_dedupefiles.h"
#include "libjodycode.h"

#ifdef __linux__
 /* Use built-in static dedupe header if requested */
 #ifdef STATIC_DEDUPE_H
  #include "linux-dedupe-static.h"
 #else
  #include <linux/fs.h>
 #endif /* STATIC_DEDUPE_H */

 /* If the Linux headers are too old, automatically use the static one */
 #ifndef FILE_DEDUPE_RANGE_SAME
  #warning Automatically enabled STATIC_DEDUPE_H due to insufficient header support
  #include "linux-dedupe-static.h"
 #endif /* FILE_DEDUPE_RANGE_SAME */
 #include <sys/ioctl.h>
 #define JDUPES_DEDUPE_SUPPORTED 1
 #define KERNEL_DEDUP_MAX_SIZE 16777216
 /* Error messages */
 static const char s_err_dedupe_notabug[] = "This is not a bug in jdupes; check your file stats/permissions.";
 static const char s_err_dedupe_repeated[] = "This verbose error description will not be repeated.";
#endif /* __linux__ */

#ifdef __APPLE__
 #ifdef NO_HARDLINKS
 #error Hard link support is required for dedupe on macOS but NO_HARDLINKS was set
 #endif
 #include "act_linkfiles.h"
 #define JDUPES_DEDUPE_SUPPORTED 1
#endif

#ifndef JDUPES_DEDUPE_SUPPORTED
#error Dedupe is only supported on Linux and macOS
#endif

void dedupefiles(file_t * restrict files)
{
#ifdef __linux__
  struct file_dedupe_range *fdr;
  struct file_dedupe_range_info *fdri;
  file_t *curfile, *curfile2, *dupefile;
  int src_fd;
  int err_twentytwo = 0, err_ninetyfive = 0;
  uint64_t total_files = 0;

  LOUD(fprintf(stderr, "\ndedupefiles: %p\n", files);)

  fdr = (struct file_dedupe_range *)calloc(1,
        sizeof(struct file_dedupe_range)
      + sizeof(struct file_dedupe_range_info) + 1);
  fdr->dest_count = 1;
  fdri = &fdr->info[0];
  for (curfile = files; curfile; curfile = curfile->next) {
    /* Skip all files that have no duplicates */
    if (!ISFLAG(curfile->flags, FF_HAS_DUPES)) continue;
    CLEARFLAG(curfile->flags, FF_HAS_DUPES);

    /* For each duplicate list head, handle the duplicates in the list */
    curfile2 = curfile;
    src_fd = open(curfile->d_name, O_RDONLY);
    /* If an open fails, keep going down the dupe list until it is exhausted */
    while (src_fd == -1 && curfile2->duplicates && curfile2->duplicates->duplicates) {
      fprintf(stderr, "dedupe: open failed (skipping): %s\n", curfile2->d_name);
      exit_status = EXIT_FAILURE;
      curfile2 = curfile2->duplicates;
      src_fd = open(curfile2->d_name, O_RDONLY);
    }
    if (src_fd == -1) continue;
    printf("  [SRC] %s\n", curfile2->d_name);

    /* Run dedupe for each set */
    for (dupefile = curfile->duplicates; dupefile; dupefile = dupefile->duplicates) {
      off_t remain;
      int err;

      /* Don't pass hard links to dedupe */
      if (dupefile->device == curfile->device && dupefile->inode == curfile->inode) {
        printf("  -==-> %s\n", dupefile->d_name);
        continue;
      }

      /* Open destination file, skipping any that fail */
      fdri->dest_fd = open(dupefile->d_name, O_RDONLY);
      if (fdri->dest_fd == -1) {
        fprintf(stderr, "dedupe: open failed (skipping): %s\n", dupefile->d_name);
        exit_status = EXIT_FAILURE;
        continue;
      }

      /* Dedupe src <--> dest, 16 MiB or less at a time */
      remain = dupefile->size;
      fdri->status = FILE_DEDUPE_RANGE_SAME;
      /* Consume data blocks until no data remains */
      while (remain) {
        errno = 0;
        fdr->src_offset = (uint64_t)(dupefile->size - remain);
        fdri->dest_offset = fdr->src_offset;
        fdr->src_length = (uint64_t)(remain <= KERNEL_DEDUP_MAX_SIZE ? remain : KERNEL_DEDUP_MAX_SIZE);
        ioctl(src_fd, FIDEDUPERANGE, fdr);
        if (fdri->status < 0) break;
        remain -= (off_t)fdr->src_length;
      }

      /* Handle any errors */
      err = fdri->status;
      if (err != FILE_DEDUPE_RANGE_SAME || errno != 0) {
        printf("  -XX-> %s\n", dupefile->d_name);
        fprintf(stderr, "error: ");
        if (err == FILE_DEDUPE_RANGE_DIFFERS) {
          fprintf(stderr, "not identical (files modified between scan and dedupe?)\n");
          exit_status = EXIT_FAILURE;
        } else if (err != 0) {
          fprintf(stderr, "%s (%d)\n", strerror(-err), err);
          exit_status = EXIT_FAILURE;
        } else if (errno != 0) {
          fprintf(stderr, "%s (%d)\n", strerror(errno), errno);
          exit_status = EXIT_FAILURE;
        }
        if ((err == -22 || errno == 22) && err_twentytwo == 0) {
          fprintf(stderr, "       One or more files being deduped are read-only or hard linked.\n");
          fprintf(stderr, "       Read-only files can only be deduped by the root user.\n");
          fprintf(stderr, "       %s\n", s_err_dedupe_notabug);
          fprintf(stderr, "       %s\n", s_err_dedupe_repeated);
          err_twentytwo = 1;
	}
        if ((err == -95 || errno == 95) && err_ninetyfive == 0) {
          fprintf(stderr, "       One or more files is on a filesystem that does not support\n");
          fprintf(stderr, "       block-level deduplication or are on different filesystems.\n");
          fprintf(stderr, "       %s\n", s_err_dedupe_notabug);
          fprintf(stderr, "       %s\n", s_err_dedupe_repeated);
          err_ninetyfive = 1;
	}
      } else {
        /* Dedupe OK; report to the user and add to file count */
        printf("  ====> %s\n", dupefile->d_name);
        total_files++;
      }
      close((int)fdri->dest_fd);
    }
    printf("\n");
    close(src_fd);
    total_files++;
  }

  if (!ISFLAG(flags, F_HIDEPROGRESS)) fprintf(stderr, "Deduplication done (%" PRIuMAX " files processed)\n", total_files);
  free(fdr);
#endif /* __linux__ */

/* On macOS, clonefile() is basically a "hard link" function, so linkfiles will do the work. */
#ifdef __APPLE__
  linkfiles(files, 2, 0);
#endif /* __APPLE__ */
  return;
}
#endif /* ENABLE_DEDUPE */
