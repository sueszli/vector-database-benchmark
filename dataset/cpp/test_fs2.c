/* SPDX-License-Identifier: BSD-2-Clause */

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>
#include <fcntl.h>
#include <time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <dirent.h>

#include "devshell.h"
#include "sysenter.h"
#include "test_common.h"

void create_test_file1(void);
void write_on_test_file1(void);

/* Test truncate() */
int cmd_fs7(int argc, char **argv)
{
   int rc;

   create_test_file1();
   write_on_test_file1();
   rc = truncate("/tmp/test1", 157);
   DEVSHELL_CMD_ASSERT(rc == 0);

   // TODO: add checks here to verify that the truncation worked as expected

   rc = unlink("/tmp/test1");
   DEVSHELL_CMD_ASSERT(rc == 0);
   return 0;
}

static const char test_str[] = "this is a test string\n";
static const char test_str2[] = "hello from the 2nd page";
static const char test_str_exp[] = "This is a test string\n";
static const char test_file[] = "/tmp/test1";

void do_mm_read(void *ptr)
{
   unsigned value;
   printf("[pid: %d] Before read at %p\n", getpid(), ptr);
   memcpy(&value, ptr, sizeof(value));
   printf("Read OK. Value at %p: %#x\n", ptr, value);
}


/* mmap file */
int cmd_fmmap1(int argc, char **argv)
{
   int fd, rc;
   char *vaddr;
   char buf[64];
   size_t file_size;
   struct stat statbuf;
   const size_t page_size = getpagesize();

   printf("- Using '%s' as test file\n", test_file);
   fd = open(test_file, O_CREAT | O_RDWR, 0644);
   DEVSHELL_CMD_ASSERT(fd > 0);

   rc = write(fd, test_str, sizeof(test_str)-1);
   DEVSHELL_CMD_ASSERT(rc == sizeof(test_str)-1);

   rc = fstat(fd, &statbuf);
   DEVSHELL_CMD_ASSERT(rc == 0);

   file_size = statbuf.st_size;
   printf("- File size: %llu\n", (ull_t)file_size);

   vaddr = mmap(NULL,                   /* addr */
                2 * page_size,          /* length */
                PROT_READ | PROT_WRITE, /* prot */
                MAP_SHARED,             /* flags */
                fd,                     /* fd */
                0);

   if (vaddr == (void *)-1)
      goto err_case;

   printf("- vaddr: %p\n", vaddr);

   printf("- Check that reading at `vaddr` succeeds\n");
   do_mm_read(vaddr);

   printf("- Check we can fork() a process with ramfs mmaps\n");
   if (test_sig(do_mm_read, vaddr, 0, 0, 0))
      goto err_case;

   printf("- Check that reading at `vaddr` still succeeds from parent\n");
   do_mm_read(vaddr);

   printf("- Done\n");
   printf("- Write something at `vaddr`...\n");

   vaddr[0] = 'T';                     // has real effect

   printf("- Write something at `vaddr + file_size`\n");
   vaddr[file_size +  0] = '?';        // gets ignored as past of EOF
   vaddr[file_size + 10] = 'x';        // gets ignored as past of EOF
   vaddr[file_size + 11] = '\n';       // gets ignored as past of EOF

   printf("- Check that read mapped area past EOF triggers SIGBUS\n");
   if (test_sig(do_mm_read, vaddr + page_size, SIGBUS, 0, 0))
      goto err_case;

   printf("- Close the file descriptor and re-open the file\n");
   close(fd);

   rc = stat(test_file, &statbuf);
   DEVSHELL_CMD_ASSERT(rc == 0);
   DEVSHELL_CMD_ASSERT(statbuf.st_size == file_size);

   fd = open(test_file, O_RDONLY, 0644);
   DEVSHELL_CMD_ASSERT(fd > 0);

   printf("- Check file's contents\n");
   rc = read(fd, buf, file_size);
   DEVSHELL_CMD_ASSERT(rc == file_size);
   buf[rc] = 0;

   if (strcmp(buf, test_str_exp)) {
      fprintf(stderr, "File contents != expected:\n");
      fprintf(stderr, "Contents: <<\n%s>>\n", buf);
      fprintf(stderr, "Expected: <<\n%s>>\n", test_str_exp);
      DEVSHELL_CMD_ASSERT(false);
   }

   printf("- Re-map and check past-EOF contents\n");

   vaddr = mmap(NULL,                   /* addr */
                2 * page_size,          /* length */
                PROT_READ,              /* prot */
                MAP_SHARED,             /* flags */
                fd,                     /* fd */
                0);

   if (vaddr == (void *)-1)
      goto err_case;

   /*
    * At least on ext4 on Linux, the past-EOF contents are kept. That's the
    * simplest behavior to implement for Tilck as well.
    */
   printf("vaddr[file_size +  0]: %c\n", vaddr[file_size +  0]);
   printf("vaddr[file_size + 10]: %c\n", vaddr[file_size + 10]);

   DEVSHELL_CMD_ASSERT(vaddr[file_size +  0] == '?');
   DEVSHELL_CMD_ASSERT(vaddr[file_size + 10] == 'x');

   printf("DONE\n");
   close(fd);
   rc = unlink(test_file);
   DEVSHELL_CMD_ASSERT(rc == 0);
   return 0;

err_case:

   if (vaddr == (void *)-1)
      fprintf(stderr, "mmap failed: %s\n", strerror(errno));

   close(fd);
   unlink(test_file);
   return 1;
}

/* mmap file and then do a partial unmap */
static void fmmap2_read_unmapped_mem(void *unused_arg)
{
   int fd, rc;
   char *vaddr;
   char buf[64] = {0};
   char *page_size_buf;
   const size_t page_size = getpagesize();

   printf("Using '%s' as test file\n", test_file);
   fd = open(test_file, O_CREAT | O_RDWR, 0644);
   DEVSHELL_CMD_ASSERT(fd > 0);

   page_size_buf = malloc(page_size);
   DEVSHELL_CMD_ASSERT(page_size_buf != NULL);

   for (int i = 0; i < 4; i++) {
      memset(page_size_buf, 'A'+i, page_size);
      rc = write(fd, page_size_buf, page_size);
      DEVSHELL_CMD_ASSERT(rc == page_size);
   }

   /* Now, let's mmap the file */

   vaddr = mmap(NULL,                   /* addr */
                4 * page_size,          /* length */
                PROT_READ,              /* prot */
                MAP_SHARED,             /* flags */
                fd,                     /* fd */
                0);

   DEVSHELL_CMD_ASSERT(vaddr != (void *)-1);

   /* Un-map the 2nd page */
   rc = munmap(vaddr + page_size, page_size);
   DEVSHELL_CMD_ASSERT(rc == 0);

   /* Excepting to receive SIGSEGV from the kernel here */
   memcpy(buf, vaddr + page_size, sizeof(buf) - 1);

   /* ----------- We should NOT get here ------------------- */

   free(page_size_buf);
   close(fd);
   rc = unlink(test_file);
   DEVSHELL_CMD_ASSERT(rc == 0);
   exit(1);
}

/* mmap file and then do a partial unmap */
int cmd_fmmap2(int argc, char **argv)
{
   int rc = test_sig(fmmap2_read_unmapped_mem, NULL, SIGSEGV, 0, 0);
   unlink(test_file);
   return rc;
}

/* mmap file with offset > 0 */
int cmd_fmmap3(int argc, char **argv)
{
   int fd, rc;
   char *vaddr;
   char buf[64] = {0};
   char exp_buf[64] = {0};
   char *page_size_buf;
   const size_t page_size = getpagesize();
   bool failed = false;

   printf("Using '%s' as test file\n", test_file);
   fd = open(test_file, O_CREAT | O_RDWR, 0644);
   DEVSHELL_CMD_ASSERT(fd > 0);

   page_size_buf = malloc(page_size);
   DEVSHELL_CMD_ASSERT(page_size_buf != NULL);

   for (int i = 0; i < 4; i++) {
      memset(page_size_buf, 'A'+i, page_size);
      rc = write(fd, page_size_buf, page_size);
      DEVSHELL_CMD_ASSERT(rc == page_size);
   }

   /* Now, let's mmap the file at offset > 0 */

   vaddr = mmap(NULL,                   /* addr */
                4 * page_size,          /* length */
                PROT_READ,              /* prot */
                MAP_SHARED,             /* flags */
                fd,                     /* fd */
                page_size);             /* offset */

   DEVSHELL_CMD_ASSERT(vaddr != (void *)-1);

   memcpy(buf, vaddr, sizeof(buf) - 1);
   memset(exp_buf, 'B', sizeof(exp_buf)-1);

   if (strcmp(buf, exp_buf)) {
      fprintf(stderr, "Reading vaddr mapped at off > 0 lead to garbage.\n");
      fprintf(stderr, "Expected: '%s'\n", exp_buf);
      fprintf(stderr, "Got:      '%s'\n", buf);
      failed = true;
   }

   free(page_size_buf);
   close(fd);
   rc = unlink(test_file);
   DEVSHELL_CMD_ASSERT(rc == 0);
   return failed;
}

static void fmmap4_read_write_after_eof(bool rw)
{
   int fd, rc;
   char *vaddr;
   size_t file_size;
   char buf[64] = {0};
   struct stat statbuf;
   const size_t page_size = getpagesize();

   printf("fmmap4_read_write_after_eof(%s)\n", rw ? "WRITE" : "READ");
   printf("Using '%s' as test file\n", test_file);
   fd = open(test_file, O_CREAT | O_RDWR, 0644);
   DEVSHELL_CMD_ASSERT(fd > 0);

   rc = write(fd, test_str, sizeof(test_str)-1);
   DEVSHELL_CMD_ASSERT(rc == sizeof(test_str)-1);

   rc = fstat(fd, &statbuf);
   DEVSHELL_CMD_ASSERT(rc == 0);

   file_size = statbuf.st_size;
   printf("File size: %llu\n", (ull_t)file_size);

   vaddr = mmap(NULL,                   /* addr */
                2 * page_size,          /* length */
                PROT_READ | PROT_WRITE, /* prot */
                MAP_SHARED,             /* flags */
                fd,                     /* fd */
                0);

   DEVSHELL_CMD_ASSERT(vaddr != (void *)-1);

   /* Expecting to be killed by SIGBUS here */

   if (!rw) {
      /* Read past EOF */
      forced_memcpy(buf, vaddr + page_size, sizeof(buf) - 1);
   } else {
      /* Write past EOF */
      forced_memcpy(vaddr + page_size, buf, sizeof(buf) - 1);
   }

   /* If we got here, something went wrong */
   printf("ERROR: got to the end, something went wrong\n");
}

static void fmmap4_read_after_eof(void *unused)
{
   fmmap4_read_write_after_eof(false);
}

static void fmmap4_write_after_eof(void *unused)
{
   fmmap4_read_write_after_eof(true);
}

/* mmap file and read past EOF, expecting SIGBUS */
int cmd_fmmap4(int argc, char **argv)
{
   int rc;

   if ((rc = test_sig(fmmap4_read_after_eof, NULL, SIGBUS, 0, 0)))
      goto end;

   if ((rc = test_sig(fmmap4_write_after_eof, NULL, SIGBUS, 0, 0)))
      goto end;

end:
   unlink(test_file);
   return rc;
}

/* mmap file, truncate (expand) it and write past the original EOF */
int cmd_fmmap5(int argc, char **argv)
{
   int fd, rc;
   char *vaddr;
   size_t file_size;
   char buf[64] = {0};
   struct stat statbuf;
   const size_t page_size = getpagesize();
   bool failed = false;

   printf("Using '%s' as test file\n", test_file);
   fd = open(test_file, O_CREAT | O_RDWR, 0644);
   DEVSHELL_CMD_ASSERT(fd > 0);

   rc = write(fd, test_str, sizeof(test_str)-1);
   DEVSHELL_CMD_ASSERT(rc == sizeof(test_str)-1);

   rc = fstat(fd, &statbuf);
   DEVSHELL_CMD_ASSERT(rc == 0);

   file_size = statbuf.st_size;
   printf("File size: %llu\n", (ull_t)file_size);

   vaddr = mmap(NULL,                   /* addr */
                2 * page_size,          /* length */
                PROT_READ | PROT_WRITE, /* prot */
                MAP_SHARED,             /* flags */
                fd,                     /* fd */
                0);

   DEVSHELL_CMD_ASSERT(vaddr != (void *)-1);

   rc = ftruncate(fd, page_size + 128);
   DEVSHELL_CMD_ASSERT(rc == 0);

   rc = fstat(fd, &statbuf);
   DEVSHELL_CMD_ASSERT(rc == 0);

   file_size = statbuf.st_size;
   printf("(NEW) File size: %llu\n", (ull_t)file_size);

   /*
    * This memory write will trigger a page-fault and the kernel should allocate
    * on-the-fly the page (ramfs_block) for us and, ultimately, resume the
    * write.
    */
   strcpy(vaddr + page_size, test_str2);
   close(fd);

   fd = open(test_file, O_CREAT | O_RDONLY, 0644);
   DEVSHELL_CMD_ASSERT(fd > 0);

   /* Check `test_str2` is actually in the file */
   rc = lseek(fd, page_size, SEEK_SET);
   DEVSHELL_CMD_ASSERT(rc == page_size);

   rc = read(fd, buf, sizeof(buf)-1);
   DEVSHELL_CMD_ASSERT(rc == sizeof(buf)-1);

   if (strcmp(buf, test_str2)) {
      fprintf(stderr, "Reading at offset page_size failure!\n");
      fprintf(stderr, "Expected: '%s'\n", test_str2);
      fprintf(stderr, "Got:      '%s'\n", buf);
      failed = true;
   }

   close(fd);
   rc = unlink(test_file);
   DEVSHELL_CMD_ASSERT(rc == 0);
   return failed;
}

/* mmap file, truncate (expand) it and READ past the original EOF */
int cmd_fmmap6(int argc, char **argv)
{
   int fd, rc;
   char *vaddr;
   char buf[64];
   size_t file_size;
   struct stat statbuf;
   const size_t page_size = getpagesize();
   bool failed = false;

   printf("Using '%s' as test file\n", test_file);
   fd = open(test_file, O_CREAT | O_RDWR, 0644);
   DEVSHELL_CMD_ASSERT(fd > 0);

   rc = write(fd, test_str, sizeof(test_str)-1);
   DEVSHELL_CMD_ASSERT(rc == sizeof(test_str)-1);

   rc = fstat(fd, &statbuf);
   DEVSHELL_CMD_ASSERT(rc == 0);

   file_size = statbuf.st_size;
   printf("File size: %llu\n", (ull_t)file_size);

   vaddr = mmap(NULL,                   /* addr */
                2 * page_size,          /* length */
                PROT_READ,              /* prot */
                MAP_SHARED,             /* flags */
                fd,                     /* fd */
                0);

   DEVSHELL_CMD_ASSERT(vaddr != (void *)-1);

   rc = ftruncate(fd, page_size + 128);
   DEVSHELL_CMD_ASSERT(rc == 0);

   rc = fstat(fd, &statbuf);
   DEVSHELL_CMD_ASSERT(rc == 0);

   file_size = statbuf.st_size;
   printf("(NEW) File size: %llu\n", (ull_t)file_size);

   memset(buf, 'X', sizeof(buf));

   /*
    * This memory read is expected trigger a page fault and the kernel to map
    * the zero page to vaddr in read-only mode.
    */
   memcpy(buf, vaddr + page_size, sizeof(buf));

   for (size_t i = 0; i < sizeof(buf); i++) {
      if (buf[i]) {
         fprintf(stderr, "Found a non-zero byte in the past-EOF read\n");
         failed = true;
      }
   }

   close(fd);
   rc = unlink(test_file);
   DEVSHELL_CMD_ASSERT(rc == 0);
   return failed;
}

/* mmap file and then trucate it */
static void fmmap7_child(void *unused_arg)
{
   int fd, rc;
   char *vaddr;
   size_t file_size;
   char buf[64] = {0};
   struct stat statbuf;
   char *page_size_buf;
   const size_t page_size = getpagesize();

   printf("Using '%s' as test file\n", test_file);
   fd = open(test_file, O_CREAT | O_RDWR, 0644);
   DEVSHELL_CMD_ASSERT(fd > 0);

   page_size_buf = malloc(page_size);
   DEVSHELL_CMD_ASSERT(page_size_buf != NULL);

   for (int i = 0; i < 4; i++) {
      memset(page_size_buf, 'A'+i, page_size);
      rc = write(fd, page_size_buf, page_size);
      DEVSHELL_CMD_ASSERT(rc == page_size);
   }

   rc = fstat(fd, &statbuf);
   DEVSHELL_CMD_ASSERT(rc == 0);

   file_size = statbuf.st_size;
   printf("File size: %llu\n", (ull_t)file_size);
   printf("Mmap 4 pages of the file\n");

   vaddr = mmap(NULL,                   /* addr */
                4 * page_size,          /* length */
                PROT_READ,              /* prot */
                MAP_SHARED,             /* flags */
                fd,                     /* fd */
                0);

   DEVSHELL_CMD_ASSERT(vaddr != (void *)-1);

   printf("Read from the 2nd mapped page...\n");

   memcpy(buf, vaddr + page_size, sizeof(buf));
   printf("Got in buf: '%s'\n", buf);
   memset(buf, 0, sizeof(buf));

   printf("Now, trucate the file to just 1 page\n");

   /* Truncate the file using truncate() instead of ftruncate() */
   rc = truncate(test_file, page_size);
   DEVSHELL_CMD_ASSERT(rc == 0);

   rc = fstat(fd, &statbuf);
   DEVSHELL_CMD_ASSERT(rc == 0);

   file_size = statbuf.st_size;
   printf("(NEW) File size: %llu\n", (ull_t)file_size);
   printf("Now, trying to read the 2nd page (mapped, but after EOF)\n");

   /*
    * On Linux, here we get SIGBUS, as if the file was never bigger than that.
    * This means that the pages after EOF have been un-mapped by the truncate()
    * syscall.
    */
   memcpy(buf, vaddr + page_size, sizeof(buf));

   /* ------------- we should never get here ----------------- */

   fprintf(stderr, "ERROR: We should never get here. ");
   fprintf(stderr, "Expected to die due to SIGBUS.\n");
   printf("Got in buf: '%s'\n", buf);

   free(page_size_buf);
   close(fd);
   rc = unlink(test_file);
   DEVSHELL_CMD_ASSERT(rc == 0);
}

/* mmap file and then trucate it */
int cmd_fmmap7(int argc, char **argv)
{
   int rc = test_sig(fmmap7_child, NULL, SIGBUS, 0, 0);
   unlink(test_file);
   return rc;
}
