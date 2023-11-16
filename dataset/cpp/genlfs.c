#include <assert.h>
#include <dirent.h>
#include <err.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/mman.h>

#include "config.h"
#include "lfs.h"

static int next_inum = 4;

int get_next_inum(void) { return ++next_inum; }

/*
   DT_BLK      This is a block device.
   DT_CHR      This is a character device.
   DT_DIR      This is a directory.
   DT_FIFO     This is a named pipe (FIFO).
   DT_LNK      This is a symbolic link.
   DT_REG      This is a regular file.
   DT_SOCK     This is a UNIX domain socket.
   DT_UNKNOWN  The file type is unknown.
   */

void walk(struct fs *fs, int parent_inum, int inum) {
	DIR *d;
	struct dirent *dirent;
	struct directory *dir = calloc(1, sizeof(struct directory));
	assert(dir);

	d = opendir(".");

	if (d == NULL) {
		return;
	}

	while ((dirent = readdir(d)) != NULL) {
		struct stat sb;

		lstat(dirent->d_name, &sb);

		switch (sb.st_mode & S_IFMT) {
		case S_IFBLK:
			printf("block device\n");
			break;
		case S_IFCHR:
			printf("character device\n");
			break;
		case S_IFDIR:
			if (strcmp(dirent->d_name, ".") == 0)
				break;
			if (strcmp(dirent->d_name, "..") == 0)
				break;
			if (strcmp(dirent->d_name, "dev") == 0)
				break;
			if (strcmp(dirent->d_name, "sys") == 0)
				break;
			if (strcmp(dirent->d_name, "proc") == 0)
				break;
			int next_inum = get_next_inum();
			assert(dir_add_entry(dir, dirent->d_name, next_inum,
				      LFS_DT_DIR) == 0);
			printf("directory (%d): %s\n", next_inum, dirent->d_name);
			if (chdir(dirent->d_name) != 0)
				errx(1, "Failed to chdir: %s", dirent->d_name);
			walk(fs, inum, next_inum);
			if (chdir("..") != 0)
				errx(1, "Failed to chdir: ..");
			break;
		case S_IFIFO:
			printf("FIFO/pipe\n");
			break;
		case S_IFLNK:
			printf("symlink\n");
			break;
		case S_IFREG: {
			int fd = openat(AT_FDCWD, dirent->d_name, O_RDONLY);
			assert(fd > 0);
			void *addr = NULL;
			if (sb.st_size > 0) {
				addr = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
				assert(addr != MAP_FAILED);
			}
			int next_inum = get_next_inum();
			printf("regular file (%d): %s\n", next_inum, dirent->d_name);
			write_file(fs, (char *)addr, sb.st_size, next_inum,
					   LFS_IFREG | 0777, 1, 0);
			munmap(addr, sb.st_size);
			close(fd);

			assert(dir_add_entry(dir, dirent->d_name, next_inum,
				      LFS_DT_REG) == 0);
			break;
		}
		case S_IFSOCK:
			printf("socket\n");
			break;
		default:
			printf("unknown?\n");
			break;
		}
	}

	dir_add_entry(dir, ".", inum, LFS_DT_DIR);
	dir_add_entry(dir, "..", parent_inum, LFS_DT_DIR);
	dir_done(dir);

	/* TODO: nlinks should be 2 for root. What about others (does ..
	 * count)? */
	write_file(fs, dir->data, dir->curr, inum, LFS_IFDIR | 0755, 1, 0);
	free(dir);

	closedir(d);
}

int main(int argc, char **argv) {
	struct fs fs;
	uint64_t nbytes = 1024 * 1024 * 1024 * 4ULL;

	if (argc != 3) {
		errx(1, "Usage: %s <directory> <image>", argv[0]);
	}

	fs.fd = open(argv[2], O_CREAT | O_RDWR, DEFFILEMODE);
	assert(fs.fd != 0);

	init_lfs(&fs, nbytes);

	if (chdir(argv[1]) != 0)
		return 1;

	walk(&fs, ULFS_ROOTINO, ULFS_ROOTINO);

	finish_lfs(&fs);
	close(fs.fd);

	return 0;
}
