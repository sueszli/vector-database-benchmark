/*
 * Copyright 2005-2011, Ingo Weinhold, ingo_weinhold@gmx.de.
 * Copyright 2012, John Scipione, jscipione@gmail.com.
 * Distributed under the terms of the MIT License.
 */

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>


// Private helper functions
static int get_path(int fd, const char* path, char fullPath[]);
static int eaccess(const char* path, int accessMode);


static int
get_path(int fd, const char* path, char fullPath[])
{
	struct stat dirst;
	if (fstat(fd, &dirst) < 0) {
		// failed to grab stat information, fstat() sets errno
		return -1;
	}

	if (!S_ISDIR(dirst.st_mode)) {
		// fd does not point to a directory
		errno = ENOTDIR;
		return -1;
	}

	if (fcntl(fd, F_GETPATH, fullPath) < 0) {
		// failed to get the path of fd, fcntl() sets errno
		return -1;
	}

	if (strlcat(fullPath, "/", MAXPATHLEN) > MAXPATHLEN
		|| strlcat(fullPath, path, MAXPATHLEN) > MAXPATHLEN) {
		// full path is too long
		errno = ENAMETOOLONG;
		return -1;
	}

	return 0;
}


static int
eaccess(const char* path, int accessMode)
{
	uid_t uid = geteuid();
	int fileMode = 0;

	struct stat st;
	if (stat(path, &st) < 0) {
		// failed to get stat information on path, stat() sets errno
		return -1;
	}

	if (uid == 0) {
		// user is root
		// root has always read/write permission, but at least one of the
		// X bits must be set for execute permission
		fileMode = R_OK | W_OK;
		if ((st.st_mode & (S_IXUSR | S_IXGRP | S_IXOTH)) != 0)
			fileMode |= X_OK;
	} else if (st.st_uid == uid) {
		// user is node owner
		if ((st.st_mode & S_IRUSR) != 0)
			fileMode |= R_OK;
		if ((st.st_mode & S_IWUSR) != 0)
			fileMode |= W_OK;
		if ((st.st_mode & S_IXUSR) != 0)
			fileMode |= X_OK;
	} else if (st.st_gid == getegid()) {
		// user is in owning group
		if ((st.st_mode & S_IRGRP) != 0)
			fileMode |= R_OK;
		if ((st.st_mode & S_IWGRP) != 0)
			fileMode |= W_OK;
		if ((st.st_mode & S_IXGRP) != 0)
			fileMode |= X_OK;
	} else {
		// user is one of the others
		if ((st.st_mode & S_IROTH) != 0)
			fileMode |= R_OK;
		if ((st.st_mode & S_IWOTH) != 0)
			fileMode |= W_OK;
		if ((st.st_mode & S_IXOTH) != 0)
			fileMode |= X_OK;
	}

	if ((accessMode & ~fileMode) != 0) {
		errno = EACCES;
		return -1;
	}

	return 0;
}


int
faccessat(int fd, const char* path, int accessMode, int flag)
{
	if (flag != AT_EACCESS && flag != 0) {
		// invalid flag
		errno = EINVAL;
		return -1;
	}

	if (fd == AT_FDCWD || (path != NULL && path[0] == '/')) {
		// call access() ignoring fd
		return (flag & AT_EACCESS) != 0 ? eaccess(path, accessMode)
			: access(path, accessMode);
	}

	if (fd < 0) {
		// Invalid file descriptor
		errno = EBADF;
		return -1;
	}

	char fullPath[MAXPATHLEN];
	if (get_path(fd, path, fullPath) < 0)
		return -1;

	return (flag & AT_EACCESS) != 0 ? eaccess(fullPath, accessMode)
		: access(fullPath, accessMode);
}


int
fchmodat(int fd, const char* path, mode_t mode, int flag)
{
	if ((flag & AT_SYMLINK_NOFOLLOW) == 0 && flag != 0) {
		// invalid flag
		errno = EINVAL;
		return -1;
	}

	if (fd == AT_FDCWD || (path != NULL && path[0] == '/')) {
		// call chmod() ignoring fd
		if ((flag & AT_SYMLINK_NOFOLLOW) != 0) {
			// fake lchmod() with open() and fchmod()
			int symlinkfd = open(path, O_RDONLY | O_SYMLINK);
			int status = fchmod(symlinkfd, mode);
			close(symlinkfd);
			return status;
		} else
			return chmod(path, mode);
	}

	if (fd < 0) {
		// Invalid file descriptor
		errno = EBADF;
		return -1;
	}

	char fullPath[MAXPATHLEN];
	if (get_path(fd, path, fullPath) < 0)
		return -1;

	int status;
	if ((flag & AT_SYMLINK_NOFOLLOW) != 0) {
		// fake lchmod() with open() and fchmod()
		int fullfd = open(fullPath, O_RDONLY | O_SYMLINK);
		status = fchmod(fullfd, mode);
		close(fullfd);
	} else
		status = chmod(fullPath, mode);

	return status;
}


int
fchownat(int fd, const char* path, uid_t owner, gid_t group, int flag)
{
	if (flag != AT_SYMLINK_NOFOLLOW && flag != 0) {
		// invalid flag
		errno = EINVAL;
		return -1;
	}

	if (fd == AT_FDCWD || (path != NULL && path[0] == '/')) {
		// call chown() ignoring fd
		return (flag & AT_SYMLINK_NOFOLLOW) != 0 ? lchown(path, owner, group)
			: chown(path, owner, group);
	}

	if (fd < 0) {
		// Invalid file descriptor
		errno = EBADF;
		return -1;
	}

	char fullPath[MAXPATHLEN];
	if (get_path(fd, path, fullPath) < 0)
		return -1;

	return (flag & AT_SYMLINK_NOFOLLOW) != 0 ? lchown(fullPath, owner, group)
		: chown(fullPath, owner, group);
}


DIR*
fdopendir(int fd)
{
	struct stat st;
	if (fstat(fd, &st)) {
		// failed to get the stat info for fd, fstat() sets errno
		return NULL;
	}

	if (!S_ISDIR(st.st_mode)) {
		errno = ENOTDIR;
		return NULL;
	}

	char path[MAXPATHLEN];
	if (fcntl(fd, F_GETPATH, path) < 0) {
		// failed to get the path of fd, fcntl() sets errno
		return NULL;
	}

	DIR* dir = opendir(path);
	if (dir != NULL)
		close(fd);

	return dir;
}


int
fstatat(int fd, const char *path, struct stat *st, int flag)
{
	if (flag != AT_SYMLINK_NOFOLLOW && flag != 0) {
		// invalid flag
		errno = EINVAL;
		return -1;
	}

	if (fd == AT_FDCWD || (path != NULL && path[0] == '/')) {
		// call stat() or lstat() ignoring fd
		return (flag & AT_SYMLINK_NOFOLLOW) != 0 ? lstat(path, st)
			: stat(path, st);
	}

	if (fd < 0) {
		// Invalid file descriptor
		errno = EBADF;
		return -1;
	}

	char fullPath[MAXPATHLEN];
	if (get_path(fd, path, fullPath) < 0)
		return -1;

	return (flag & AT_SYMLINK_NOFOLLOW) != 0 ? lstat(fullPath, st)
		: stat(fullPath, st);
}


int
mkdirat(int fd, const char *path, mode_t mode)
{
	if (fd == AT_FDCWD || (path != NULL && path[0] == '/')) {
		// call mkdir() ignoring fd
		return mkdir(path, mode);
	}

	if (fd < 0) {
		// Invalid file descriptor
		errno = EBADF;
		return -1;
	}

	char fullPath[MAXPATHLEN];
	if (get_path(fd, path, fullPath) < 0)
		return -1;

	return mkdir(fullPath, mode);
}


int
mkfifoat(int fd, const char *path, mode_t mode)
{
	if (fd == AT_FDCWD || (path != NULL && path[0] == '/')) {
		// call mkfifo() ignoring fd
		return mkfifo(path, mode);
	}

	if (fd < 0) {
		// Invalid file descriptor
		errno = EBADF;
		return -1;
	}

	char fullPath[MAXPATHLEN];
	if (get_path(fd, path, fullPath) < 0)
		return -1;

	return mkfifo(fullPath, mode);
}


int
mknodat(int fd, const char *path, mode_t mode, dev_t dev)
{
	if (fd == AT_FDCWD || (path != NULL && path[0] == '/')) {
		// call mknod() ignoring fd
		return mknod(path, mode, dev);
	}

	if (fd < 0) {
		// Invalid file descriptor
		errno = EBADF;
		return -1;
	}

	char fullPath[MAXPATHLEN];
	if (get_path(fd, path, fullPath) < 0)
		return -1;

	return mknod(fullPath, mode, dev);
}


int
renameat(int oldFD, const char* oldPath, int newFD, const char* newPath)
{
	bool ignoreOldFD = false;
	bool ignoreNewFD = false;

	if (oldFD == AT_FDCWD || (oldPath != NULL && oldPath[0] == '/'))
		ignoreOldFD = true;

	if (newFD == AT_FDCWD || (newPath != NULL && newPath[0] == '/'))
		ignoreNewFD = true;

	if (ignoreOldFD && ignoreNewFD) {
		// call rename() ignoring the fd's
		return rename(oldPath, newPath);
	}

	char oldFullPath[MAXPATHLEN];
	if (!ignoreOldFD) {
		if (oldFD < 0) {
			// Invalid file descriptor
			errno = EBADF;
			return -1;
		}

		if (get_path(oldFD, oldPath, oldFullPath) < 0)
			return -1;
	}

	char newFullPath[MAXPATHLEN];
	if (!ignoreNewFD) {
		if (newFD < 0) {
			// Invalid file descriptor
			errno = EBADF;
			return -1;
		}

		if (get_path(newFD, newPath, newFullPath) < 0)
			return -1;
	}

	return rename(ignoreOldFD ? oldPath : oldFullPath,
		ignoreNewFD ? newPath : newFullPath);
}


ssize_t
readlinkat(int fd, const char *path, char *buffer, size_t bufferSize)
{
	if (fd == AT_FDCWD || (path != NULL && path[0] == '/')) {
		// call readlink() ignoring fd
		return readlink(path, buffer, bufferSize);
	}

	if (fd < 0) {
		// Invalid file descriptor
		errno = EBADF;
		return -1;
	}

	char fullPath[MAXPATHLEN];
	if (get_path(fd, path, fullPath) < 0)
		return -1;

	return readlink(fullPath, buffer, bufferSize);
}


int
symlinkat(const char *oldPath, int fd, const char *newPath)
{
	if (fd == AT_FDCWD || (newPath != NULL && newPath[0] == '/')) {
		// call symlink() ignoring fd
		return symlink(oldPath, newPath);
	}

	if (fd < 0) {
		// Invalid file descriptor
		errno = EBADF;
		return -1;
	}

	// newPath is relative to the fd
	char newFullPath[MAXPATHLEN];
	if (get_path(fd, newPath, newFullPath) < 0)
		return -1;

	return symlink(oldPath, newFullPath);
}


int
unlinkat(int fd, const char *path, int flag)
{
	if (flag != AT_REMOVEDIR && flag != 0) {
		// invalid flag
		errno = EINVAL;
		return -1;
	}

	if (fd == AT_FDCWD || (path != NULL && path[0] == '/')) {
		// call rmdir() or unlink() ignoring fd
		return (flag & AT_REMOVEDIR) != 0 ? rmdir(path) : unlink(path);
	}

	if (fd < 0) {
		// Invalid file descriptor
		errno = EBADF;
		return -1;
	}

	char fullPath[MAXPATHLEN];
	if (get_path(fd, path, fullPath) < 0)
		return -1;

	return (flag & AT_REMOVEDIR) != 0 ? rmdir(fullPath)
		: unlink(fullPath);
}


int
linkat(int oldFD, const char *oldPath, int newFD, const char *newPath,
	   int flag)
{
	if ((flag & AT_SYMLINK_FOLLOW) != 0) {
		// Dereference oldPath
		// CURRENTLY UNSUPPORTED
		errno = ENOTSUP;
		return -1;
	} else if (flag != 0) {
		errno = EINVAL;
		return -1;
	}

	bool ignoreOldFD = false;
	bool ignoreNewFD = false;

	if (oldFD == AT_FDCWD || (oldPath != NULL && oldPath[0] == '/'))
		ignoreOldFD = true;

	if (newFD == AT_FDCWD || (newPath != NULL && newPath[0] == '/'))
		ignoreNewFD = true;

	if (ignoreOldFD && ignoreNewFD) {
		// call link() ignoring the fd's
		return link(oldPath, newPath);
	}

	char oldFullPath[MAXPATHLEN];
	if (!ignoreOldFD) {
		if (oldFD < 0) {
			// Invalid file descriptor
			errno = EBADF;
			return -1;
		}

		if (get_path(oldFD, oldPath, oldFullPath) < 0)
			return -1;
	}

	char newFullPath[MAXPATHLEN];
	if (!ignoreNewFD) {
		if (newFD < 0) {
			// Invalid file descriptor
			errno = EBADF;
			return -1;
		}

		if (get_path(newFD, newPath, newFullPath) < 0)
			return -1;
	}

	return link(ignoreOldFD ? oldPath : oldFullPath,
		ignoreNewFD ? newPath : newFullPath);
}


int
futimesat(int fd, const char *path, const struct timeval times[2])
{
	if (fd == AT_FDCWD || (path != NULL && path[0] == '/')) {
		// call utimes() ignoring fd
		return utimes(path, times);
	}

	if (fd < 0) {
		// Invalid file descriptor
		errno = EBADF;
		return -1;
	}

	char fullPath[MAXPATHLEN];
	if (get_path(fd, path, fullPath) < 0)
		return -1;

	return utimes(fullPath, times);
}
