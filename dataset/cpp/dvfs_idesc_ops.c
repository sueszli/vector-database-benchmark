/**
 * @file
 * @brief File descriptor (fd) abstraction over FILE *
 * @details Provide POSIX kernel support to operate with flides (int)
 * instead of FILE *. Rewritten from old VFS.
 * @date 20.05.15
 * @author Anton Kozlov
 * @author Denis Deryugin
 */

#include <assert.h>
#include <errno.h>
#include <sys/uio.h>

#include <fs/dvfs.h>

#include <kernel/task/resource/idesc.h>

extern const struct idesc_ops idesc_file_ops;

static void idesc_file_ops_close(struct idesc *idesc) {
	assert(idesc);

	dvfs_close((struct file_desc *)idesc);
}

static ssize_t idesc_file_ops_read(struct idesc *idesc, const struct iovec *iov, int cnt) {
	void *buf;
	size_t nbyte;

	assert(idesc);
	assert(idesc->idesc_ops == &idesc_file_ops);

	assert(iov);
	assert(cnt == 1);

	buf = iov->iov_base;
	nbyte = iov->iov_len;

	return dvfs_read((struct file_desc *) idesc, buf, nbyte);
}

static ssize_t idesc_file_ops_write(struct idesc *idesc, const struct iovec *iov, int cnt) {
	void *buf;
	size_t nbyte;

	assert(idesc);
	assert(idesc->idesc_ops == &idesc_file_ops);

	assert(iov);
	assert(cnt == 1);

	buf = iov->iov_base;
	nbyte = iov->iov_len;

	return dvfs_write((struct file_desc *)idesc, (char *)buf, nbyte);
}

static int idesc_file_ops_stat(struct idesc *idesc, void *buf) {
	assert(idesc);
	assert(idesc->idesc_ops == &idesc_file_ops);

	return dvfs_fstat((struct file_desc *)idesc, buf);;
}

static int idesc_file_ops_ioctl(struct idesc *idesc, int request, void *data) {
	assert(idesc);
	assert(idesc->idesc_ops == &idesc_file_ops);

	return ((struct file_desc *)idesc)->f_ops->ioctl((struct file_desc *)idesc, request, data);
}

static int idesc_file_ops_status(struct idesc *idesc, int mask) {
	assert(idesc);
	assert(idesc->idesc_ops == &idesc_file_ops);

	return 1;
}

const struct idesc_ops idesc_file_ops = {
	.close = idesc_file_ops_close,
	.id_readv  = idesc_file_ops_read,
	.id_writev = idesc_file_ops_write,
	.ioctl = idesc_file_ops_ioctl,
	.fstat = idesc_file_ops_stat,
	.status = idesc_file_ops_status,
};
