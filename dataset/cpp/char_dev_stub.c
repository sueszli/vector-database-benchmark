/**
 * @file
 * @brief Stub for init and register char devices
 * @author Denis Deryugin <deryugin.denis@gmail.com>
 * @version 0.1
 * @date 2015-09-10
 *
 * This stub is required to have FS and have no char devs
 * at the same time.
 */
#include <errno.h>
#include <stddef.h>
#include <drivers/char_dev.h>

/**
 * @brief Stub
 *
 * @return Always 0
 */
int char_dev_init_all(void) {
	return 0;
}

/**
 * @brief Stub
 *
 * @param name Device name
 * @param ops Device file operations
 *
 * @return Always -1
 */
int char_dev_register(const struct dev_module *cdev) {
	return -ENOSUPP;
}

struct inode;

struct idesc *char_dev_open(struct inode *node, int flags) {
	return NULL;
}

int char_dev_idesc_fstat(struct idesc *idesc, void *buff) {
	return 0;
}

struct idesc *char_dev_default_open(struct dev_module *cdev, void *priv) {
	return NULL;
}

void char_dev_default_close(struct idesc *idesc) {
	/* Do nothing */
}
