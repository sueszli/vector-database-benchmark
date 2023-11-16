/**
 * @file
 * @brief  DVFS interface implementation
 * @author Denis Deryugin
 * @date   11 Mar 2014
 */

#include <assert.h>
#include <errno.h>

#include <util/math.h>

#include <fs/file_desc.h>
#include <fs/dentry.h>
#include <drivers/block_dev.h>

/**
 * @brief Uninitialize file descriptor
 * @param desc File descriptor to be uninitialized
 *
 * @return Negative error code
 * @retval  0 Ok
 * @retval -1 Descriptor fields are inconsistent
 */
int dvfs_close(struct file_desc *desc) {
	if (!desc || !desc->f_inode || !desc->f_dentry)
		return -1;

	if (!(desc->f_dentry->flags & VFS_DIR_VIRTUAL)) {
		assert(desc->f_ops);
	}

	if (desc->f_ops && desc->f_ops->close) {
		desc->f_ops->close(desc);
	}

	if (!dentry_ref_dec(desc->f_dentry))
		dvfs_destroy_dentry(desc->f_dentry);

	dvfs_destroy_file(desc);
	return 0;
}

/**
 * @brief Application level interface to write the file
 * @param desc  File to be written
 * @param buf   Source of the data
 * @param count Length of the data
 *
 * @return Bytes written or negative error code
 * @retval       0 Ok
 * @retval -ENOSYS Function is not implemented in file system driver
 */
int dvfs_write(struct file_desc *desc, char *buf, int count) {
	int res = 0; /* Assign to avoid compiler warning when use -O2 */
	int retcode = count;
	struct inode *inode;

	if (!desc) {
		return -EINVAL;
	}

	inode = desc->f_inode;
	assert(inode);

	if (inode->length - desc->pos < count && !(inode->i_mode & DVFS_NO_LSEEK)) {
		if (inode->i_ops && inode->i_ops->truncate) {
			res = inode->i_ops->truncate(desc->f_inode, desc->pos + count);
			if (res) {
				retcode = -EFBIG;
			}
		} else {
			retcode = -EFBIG;
		}
	}

	if (desc->f_ops && desc->f_ops->write) {
		res = desc->f_ops->write(desc, buf, count);
	} else {
		retcode = -ENOSYS;
	}

	if (res > 0) {
		desc->pos += res;
	}

	return retcode;
}

/**
 * @brief Application level interface to read the file
 * @param desc  File to be read
 * @param buf   Destination
 * @param count Length of the data
 *
 * @return Bytes read or negative error code
 * @retval       0 Ok
 * @retval -ENOSYS Function is not implemented in file system driver
 */
int dvfs_read(struct file_desc *desc, char *buf, int count) {
	int res;
	int sz;
	if (!desc)
		return -1;

	sz = min(count, desc->f_inode->length - desc->pos);

	if (sz <= 0)
		return 0;

	if (desc->f_ops && desc->f_ops->read)
		res = desc->f_ops->read(desc, buf, count);
	else
		return -ENOSYS;

	if (res > 0)
		desc->pos += res;

	return res;
}

int dvfs_fstat(struct file_desc *desc, struct stat *sb) {
	*sb = (struct stat) {
		.st_size = desc->f_inode->length,
		.st_mode = desc->f_inode->i_mode,
		.st_uid = 0,
		.st_gid = 0
	};

	sb->st_blocks = sb->st_size;

	if (desc->f_inode->i_sb->bdev) {
        	sb->st_blocks /= block_dev_block_size(desc->f_inode->i_sb->bdev);
	       	sb->st_blocks += (sb->st_blocks % block_dev_block_size(desc->f_inode->i_sb->bdev) != 0);
        }

	return 0;
}
