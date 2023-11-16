/**
 * @file
 * @brief
 *
 * @date 26.10.2012
 * @author Andrey Gazukin
 */

#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>

#include <asm/io.h>

#include <kernel/irq_lock.h>
#include <drivers/ide.h>
#include <drivers/block_dev.h>
#include <drivers/block_dev/partition.h>
#include <mem/phymem.h>

#include <kernel/time/ktime.h>
#include <kernel/thread/waitq.h>

#define HD_WAIT_MS 10

extern int hd_ioctl(struct block_dev *bdev, int cmd, void *args, size_t size);
static const struct block_dev_ops idedisk_pio_driver;
static int hd_read_pio(struct block_dev *bdev, char *buffer, size_t count, blkno_t blkno) {
	struct hd *hd;
	struct hdc *hdc;
	int sectsleft;
	int nsects;
	int result = 0;
	char *bufp;

	if (count == 0) {
		return 0;
	}
	bufp = (char *) buffer;
	hd = block_dev_priv(bdev);
	hdc = hd->hdc;
	sectsleft = count / bdev->block_size;
	if (count % bdev->block_size) {
		sectsleft++;
	}

	while (sectsleft > 0) {
		/* Select drive */
		ide_select_drive(hd);

		/* Wait for controller ready */
		result = ide_wait(hdc, HDCS_DRDY, HDTIMEOUT_DRDY);
		if (result != 0) {
			hdc->result = -EIO;
			break;
		}

		nsects = 1;

		/* Prepare transfer */
		hdc->bufp = bufp;
		hdc->nsects = nsects;
		hdc->result = 0;
		hdc->dir = HD_XFER_READ;
		hdc->active = hd;

		hd_setup_transfer(hd, blkno, nsects);
		outb(hd->multsect > 1 ? HDCMD_MULTREAD : HDCMD_READ,
				hdc->iobase + HDC_COMMAND);

		/* Wait until data read */
		WAITQ_WAIT(&hdc->waitq, hdc->result);

		if (hdc->result < 0) {
			break;
		}

		/* Advance to next */
		sectsleft -= nsects;
		bufp += nsects * bdev->block_size;
		blkno += nsects; /*WTF?*/
	}

	/* Cleanup */
	hdc->dir = HD_XFER_IDLE;
	hdc->active = NULL;
	if (0 < hdc->result) {
		result = hdc->result = 0;
	}

	return result == 0 ? count : result;
}

static int hd_write_pio(struct block_dev *bdev, char *buffer, size_t count, blkno_t blkno) {
	struct hd *hd;
	struct hdc *hdc;
	int sectsleft;
	int nsects;
	int n;
	int result = 0;
	char *bufp;

	if (count == 0) {
		return 0;
	}
	bufp = (char *) buffer;
	hd = block_dev_priv(bdev);
	hdc = hd->hdc;
	sectsleft = count / bdev->block_size;


	while (sectsleft > 0) {
		/* Select drive */
		ide_select_drive(hd);

		/* Wait for controller ready */
		result = ide_wait(hdc, HDCS_DRDY, HDTIMEOUT_DRDY);
		if (result != 0) {
			hdc->result = -EIO;
			break;
		}

		nsects = 1;

		/* Prepare transfer */
		hdc->bufp = bufp;
		hdc->nsects = nsects;
		hdc->result = 0;
		hdc->dir = HD_XFER_WRITE;
		hdc->active = hd;

		hd_setup_transfer(hd, blkno, nsects);
		outb(hd->multsect > 1 ? HDCMD_MULTWRITE : HDCMD_WRITE,
				hdc->iobase + HDC_COMMAND);

		/* Wait for data ready */
		if (!(inb(hdc->iobase + HDC_ALT_STATUS) & HDCS_DRQ)) {
			result = ide_wait(hdc, HDCS_DRQ, HDTIMEOUT_DRQ);
			if (result != 0) {
				hdc->result = -EIO;
				break;
			}
		}

		/* Write first sector(s) */
		n = hd->multsect;
		if (n > nsects) {
			n = nsects;
		}
		while (n-- > 0) {
			pio_write_buffer(hd, hdc->bufp, bdev->block_size);
			hdc->bufp += bdev->block_size;
		}

		/* Wait until data written */
		WAITQ_WAIT(&hdc->waitq, hdc->result);

		if (hdc->result < 0) {
			break;
		}

		/* Advance to next */
		sectsleft -= nsects;
		bufp += nsects * bdev->block_size;
		blkno += nsects; /*WTF?*/
	}

	/* Cleanup */
	hdc->dir = HD_XFER_IDLE;
	hdc->active = NULL;

	/*
	 * FIXME This assignment crashes writing, because
	 * hdc->result equal 1 after writing. So, it set result to 1, afterward
	 * (return result == 0 ? count : result) return 1.
	 * --Alexander
	 *
	result = hdc->result;
	*/

	return result == 0 ? count : result;
}

static int idedisk_init (struct block_dev *bdev, void *args) {
	struct hd *drive;
	size_t size;
	char path[PATH_MAX];

	drive = (struct hd *)args;
	/* Make new device */
	if (drive && (drive->media == IDE_DISK) && (drive->udmamode == -1)) {
		*path = 0;
		strcat(path, "/dev/hd*");
		if (0 > (drive->idx = block_dev_named(path, idedisk_idx))) {
			return drive->idx;
		}
		drive->bdev = block_dev_create(path,
				&idedisk_pio_driver, drive);
		if (NULL != drive->bdev) {
			bdev = (struct block_dev *) drive->bdev;
			bdev->size = drive->blks * bdev->block_size;
		} else {
			return -1;
		}
		bdev->block_size = 512;
		size = drive->blks * bdev->block_size;
		bdev->size = size;
		drive->bdev = bdev;
		create_partitions(bdev);
	}
	return 0;
}

static const struct block_dev_ops idedisk_pio_driver = {
	.bdo_ioctl = hd_ioctl,
	.bdo_read = hd_read_pio,
	.bdo_write = hd_write_pio,
	.bdo_probe = idedisk_init,
};

BLOCK_DEV_DRIVER_DEF("idedisk", &idedisk_pio_driver);
