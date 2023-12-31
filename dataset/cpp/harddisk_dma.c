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
#include <limits.h>

#include <asm/io.h>

#include <drivers/ide.h>
#include <drivers/block_dev.h>
#include <drivers/block_dev/partition.h>
#include <mem/phymem.h>


extern int hd_ioctl(struct block_dev *bdev, int cmd, void *args, size_t size);
static const struct block_dev_ops idedisk_udma_driver;

static void setup_dma(struct hdc *hdc, char *buffer, int count, int cmd) {
	int i;
	int len;
	char *next;

	i = 0;
	next = (char *) ((unsigned long) buffer & ~(PAGESIZE - 1)) + PAGESIZE;
	while (1) {
		hdc->prds[i].addr = (unsigned long) buffer;
		len = next - buffer;
		if (count > len) {
			hdc->prds[i].len = len;
			count -= len;
			buffer = next;
			next += PAGESIZE;
			i++;
		} else {
			hdc->prds[i].len = count | 0x80000000;
			break;
		}
	}

	/* Setup PRD table */
	outl(hdc->prds_phys, hdc->bmregbase + BM_PRD_ADDR);

	/* Specify read/write */
	outb(cmd | BM_CR_STOP, hdc->bmregbase + BM_COMMAND_REG);

	/* Clear INTR & ERROR flags */
	outb(inb(hdc->bmregbase + BM_STATUS_REG) | BM_SR_INT | BM_SR_ERR,
		 hdc->bmregbase + BM_STATUS_REG);
}

static void start_dma(struct hdc *hdc) {
	/* Start DMA operation */
	outb(inb(hdc->bmregbase + BM_COMMAND_REG) | BM_CR_START,
		 hdc->bmregbase + BM_COMMAND_REG);
}

static int stop_dma(struct hdc *hdc) {
	int dmastat;

	/* Stop DMA channel and check DMA status */
	outb(inb(hdc->bmregbase + BM_COMMAND_REG) & ~BM_CR_START,
	   hdc->bmregbase + BM_COMMAND_REG);

	/* Get DMA status */
	dmastat = inb(hdc->bmregbase + BM_STATUS_REG);

	/* Clear INTR && ERROR flags */
	outb(dmastat | BM_SR_INT | BM_SR_ERR, hdc->bmregbase + BM_STATUS_REG);

	/* Check for DMA errors */
	if (dmastat & BM_SR_ERR) {
		return -EIO;
	}

	return 0;
}

static int hd_read_udma(struct block_dev *bdev, char *buffer, size_t count, blkno_t blkno) {
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


	while (sectsleft > 0) {
		/* Select drive */
		ide_select_drive(hd);

		/* Wait for controller ready */
		result = ide_wait(hdc, HDCS_DRDY, HDTIMEOUT_DRDY);
		if (result != 0) {
			result = -EIO;
			break;
		}

		/* Calculate maximum number of sectors we can transfer */
		if (sectsleft > 256) {
			nsects = 256;
		} else {
			nsects = sectsleft;
		}

		if (nsects > MAX_DMA_XFER_SIZE / bdev->block_size) {
			nsects = MAX_DMA_XFER_SIZE / bdev->block_size;
		}

		/* Prepare transfer */
		result = 0;
		hdc->dir = HD_XFER_DMA;
		hdc->active = hd;

		hd_setup_transfer(hd, blkno, nsects);

		/* Setup DMA */
		setup_dma(hdc, bufp, nsects * bdev->block_size, BM_CR_WRITE);

		/* Start read */
		outb(HDCMD_READDMA, hdc->iobase + HDC_COMMAND);
		start_dma(hdc);

		/* Stop DMA channel and check DMA status */
		result = stop_dma(hdc);
		if (result < 0) {
			break;
		}

		/* Check controller status */
		if (hdc->status & HDCS_ERR) {
			result = -EIO;
			break;
		}

		/* Advance to next */
		sectsleft -= nsects;
		bufp += nsects * bdev->block_size;
	}

	/* Cleanup */
	hdc->dir = HD_XFER_IDLE;
	hdc->active = NULL;

	return result == 0 ? count : result;
}

static int hd_write_udma(struct block_dev *bdev, char *buffer, size_t count, blkno_t blkno) {
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

	while (sectsleft > 0) {
		/* Select drive */
		ide_select_drive(hd);

		/* Wait for controller ready */
		result = ide_wait(hdc, HDCS_DRDY, HDTIMEOUT_DRDY);
		if (result != 0) {
			result = -EIO;
			break;
		}

		/* Calculate maximum number of sectors we can transfer */
		if (sectsleft > 256) {
			nsects = 256;
		} else {
			nsects = sectsleft;
		}
		if (nsects > MAX_DMA_XFER_SIZE / bdev->block_size) {
			nsects = MAX_DMA_XFER_SIZE / bdev->block_size;
		}

		/* Prepare transfer */
		result = 0;
		hdc->dir = HD_XFER_DMA;
		hdc->active = hd;

		hd_setup_transfer(hd, blkno, nsects);

		/* Setup DMA */
		setup_dma(hdc, bufp, nsects * bdev->block_size, BM_CR_READ);

		/* Start write */
		outb(HDCMD_WRITEDMA, hdc->iobase + HDC_COMMAND);
		start_dma(hdc);

		/* Stop DMA channel and check DMA status */
		result = stop_dma(hdc);
		if (result < 0) {
			break;
		}

		/* Check controller status */
		if (hdc->status & HDCS_ERR) {
			result = -EIO;
			break;
		}

		/* Advance to next */
		sectsleft -= nsects;
		bufp += nsects * bdev->block_size;
	}

	/* Cleanup */
	hdc->dir = HD_XFER_IDLE;
	hdc->active = NULL;

	return result == 0 ? count : result;
}

static int idedisk_udma_init (struct block_dev *bdev, void *args) {
	struct hd *drive;
	double size;
	char   path[PATH_MAX];

	drive = (struct hd *)args;
	/* Make new device */
	if (drive && (drive->media == IDE_DISK) && (drive->udmamode != -1)) {
		*path = 0;
		strcat(path, "/dev/hd*");
		if (0 > (drive->idx = block_dev_named(path, idedisk_idx))) {
			return drive->idx;
		}
		drive->bdev = block_dev_create(path,
				&idedisk_udma_driver, drive);
		if (NULL != drive->bdev) {
			size = (double) drive->param.cylinders *
				   (double) drive->param.heads *
				   (double) drive->param.unfbytes *
				   (double) (drive->param.sectors + 1);
			block_dev(drive->bdev)->size = (size_t) size;
		} else {
			return -1;
		}
		create_partitions(drive->bdev);
	}
	return 0;
}

static const struct block_dev_ops idedisk_udma_driver = {
	.bdo_ioctl = hd_ioctl,
	.bdo_read = hd_read_udma,
	.bdo_write = hd_write_udma,
	.bdo_probe = idedisk_udma_init,
};

BLOCK_DEV_DRIVER_DEF("idedisk_udma", &idedisk_udma_driver);
