/**
 * @file dw_spi.c
 * @brief Designware SPI driver
 * @author Denis Deryugin <deryugin.denis@gmail.com>
 * @version 0.1
 * @date 19.12.2019
 */
#include <errno.h>

#include <drivers/common/memory.h>
#include <drivers/spi.h>
#include <embox/unit.h>
#include <framework/mod/options.h>
#include <hal/reg.h>
#include <util/log.h>
#include <util/math.h>
#include <kernel/irq.h>

#include <drivers/gpio/gpio.h>

#include "dw_spi.h"

#define SPI_DE0_NANO_SOC      OPTION_GET(BOOLEAN,spi_de0_nano_soc)

/* I am not sure it's a constant for all types of DW SPI controllers.
 * For example, Linux calculates this value in drivers/spi/spi-dw.c:spi_hw_init,
 * But at the same time, DW SPI datasheet says something
 * about 256 entries (TXFLR). */
#define DW_SPI_TX_FIFO_LEN   256

static uint32_t dw_spi_read(struct dw_spi *dw_spi, int offset) {
	return REG32_LOAD(dw_spi->base_addr + offset);
}

static void dw_spi_write(struct dw_spi *dw_spi, int offset, uint32_t val) {
	REG32_STORE(dw_spi->base_addr + offset, val);
}


/* Init hook for DE0 Nano SOC board. SPI and I2C share
 * same pins, so we need to switch to SPI. */
static void __attribute__((used)) spi1_de0_nano_soc_init(void) {
	gpio_setup_mode(GPIO_PORT_B, 1 << 11, GPIO_MODE_OUT);
	gpio_set(GPIO_PORT_B, 1 << 11, GPIO_PIN_LOW);
}

static int dw_spi_init(struct dw_spi *dw_spi, uintptr_t base_addr, int spi_nr) {
	uint32_t reg;

	dw_spi->base_addr = base_addr;

#if SPI_DE0_NANO_SOC
	if (spi_nr == 1) {
		spi1_de0_nano_soc_init();
	}
#endif

	/* FIXME
	 * It also required to set clocks and clear resets.
	 * Currently, all that stuff is managed by uboot.
	 *
	 * "Cyclone V Hard Processor System Technical Reference Manual":
	 *    1. Clocks: "Clock Manager" chapter.
	 *    2. Reset:  "Reset Manager" chapter.
	 **/

	dw_spi_write(dw_spi, DW_SPI_ENR, 0);

	reg = dw_spi_read(dw_spi, DW_SPI_SPI_VERSION_ID);
	if (reg != DW_SPI_VERSION) {
		log_error("SPI Version mismatch!");
		return -1;
	}

	reg = dw_spi_read(dw_spi, DW_SPI_IDR);
	if (reg != DW_SPI_IDR_VAL) {
		log_error("SPI ID mismatch!");
		return -1;
	}
	/* Disable all interrupts */
	dw_spi_write(dw_spi, DW_SPI_IMR, 0);

	dw_spi_write(dw_spi, DW_SPI_BAUDR, 16);

	reg = dw_spi_read(dw_spi, DW_SPI_CTRLR0);
	reg |= DW_SPI_CTRLR0_SCPOL;
	reg |= DW_SPI_CTRLR0_SCPH;
	dw_spi_write(dw_spi, DW_SPI_CTRLR0, reg);

	dw_spi_write(dw_spi, DW_SPI_ENR, 1);

	return 0;
}

static int dw_spi_tx(struct dw_spi *dw_spi, uint8_t *buf, int count) {
	int i;
	int tx_slots = DW_SPI_TX_FIFO_LEN - dw_spi_read(dw_spi, DW_SPI_TXFLR);

	count = min(count, tx_slots);
	if (!count) {
		return 0;
	}

	i = count;
	while (i--) {
		dw_spi_write(dw_spi, DW_SPI_DR, *buf++);
	}

	return count;
}

static int dw_spi_rx(struct dw_spi *dw_spi, uint8_t *buf, int count) {
	int i;

	count = min(count, dw_spi_read(dw_spi, DW_SPI_RXFLR));
	if (!count) {
		return 0;
	}

	i = count;
	while (i--) {
		*buf++ = dw_spi_read(dw_spi, DW_SPI_DR);
	}

	return count;
}

static int dw_spi_select(struct spi_device *dev, int cs) {
	struct dw_spi *dw_spi = dev->priv;

	if (cs < 0 || cs > 3) {
		log_error("Only cs=0..3 are avalable!");
		return -EINVAL;
	}

	dw_spi_write(dw_spi, DW_SPI_SER, 1 << cs);

	return 0;
}

static int dw_spi_transfer(struct spi_device *dev, uint8_t *inbuf,
		uint8_t *outbuf, int count) {
	struct dw_spi *dw_spi = dev->priv;
	int tx_cnt, rx_cnt;

	if (!inbuf || !outbuf) {
		return -1;
	}

	tx_cnt = rx_cnt = 0;
	irq_lock();
	while (tx_cnt < count || rx_cnt < count) {
		/* Do not add log_debug() or some another stuff here,
		 * because we need to write all tx data before transfer competed. */
		tx_cnt += dw_spi_tx(dw_spi, inbuf + tx_cnt,  count - tx_cnt);
		rx_cnt += dw_spi_rx(dw_spi, outbuf + rx_cnt, count - rx_cnt);
	}
	irq_unlock();

	return 0;
}

struct spi_ops dw_spi_ops = {
	.select   = dw_spi_select,
	.transfer = dw_spi_transfer
};

#define DW_SPI0_BASE OPTION_GET(NUMBER, base_addr0)
#define DW_SPI1_BASE OPTION_GET(NUMBER, base_addr1)
#define DW_SPI2_BASE OPTION_GET(NUMBER, base_addr2)
#define DW_SPI3_BASE OPTION_GET(NUMBER, base_addr3)

#define SPI_DEV_NAME     dw_spi

#if DW_SPI0_BASE != 0
static struct dw_spi dw_spi0;
PERIPH_MEMORY_DEFINE(dw_spi0, DW_SPI0_BASE, 0x100);
SPI_DEV_DEF(SPI_DEV_NAME, &dw_spi_ops, &dw_spi0, 0);
#endif

#if DW_SPI1_BASE != 0
static struct dw_spi dw_spi1;
PERIPH_MEMORY_DEFINE(dw_spi1, DW_SPI1_BASE, 0x100);
SPI_DEV_DEF(SPI_DEV_NAME, &dw_spi_ops, &dw_spi1, 1);
#endif

#if DW_SPI2_BASE != 0
static struct dw_spi dw_spi2;
PERIPH_MEMORY_DEFINE(dw_spi2, DW_SPI2_BASE, 0x100);
SPI_DEV_DEF(SPI_DEV_NAME, &dw_spi_ops, &dw_spi2, 2);
#endif

#if DW_SPI3_BASE != 0
static struct dw_spi dw_spi3;
PERIPH_MEMORY_DEFINE(dw_spi3, DW_SPI3_BASE, 0x100);
SPI_DEV_DEF(SPI_DEV_NAME, &dw_spi_ops, &dw_spi3, 3);
#endif

EMBOX_UNIT_INIT(dw_spi_module_init);
static int dw_spi_module_init(void) {
#if DW_SPI0_BASE != 0
	dw_spi_init(&dw_spi0, DW_SPI0_BASE, 0);
#endif
#if DW_SPI1_BASE != 0
	dw_spi_init(&dw_spi1, DW_SPI1_BASE, 1);
#endif
#if DW_SPI2_BASE != 0
	dw_spi_init(&dw_spi2, DW_SPI2_BASE, 2);
#endif
#if DW_SPI3_BASE != 0
	dw_spi_init(&dw_spi3, DW_SPI3_BASE, 3);
#endif
	return 0;
}
