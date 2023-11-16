/**
 * @file
 * @brief OMAP35x clock device driver
 *
 * @date 15.10.12
 * @author Ilia Vaprol
 */

#include <drivers/common/memory.h>
#include <drivers/irqctrl.h>
#include <hal/clock.h>
#include <hal/mmu.h>
#include <hal/reg.h>
#include <hal/system.h>
#include <kernel/irq.h>
#include <kernel/time/clock_source.h>
#include <kernel/printk.h>
#include <util/binalign.h>

#include <embox/unit.h>

/* omap's gptimer */

#define CM_FCLKEN_WKUP  0x48004C00
#define CM_ICLKEN_WKUP  0x48004C10
#define CM_CLKSEL_WKUP  0x48004C40

#define GPTIMER1_BASE ((void *) 0x48318000)
#define GPTIMER1_IRQ 37

struct gptimerxx_x {
	uint32_t tidr; 		/* 0x00 */
	char unused0[12];	/* 0x04 */
	uint32_t cfg;		/* 0x10 */
	uint32_t tistat;	/* 0x14 */
	uint32_t tisr;		/* 0x18 */
	uint32_t tier;		/* 0x1c */
	uint32_t twer;		/* 0x20 */
	uint32_t tclr;		/* 0x24 */
	uint32_t tcrr;		/* 0x28 */
	uint32_t tldr;		/* 0x2c */
	uint32_t ttgr;		/* 0x30 */
	uint32_t twps;		/* 0x34 */
	uint32_t tmar;		/* 0x38 */
	uint32_t tcar1;		/* 0x3c */
	uint32_t tsicr;		/* 0x40 */
	uint32_t tcar2;		/* 0x44 */
	uint32_t tpir;		/* 0x48 */
	uint32_t tnir;		/* 0x4c */
	uint32_t tcvr;		/* 0x50 */
	uint32_t tocr;		/* 0x54 */
	uint32_t towr;		/* 0x54 */
} __attribute__((packed));

#define OMAP_LOAD_VALUE 0xffffffe0

/* common register values */

#define GPTIMER_TCLR_START	(1 << 0)
#define GPTIMER_TCLR_AUTORELOAD (1 << 1)

#define GPTIMER_TIER_OVERFLOW   (1 << 1)
#define GPTIMER_TISR_OVERFLOW   (1 << 1)

static irq_return_t clock_handler(unsigned int irq_nr, void *data) {
	volatile struct gptimerxx_x *gptimer = GPTIMER1_BASE;
	clock_tick_handler(data);
	REG_STORE(&gptimer->tisr, GPTIMER_TISR_OVERFLOW);
	return IRQ_HANDLED;
}

static int omap3_clk_set_periodic(struct clock_source *cs) {
	volatile struct gptimerxx_x *gptimer = GPTIMER1_BASE;

	REG_ORIN(CM_FCLKEN_WKUP, 1);
	REG_ORIN(CM_ICLKEN_WKUP, 1);
	REG_ANDIN(CM_CLKSEL_WKUP, ~1);

	REG_STORE(&gptimer->cfg, 0x2);

	REG_STORE(&gptimer->tpir, 232000);
	REG_STORE(&gptimer->tnir, -768000);

	REG_STORE(&gptimer->tcrr, OMAP_LOAD_VALUE);
	REG_STORE(&gptimer->tldr, OMAP_LOAD_VALUE);

	REG_STORE(&gptimer->tclr, GPTIMER_TCLR_START | GPTIMER_TCLR_AUTORELOAD);

	REG_STORE(&gptimer->tier, GPTIMER_TIER_OVERFLOW);
	REG_STORE(&gptimer->twer, GPTIMER_TIER_OVERFLOW);

	return 0;
}

static struct time_event_device omap3_clk_event = {
	.set_periodic = omap3_clk_set_periodic,
	.irq_nr = GPTIMER1_IRQ,
};

static int omap_clk_init(struct clock_source *cs) {
	/* Map one vmem page to handle this device if mmu is used */
	return irq_attach(GPTIMER1_IRQ, clock_handler, 0, cs, "omap3_clk");
}

PERIPH_MEMORY_DEFINE(omap3_gptimer, (uintptr_t) GPTIMER1_BASE, 0x1000);

PERIPH_MEMORY_DEFINE(omap3_cm, CM_FCLKEN_WKUP, 0x50);

CLOCK_SOURCE_DEF(omap3_clk, omap_clk_init, NULL,
	&omap3_clk_event, NULL);
