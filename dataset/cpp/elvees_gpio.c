/**
 * @file
 *
 * @date Jan 24, 2023
 * @author Anton Bondarev
 */

#include <util/log.h>

#include <stdint.h>
#include <stddef.h>
#include <sys/mman.h>

#include <hal/reg.h>

#include <drivers/common/memory.h>
#include <drivers/gpio/gpio.h>
#include <drivers/gpio/gpio_driver.h>

#include <framework/mod/options.h>


#define GPIO_BASE_ADDR(port_num) \
	((uintptr_t) OPTION_GET(NUMBER,base_addr) + (port_num) * 0x1000)

#define GPIO_PORTS_COUNT           OPTION_GET(NUMBER,gpio_ports)

#define OUT_EN_SET     (0x10)
#define OUT_EN_CLR     (0x14)

#define GPIO_OUT_EN_SET(port)    (GPIO_BASE_ADDR(port) + OUT_EN_SET)
#define GPIO_OUT_EN_CLR(port)    (GPIO_BASE_ADDR(port) + OUT_EN_CLR)

extern void elvees_gpio_setup_func(uint32_t port, uint32_t pin, int func);
extern void elvees_gpio_setup_digit(uint32_t port, uint32_t pin);

static int elvees_gpio_setup_mode(unsigned char port, gpio_mask_t mask, int m) {
	int i;

	for (i = 0; i < 16; i++) {
		if (mask & (1 << i)) {
			if (m & GPIO_MODE_OUT_ALTERNATE) {
				elvees_gpio_setup_func(port, i, GPIO_GET_ALTERNATE(m));
			}
			if (m & GPIO_MODE_OUT) {
				elvees_gpio_setup_digit(port, i);
				REG32_STORE(GPIO_OUT_EN_SET(port), (1 << i));
			}
			if (m & GPIO_MODE_IN) {
				elvees_gpio_setup_digit(port, i);
				REG32_STORE(GPIO_OUT_EN_CLR(port), (1 << i));
			}
		}
	}

	return 0;
}

static void elvees_gpio_set(unsigned char port, gpio_mask_t mask, char level) {
}

static gpio_mask_t elvees_gpio_get(unsigned char port, gpio_mask_t mask) {
	return 0;
}

struct gpio_chip elvees_gpio_chip = {
	.setup_mode = elvees_gpio_setup_mode,
	.get = elvees_gpio_get,
	.set = elvees_gpio_set,
	.nports = GPIO_PORTS_COUNT
};

GPIO_CHIP_DEF(&elvees_gpio_chip);

PERIPH_MEMORY_DEFINE(elvees_gpio, GPIO_BASE_ADDR(0), 0x1000 * GPIO_PORTS_COUNT);
