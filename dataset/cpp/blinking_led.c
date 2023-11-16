/**
 * @file
 * @brief blinking led
 *
 * @date 01.10.10
 * @author Anton Kozlov
 */

#include <embox/test.h>
#include <drivers/at91_olimex_debug_led.h>
#include <drivers/pins.h>
#include <unistd.h>

EMBOX_TEST_SUITE("olimex blinking_led");

#define DELAY   0x25000
#define INC     0x0

static void delay(int d) {
	int i = 0;
	while (i < d) {
		i += 1;
	}
}

static int count = 3;

static void led1_on(void) {
	pin_clear_output(OLIMEX_SAM7_LED1);
}
static void led1_off(void) {
	pin_set_output(OLIMEX_SAM7_LED1);
}

TEST_CASE("olimex blinking_led test") {
	volatile int del = DELAY;
	pin_config_output(OLIMEX_SAM7_LED1 | OLIMEX_SAM7_LED2);

	while (count--) {
		led1_on();
		delay(del);
		led1_off();
		delay(del);
		del += INC;
	}
}
