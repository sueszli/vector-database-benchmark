//
// Created by tcz001 on 2020/8/7.
//

#include "raspi2/led.h"
#include "raspi2/gpio.h"

/*
 * Configure number of loops reading HW register to wait one second
 */
#define TOGGLE_LOOP_CNT 5000000 /* 5MHz is about 1 second on B+ */

/* and -O2 compiler optimizations. */

int led_init(void) {
    unsigned int i, j, select;

#if defined(RASPI4)
    /* GPIO 42 is 2nd register in GPFSEL4, so 2 * 3 bits or bit 6. */
    /* Clear the 3 bit range (7) starting at bit 6 */
    select = REG32(GPFSEL4);
    select &= ~(7 << 6);

    /* Configure the LED (GPIO 42) starting at bit 6, as output (1). */
    select |= (GPIO_OUTPUT << 6);
    REG32(GPFSEL4) = select;
#elif defined(RASPI3) || defined(RASPI2)
    /*
  ** Enable LED. 3 bits per GPIO, so 10 GPIOs per select register means
  ** GPIO 29 is select register two number 9. 3 bits per GPIO so 9
  ** starts at bit 27.
  */

    /* Clear the 3 bit range (7) starting at bit 27 */
    select = REG32(GPFSEL2);
    select &= ~(7 << 27);

    /* 3 bits per GPIO, input (0), output (1) and alternate select */
    /* 0 through 5. */

    /* Configure the LED (GPIO 29) starting at bit 27, as output (1). */
    select |= (GPIO_OUTPUT << 27);
    REG32(GPFSEL2) = select;
#else
    /* Enable LED. 3 bits per GPIO so 10 GPIOs per select register. */
    /* GPIO 47 is 7th register in GPFSEL4, so 7 * 3 bits or bit 21. */

    /* Clear the 3 bit range (7) starting at bit 21 */
    select = REG32(GPFSEL4);
    select &= ~(7 << 21);

    /* 3 bits per GPIO, input (0), output (1) and alternate select */
    /* 0 through 5. */

    /* Configure the LED (GPIO 47) starting at bit 21, as output (1). */
    select |= (GPIO_OUTPUT << 21);
    REG32(GPFSEL4) = select;
#endif

    /* GPPUD - GPio Pin Up Down configuration */
    /*   (0) disable pull up and pull down to float the GPIO */
    /*   (1 << 0) enable pull down (low) */
    /*   (1 << 1) enable pull up (high) */

    /* Always pull up (high) for LEDs as they require voltage. */
    REG32(GPPUD) = GPPUD_PULL_UP;

    /* Loop to wait until GPPUD assignment persists. */
    for (i = 0; i < TOGGLE_LOOP_CNT / 1000; ++i) {
        select = REG32(GPFSEL4);
    }

#if defined(RASPI4)
    /* Push GPPUD settings to GPPUDCLK1 GPIO 42. */
    REG32(GPPUDCLK1) = (1 << (42 - 32)); /* GPIO 42 */
#elif defined(RASPI3) || defined(RASPI2)
    /* Push GPPUD settings to GPPUDCLK0 GPIO 29. */
    REG32(GPPUDCLK0) = (1 << 29); /* GPIO 29 */
#else
    /* Push GPPUD settings to GPPUDCLK1 GPIO 47. */
    REG32(GPPUDCLK1) = (1 << (47 - 32)); /* GPIO 47 */
#endif

    /* Loop to wait until GPPUD clock assignment persists. */
    for (i = 0; i < TOGGLE_LOOP_CNT / 1000; ++i) {
        select = REG32(GPFSEL4);
    }

    /*
   * Enable the JTAG GPIOs
   */

    /* Disable pull up/down for the next configured GPIO. */
    REG32(GPPUD) = GPPUD_OFF;

    /* Loop to wait until GPPUD assignment persists. */
    for (i = 0; i < TOGGLE_LOOP_CNT / 1000; ++i) {
        select = REG32(GPFSEL4);
    }

    // Apply to all the JTAG GPIO pins
    REG32(GPPUDCLK0) = (1 << 22) | (1 << 23) | (1 << 24) | (1 << 25) | (1 << 26) | (1 << 27);

    /* Loop to wait until GPPUD clock assignment persists. */
    for (i = 0; i < TOGGLE_LOOP_CNT / 1000; ++i) {
        select = REG32(GPFSEL4);
    }

    // Select level alternate 4 to enable JTAG
    select = REG32(GPFSEL2);
    select &= ~(7 << 6);      // gpio22
    select |= GPIO_ALT4 << 6; // alt4 ARM_TRST
    select &= ~(7 << 9);      // gpio23
    select |= GPIO_ALT4 << 9; // alt4 ARM_RTCK
    select &= ~(7 << 12);     // gpio24
    select |= GPIO_ALT4 << 12;// alt4 ARM_TDO
    select &= ~(7 << 15);     // gpio25
    select |= GPIO_ALT4 << 15;// alt4 ARM_TCK
    select &= ~(7 << 18);     // gpio26
    select |= GPIO_ALT4 << 18;// alt4 ARM_TDI
    select &= ~(7 << 21);     // gpio27
    select |= GPIO_ALT4 << 21;// alt4 ARM_TMS
    REG32(GPFSEL2) = select;

    /* Loop turning the activity LED on and off. */
    for (j=0;j<5;j++) {
        /* Turn on the activity LED. */
#if defined(RASPI4)
        /* RPI 4 has LED at GPIO 42, so set GPIO 42. */
        REG32(GPSET1) = 1 << (42 - 32);
#elif defined(RASPI3) || defined(RASPI2)
        /* RPI 3 has LED at GPIO 29, so set GPIO 29. */
        REG32(GPSET0) = 1 << 29;
#else
        /* Other RPIs have LED at GPIO 47, so set GPIO 47. */
        REG32(GPSET1) = 1 << (47 - 32);
#endif

        // Loop to wait a bit
        for (i = 0; i < TOGGLE_LOOP_CNT/10; ++i) { /* loop to pause LED on */
            select = REG32(GPFSEL4);
        }

        /* Turn off the activity LED. */
#if defined(RASPI4)
        /* RPI 4 has LED at GPIO 42, so clear GPIO 42. */
        REG32(GPCLR1) = 1 << (42 - 32);
#elif defined(RASPI3) || defined(RASPI2)
        /* RPI 3 has LED at GPIO 29, so clear GPIO 29. */
        REG32(GPCLR0) = 1 << 29;
#else
        /* Other RPIs have LED at GPIO 47, so clear GPIO 47. */
        REG32(GPCLR1) = 1 << (47 - 32);
#endif

        // Loop to wait a bit
        for (i = 0; i < TOGGLE_LOOP_CNT/10; ++i) { /* loop to pause LED off */
            select = REG32(GPFSEL4);
        }
    }
    return select;
}
