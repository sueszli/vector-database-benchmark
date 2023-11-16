/**
 * @file
 *
 * @date 26.05.10
 * @author Alexander Batyukov
 * @author Anton Kozlov
 */

#include <embox/test.h>

#include <stdint.h>
#include <hal/env/traps_core.h>
#include <hal/test/traps_core.h>

EMBOX_TEST_SUITE("divzero test");

static volatile uint32_t a = 17;
static int was_in_trap = 0;

/* MMU data access exception handler */
static int dfault_handler(uint32_t trap_nr, void *data) {
	was_in_trap = 1;
	/* skip instruction */
	return 0;
}

TEST_CASE("division by zero must generate an exception") {
	volatile uint32_t zero = 0;

	traps_env_t old_env;

	traps_save_env(&old_env);
	traps_set_env(testtraps_env());

	set_fault_handler(DIVZERO_FAULT, dfault_handler);

	a = (23 / zero);

	traps_restore_env(&old_env);

	test_assert_true(was_in_trap);
}
