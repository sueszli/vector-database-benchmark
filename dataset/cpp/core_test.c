/* radare - LGPL - Copyright 2014 - pancake */
#if 0
gcc -o core_test.so -fPIC `pkg-config --cflags --libs r_core` core_test.c -shared
mkdir -p ~/.config/radare2/plugins
mv core_test.so ~/.config/radare2/plugins
#endif

#include <r_types.h>
#include <r_lib.h>
#include <r_cmd.h>
#include <r_core.h>
#include <r_cons.h>
#include <string.h>
#include <r_anal.h>

#undef R_API
#define R_API static
#undef R_IPI
#define R_IPI static

static int r_cmd_test_call(void) {
	eprintf ("Dummy!\n");
	return false;
}

RCorePlugin r_core_plugin_test = {
	.name = "test",
	.desc = "lalallala",
	.license = "MIT",
	.call = r_cmd_test_call,
};

#ifndef R2_PLUGIN_INCORE
R_API RLibStruct radare_plugin = {
	.type = R_LIB_TYPE_CORE,
	.data = &r_core_plugin_test,
	.version = R2_VERSION
};
#endif
