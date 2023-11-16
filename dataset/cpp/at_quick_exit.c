#include <stdlib.h>

__attribute__((weak)) int at_quick_exit(void (*func)(void))
{
	(void)func;

	// EA libc does not exit on bare metal systems
	return 0;
}
