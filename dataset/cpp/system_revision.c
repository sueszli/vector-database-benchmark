/*
 * Copyright 2011, Oliver Tappe <zooey@hirschkaefer.de>.
 * Distributed under the terms of the MIT License.
 */


#include <system_revision.h>


// Haiku revision (hrev). Will be set when copying libroot.so to the image.
// Lives in a separate section so that it can easily be found.
static char sHaikuRevision[SYSTEM_REVISION_LENGTH]
	__attribute__((section("_haiku_revision")));


const char*
#if defined(_KERNEL_MODE) || defined(_BOOT_MODE)
get_haiku_revision(void)
#else
__get_haiku_revision(void)
#endif
{
	return sHaikuRevision;
}
