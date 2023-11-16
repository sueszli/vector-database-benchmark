/*
 * Copyright 2011, Oliver Tappe <zooey@hirschkaefer.de>
 * Distributed under the terms of the MIT License.
 */


#include <stdio.h>

#include "JobStateListener.h"
#include "pkgman.h"


using BSupportKit::BJob;


JobStateListener::JobStateListener(uint32 flags)
	:
	fFlags(flags)
{
}


void
JobStateListener::JobStarted(BJob* job)
{
	printf("%s ...\n", job->Title().String());
}


void
JobStateListener::JobSucceeded(BJob* job)
{
}


void
JobStateListener::JobFailed(BJob* job)
{
	BString error = job->ErrorString();
	if (error.Length() > 0) {
		error.ReplaceAll("\n", "\n*** ");
		fprintf(stderr, "%s", error.String());
	}
	if ((fFlags & EXIT_ON_ERROR) != 0)
		DIE(job->Result(), "failed!");
}


void
JobStateListener::JobAborted(BJob* job)
{
	if ((fFlags & EXIT_ON_ABORT) != 0)
		DIE(job->Result(), "aborted");
}
