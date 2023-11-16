/*
 * Copyright 2021-2022, Andrew Lindesay <apl@lindesay.co.nz>.
 * All rights reserved. Distributed under the terms of the MIT License.
 */


#include "AbstractProcessNode.h"

#include <unistd.h>

#include "AbstractProcess.h"
#include "Logger.h"


#define SPIN_DELAY_MI 500 * 1000
	// half of a second

#define TIMEOUT_UNTIL_STARTED_SECS 10
#define TIMEOUT_UNTIL_STOPPED_SECS 10


AbstractProcessNode::AbstractProcessNode(AbstractProcess* process)
	:
	fLock(),
	fListener(NULL),
	fProcess(process)
{
}


AbstractProcessNode::~AbstractProcessNode()
{
	delete fProcess;
}


AbstractProcess*
AbstractProcessNode::Process() const
{
	return fProcess;
}


bool
AbstractProcessNode::IsRunning()
{
	return Process()->ProcessState() != PROCESS_COMPLETE;
}


void
AbstractProcessNode::SetListener(ProcessListener* listener)
{
	if (fListener != listener) {
		AutoLocker<BLocker> locker(&fLock);
		fListener = listener;
	}
}


/*! This method will spin-lock the thread until the process is in one of the
    states defined by the mask.
 */

status_t
AbstractProcessNode::_SpinUntilProcessState(
	uint32 desiredStatesMask, int32 timeoutSeconds)
{
	bigtime_t start = system_time();
	while (true) {
		if ((Process()->ProcessState() & desiredStatesMask) != 0)
			return B_OK;

		usleep(SPIN_DELAY_MI);

		int32 secondElapsed = static_cast<int32>(
			(system_time() - start) / (1000 * 1000));

		if (timeoutSeconds > 0 && secondElapsed > timeoutSeconds) {
			HDERROR("[Node<%s>] timeout waiting for process state after %"
				B_PRIi32 " seconds", Process()->Name(), secondElapsed);
			return B_ERROR;
		}
	}
}


void
AbstractProcessNode::AddPredecessor(AbstractProcessNode *node)
{
	fPredecessorNodes.AddItem(node);
	node->_AddSuccessor(this);
}


int32
AbstractProcessNode::CountPredecessors() const
{
	return fPredecessorNodes.CountItems();
}


AbstractProcessNode*
AbstractProcessNode::PredecessorAt(int32 index) const
{
	return fPredecessorNodes.ItemAt(index);
}


bool
AbstractProcessNode::AllPredecessorsComplete() const
{
	for (int32 i = 0; i < CountPredecessors(); i++) {
		if (PredecessorAt(i)->Process()->ProcessState() != PROCESS_COMPLETE)
			return false;
	}

	return true;
}


void
AbstractProcessNode::_AddSuccessor(AbstractProcessNode* node)
{
	fSuccessorNodes.AddItem(node);
}


int32
AbstractProcessNode::CountSuccessors() const
{
	return fSuccessorNodes.CountItems();
}


AbstractProcessNode*
AbstractProcessNode::SuccessorAt(int32 index) const
{
	return fSuccessorNodes.ItemAt(index);
}


BString
AbstractProcessNode::LogReport()
{
	return Process()->LogReport();
}