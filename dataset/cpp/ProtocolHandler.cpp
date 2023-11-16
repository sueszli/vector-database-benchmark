/*
 * Copyright 2009-2011, Michael Lotz, mmlr@mlotz.ch.
 * Distributed under the terms of the MIT License.
 */


#include <stdlib.h>
#include <ring_buffer.h>
#include <kernel.h>

#include "Driver.h"
#include "HIDCollection.h"
#include "HIDDevice.h"
#include "HIDReport.h"
#include "ProtocolHandler.h"

// includes for the different protocol handlers
#include "JoystickProtocolHandler.h"
#include "KeyboardProtocolHandler.h"
#include "MouseProtocolHandler.h"
#include "TabletProtocolHandler.h"


ProtocolHandler::ProtocolHandler(HIDDevice *device, const char *basePath,
	size_t ringBufferSize)
	:	fStatus(B_NO_INIT),
		fDevice(device),
		fBasePath(basePath),
		fPublishPath(NULL),
		fRingBuffer(NULL),
		fNextHandler(NULL)
{
	if (ringBufferSize > 0) {
		fRingBuffer = create_ring_buffer(ringBufferSize);
		if (fRingBuffer == NULL) {
			TRACE_ALWAYS("failed to create requested ring buffer\n");
			fStatus = B_NO_MEMORY;
			return;
		}
	}

	fStatus = B_OK;
}


ProtocolHandler::~ProtocolHandler()
{
	if (fRingBuffer) {
		delete_ring_buffer(fRingBuffer);
		fRingBuffer = NULL;
	}

	free(fPublishPath);
}


void
ProtocolHandler::SetPublishPath(char *publishPath)
{
	free(fPublishPath);
	fPublishPath = publishPath;
}


void
ProtocolHandler::AddHandlers(HIDDevice &device, ProtocolHandler *&handlerList,
	uint32 &handlerCount)
{
	TRACE("adding protocol handlers\n");

	HIDParser &parser = device.Parser();
	HIDCollection *rootCollection = parser.RootCollection();
	if (rootCollection == NULL)
		return;

	uint32 appCollectionCount = rootCollection->CountChildrenFlat(
		COLLECTION_APPLICATION);
	TRACE("root collection holds %" B_PRIu32 " application collection%s\n",
		appCollectionCount, appCollectionCount != 1 ? "s" : "");

	for (uint32  i = 0; i < appCollectionCount; i++) {
		HIDCollection *collection = rootCollection->ChildAtFlat(
			COLLECTION_APPLICATION, i);
		if (collection == NULL)
			continue;

		TRACE("collection usage page %u usage id %u\n",
			collection->UsagePage(), collection->UsageID());

		// NOTE: The driver publishes devices for all added handlers.

		// TODO: How does this work if a device is not a compound device
		// like a keyboard with built-in touchpad, but allows multiple
		// alternative configurations like a tablet that works as either
		// regular (relative) mouse, or (absolute) tablet?
		KeyboardProtocolHandler::AddHandlers(device, *collection, handlerList);
		JoystickProtocolHandler::AddHandlers(device, *collection, handlerList);
		MouseProtocolHandler::AddHandlers(device, *collection, handlerList);
		TabletProtocolHandler::AddHandlers(device, *collection, handlerList);
	}

	handlerCount = 0;
	ProtocolHandler *handler = handlerList;
	while (handler != NULL) {
		handler = handler->NextHandler();
		handlerCount++;
	}

	if (handlerCount == 0) {
		TRACE_ALWAYS("no handlers for hid device\n");
		return;
	}

	TRACE("added %" B_PRId32 " handlers for hid device\n", handlerCount);
}


status_t
ProtocolHandler::Open(uint32 flags, uint32 *cookie)
{
	return fDevice->Open(this, flags);
}


status_t
ProtocolHandler::Close(uint32 *cookie)
{
	*cookie |= PROTOCOL_HANDLER_COOKIE_FLAG_CLOSED;
		// This lets the handlers know that this user is gone.

	return fDevice->Close(this);
}


status_t
ProtocolHandler::Read(uint32 *cookie, off_t position, void *buffer,
	size_t *numBytes)
{
	TRACE_ALWAYS("unhandled read on protocol handler\n");
	*numBytes = 0;
	return B_ERROR;
}


status_t
ProtocolHandler::Write(uint32 *cookie, off_t position, const void *buffer,
	size_t *numBytes)
{
	TRACE_ALWAYS("unhandled write on protocol handler\n");
	*numBytes = 0;
	return B_ERROR;
}


status_t
ProtocolHandler::Control(uint32 *cookie, uint32 op, void *buffer, size_t length)
{
	TRACE_ALWAYS("unhandled control on protocol handler\n");
	return B_ERROR;
}


int32
ProtocolHandler::RingBufferReadable()
{
	return ring_buffer_readable(fRingBuffer);
}


status_t
ProtocolHandler::RingBufferRead(void *buffer, size_t length)
{
	ring_buffer_user_read(fRingBuffer, (uint8 *)buffer, length);
	return B_OK;
}


status_t
ProtocolHandler::RingBufferWrite(const void *buffer, size_t length)
{
	ring_buffer_write(fRingBuffer, (const uint8 *)buffer, length);
	return B_OK;
}


void
ProtocolHandler::SetNextHandler(ProtocolHandler *nextHandler)
{
	fNextHandler = nextHandler;
}

status_t
ProtocolHandler::IOGetDeviceName(const char *name, void *buffer, size_t length)
{

	if (!IS_USER_ADDRESS(buffer))
		return B_BAD_ADDRESS;

	if (user_strlcpy((char *)buffer, name, length) > 0)
		return B_OK;

	return B_ERROR;
}
