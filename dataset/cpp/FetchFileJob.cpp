/*
 * Copyright 2011-2021, Haiku, Inc. All Rights Reserved.
 * Distributed under the terms of the MIT License.
 *
 * Authors:
 *		Axel Dörfler <axeld@pinc-software.de>
 *		Rene Gollent <rene@gollent.com>
 *		Oliver Tappe <zooey@hirschkaefer.de>
 *		Stephan Aßmus <superstippi@gmx.de>
 */


#include "FetchFileJob.h"

#include <stdio.h>
#include <sys/wait.h>

#include <Path.h>

#ifdef HAIKU_TARGET_PLATFORM_HAIKU
#	include <HttpRequest.h>
#	include <UrlRequest.h>
#	include <UrlProtocolRoster.h>
using namespace BPrivate::Network;
#endif

#include "FetchUtils.h"


namespace BPackageKit {

namespace BPrivate {


#ifdef HAIKU_TARGET_PLATFORM_HAIKU

FetchFileJob::FetchFileJob(const BContext& context, const BString& title,
	const BString& fileURL, const BEntry& targetEntry)
	:
	inherited(context, title),
	fFileURL(fileURL),
	fTargetEntry(targetEntry),
	fTargetFile(&targetEntry, B_CREATE_FILE | B_WRITE_ONLY),
	fError(B_ERROR),
	fDownloadProgress(0.0)
{
}


FetchFileJob::~FetchFileJob()
{
}


float
FetchFileJob::DownloadProgress() const
{
	return fDownloadProgress;
}


const char*
FetchFileJob::DownloadURL() const
{
	return fFileURL.String();
}


const char*
FetchFileJob::DownloadFileName() const
{
	return fTargetEntry.Name();
}


off_t
FetchFileJob::DownloadBytes() const
{
	return fBytes;
}


off_t
FetchFileJob::DownloadTotalBytes() const
{
	return fTotalBytes;
}


status_t
FetchFileJob::Execute()
{
	status_t result = fTargetFile.InitCheck();
	if (result != B_OK)
		return result;

	result = FetchUtils::SetFileType(fTargetFile,
		"application/x-vnd.haiku-package");
	if (result != B_OK) {
		fprintf(stderr, "failed to set file type for '%s': %s\n",
			DownloadFileName(), strerror(result));
	}

	do {
		BUrlRequest* request = BUrlProtocolRoster::MakeRequest(fFileURL.String(),
			&fTargetFile, this);
		if (request == NULL)
			return B_BAD_VALUE;

		// Try to resume the download where we left off
		off_t currentPosition;
		BHttpRequest* http = dynamic_cast<BHttpRequest*>(request);
		if (http != NULL && fTargetFile.GetSize(&currentPosition) == B_OK
			&& currentPosition > 0) {
			http->SetRangeStart(currentPosition);
			fTargetFile.Seek(0, SEEK_END);
		}

		thread_id thread = request->Run();
		wait_for_thread(thread, NULL);

		if (fError != B_IO_ERROR && fError != B_DEV_TIMEOUT && fError != B_OK) {
			// Something went wrong with the download and it's not just a
			// timeout. Remove whatever we wrote to the file, since the content
			// returned by the server was probably not part of the file.
			fTargetFile.SetSize(currentPosition);
		}
	} while (fError == B_IO_ERROR || fError == B_DEV_TIMEOUT);

	if (fError == B_OK) {
		result = FetchUtils::MarkDownloadComplete(fTargetFile);
		if (result != B_OK) {
			fprintf(stderr, "failed to mark download '%s' as complete: %s\n",
				DownloadFileName(), strerror(result));
		}
	}

	return fError;
}


void
FetchFileJob::DownloadProgress(BUrlRequest*, off_t bytesReceived,
	off_t bytesTotal)
{
	if (bytesTotal != 0) {
		fBytes = bytesReceived;
		fTotalBytes = bytesTotal;
		fDownloadProgress = (float)bytesReceived/bytesTotal;
		NotifyStateListeners();
	}
}


void
FetchFileJob::RequestCompleted(BUrlRequest* request, bool success)
{
	fError = request->Status();

	if (success) {
		const BHttpResult* httpResult = dynamic_cast<const BHttpResult*>
			(&request->Result());
		if (httpResult != NULL) {
			uint16 code = httpResult->StatusCode();
			uint16 codeClass = BHttpRequest::StatusCodeClass(code);

			switch (codeClass) {
				case B_HTTP_STATUS_CLASS_CLIENT_ERROR:
				case B_HTTP_STATUS_CLASS_SERVER_ERROR:
					fError = B_IO_ERROR;
					break;
				default:
					fError = B_OK;
					break;
			}
			switch (code) {
				case B_HTTP_STATUS_OK:
				case B_HTTP_STATUS_PARTIAL_CONTENT:
					fError = B_OK;
					break;
				case B_HTTP_STATUS_REQUEST_TIMEOUT:
				case B_HTTP_STATUS_GATEWAY_TIMEOUT:
					fError = B_DEV_TIMEOUT;
					break;
				case B_HTTP_STATUS_NOT_IMPLEMENTED:
					fError = B_NOT_SUPPORTED;
					break;
				case B_HTTP_STATUS_UNSUPPORTED_MEDIA_TYPE:
					fError = B_UNKNOWN_MIME_TYPE;
					break;
				case B_HTTP_STATUS_REQUESTED_RANGE_NOT_SATISFIABLE:
					fError = B_RESULT_NOT_REPRESENTABLE; // alias for ERANGE
					break;
				case B_HTTP_STATUS_UNAUTHORIZED:
					fError = B_PERMISSION_DENIED;
					break;
				case B_HTTP_STATUS_FORBIDDEN:
				case B_HTTP_STATUS_METHOD_NOT_ALLOWED:
				case B_HTTP_STATUS_NOT_ACCEPTABLE:
					fError = B_NOT_ALLOWED;
					break;
				case B_HTTP_STATUS_NOT_FOUND:
					fError = B_NAME_NOT_FOUND;
					break;
				case B_HTTP_STATUS_BAD_GATEWAY:
					fError = B_BAD_DATA;
					break;
				default:
					break;
			}
		}
	}
}


void
FetchFileJob::Cleanup(status_t jobResult)
{
	if (jobResult != B_OK)
		fTargetEntry.Remove();
}


#else // HAIKU_TARGET_PLATFORM_HAIKU


FetchFileJob::FetchFileJob(const BContext& context, const BString& title,
	const BString& fileURL, const BEntry& targetEntry)
	:
	inherited(context, title),
	fFileURL(fileURL),
	fTargetEntry(targetEntry),
	fTargetFile(&targetEntry, B_CREATE_FILE | B_WRITE_ONLY),
	fDownloadProgress(0.0)
{
}


FetchFileJob::~FetchFileJob()
{
}


float
FetchFileJob::DownloadProgress() const
{
	return fDownloadProgress;
}


const char*
FetchFileJob::DownloadURL() const
{
	return fFileURL.String();
}


const char*
FetchFileJob::DownloadFileName() const
{
	return fTargetEntry.Name();
}


off_t
FetchFileJob::DownloadBytes() const
{
	return fBytes;
}


off_t
FetchFileJob::DownloadTotalBytes() const
{
	return fTotalBytes;
}


status_t
FetchFileJob::Execute()
{
	return B_UNSUPPORTED;
}


void
FetchFileJob::Cleanup(status_t jobResult)
{
	if (jobResult != B_OK)
		fTargetEntry.Remove();
}


#endif // HAIKU_TARGET_PLATFORM_HAIKU

}	// namespace BPrivate

}	// namespace BPackageKit
