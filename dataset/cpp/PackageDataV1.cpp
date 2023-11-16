/*
 * Copyright 2009, Ingo Weinhold, ingo_weinhold@gmx.de.
 * Distributed under the terms of the MIT License.
 */


#include <package/hpkg/v1/PackageData.h>

#include <string.h>

#include <package/hpkg/v1/HPKGDefsPrivate.h>


namespace BPackageKit {

namespace BHPKG {

namespace V1 {


using namespace BPrivate;


BPackageData::BPackageData()
	:
	fCompressedSize(0),
	fUncompressedSize(0),
	fChunkSize(0),
	fCompression(B_HPKG_COMPRESSION_NONE),
	fEncodedInline(true)
{
}


void
BPackageData::SetData(uint64 size, uint64 offset)
{
	fUncompressedSize = fCompressedSize = size;
	fOffset = offset;
	fEncodedInline = false;
}


void
BPackageData::SetData(uint8 size, const void* data)
{
	fUncompressedSize = fCompressedSize = size;
	if (size > 0)
		memcpy(fInlineData, data, size);
	fEncodedInline = true;
}


}	// namespace V1

}	// namespace BHPKG

}	// namespace BPackageKit
