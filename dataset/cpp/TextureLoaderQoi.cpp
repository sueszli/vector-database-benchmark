#include "TextureLoaderQoi.h"

#if defined(WITH_QOI)

#define QOI_IMPLEMENTATION
#define QOI_DECODE_ONLY
#define QOI_NO_STDIO
#if defined(__has_include)
#	if __has_include("../../../Libs/Includes/qoi.h")
#		define __HAS_LOCAL_QOI
#	endif
#endif
#if defined(__HAS_LOCAL_QOI)
#	include "../../../Libs/Includes/qoi.h"
#else
#	include <qoi.h>
#endif

using namespace Death::IO;

namespace nCine
{
	TextureLoaderQoi::TextureLoaderQoi(std::unique_ptr<Stream> fileHandle)
		: ITextureLoader(std::move(fileHandle))
	{
		RETURN_ASSERT_MSG(fileHandle_->IsValid(), "File \"%s\" cannot be opened", fileHandle_->GetPath().data());

		auto fileSize = fileHandle_->GetSize();
		if (fileSize < QOI_HEADER_SIZE || fileSize > 64 * 1024 * 1024) {
			// 64 MB file size limit, files are usually smaller than 1MB
			return;
		}

		auto buffer = std::make_unique<char[]>(fileSize);
		fileHandle_->Read(buffer.get(), fileSize);

		qoi_desc desc = { };
		void* data = qoi_decode(buffer.get(), fileSize, &desc, 4);
		if (data == nullptr) {
			return;
		}

		int imageSize = desc.width * desc.height * desc.channels;
		pixels_ = std::make_unique<GLubyte[]>(imageSize);
		// TODO: remove this additional copy
		memcpy(pixels_.get(), data, imageSize);
		QOI_FREE(data);

		width_ = desc.width;
		height_ = desc.height;
		mipMapCount_ = 1;
		texFormat_ = TextureFormat(GL_RGBA8);

		hasLoaded_ = true;
	}
}

#endif