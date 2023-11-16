//#include "return_macros.h"
//#include "TextureLoaderWebP.h"
//
//namespace nCine {
//
/////////////////////////////////////////////////////////////
//// CONSTRUCTORS and DESTRUCTOR
/////////////////////////////////////////////////////////////
//
//TextureLoaderWebP::TextureLoaderWebP(std::unique_ptr<IFile> fileHandle)
//    : ITextureLoader(std::move(fileHandle))
//{
//	LOGI("Loading \"%s\"", fileHandle_->filename());
//
//	// Loading the whole file in memory
//	RETURN_ASSERT_MSG(fileHandle_->IsValid(), "File \"%s\" cannot be opened", fileHandle_->GetPath());
//	const long int fileSize = fileHandle_->size();
//	std::unique_ptr<unsigned char[]> fileBuffer = std::make_unique<unsigned char[]>(fileSize);
//	fileHandle_->read(fileBuffer.get(), fileSize);
//
//	if (WebPGetInfo(fileBuffer.get(), fileSize, &width_, &height_) == 0)
//	{
//		fileBuffer.reset(nullptr);
//		RETURN_MSG("Cannot read WebP header");
//	}
//
//	LOGI("Header found: w:%d h:%d", width_, height_);
//
//	WebPBitstreamFeatures features;
//	if (WebPGetFeatures(fileBuffer.get(), fileSize, &features) != VP8_STATUS_OK)
//	{
//		fileBuffer.reset(nullptr);
//		RETURN_MSG("Cannot retrieve WebP features from headers");
//	}
//
//	LOGI("Bitstream features found: alpha:%d animation:%d format:%d",
//	       features.has_alpha, features.has_animation, features.format);
//
//	mipMapCount_ = 1; // No MIP Mapping
//	texFormat_ = features.has_alpha ? TextureFormat(GL_RGBA8) : TextureFormat(GL_RGB8);
//	dataSize_ = width_ * height_ * texFormat_.numChannels();
//	pixels_ = std::make_unique<unsigned char[]>(dataSize_);
//
//	if (features.has_alpha)
//	{
//		if (WebPDecodeRGBAInto(fileBuffer.get(), fileSize, pixels_.get(), dataSize_, width_ * 4) == nullptr)
//		{
//			fileBuffer.reset(nullptr);
//			pixels_.reset(nullptr);
//			RETURN_MSG("Cannot decode RGBA WebP image");
//		}
//	}
//	else
//	{
//		if (WebPDecodeRGBInto(fileBuffer.get(), fileSize, pixels_.get(), dataSize_, width_ * 3) == nullptr)
//		{
//			fileBuffer.reset(nullptr);
//			pixels_.reset(nullptr);
//			RETURN_MSG("Cannot decode RGB WebP image");
//		}
//	}
//
//	hasLoaded_ = true;
//}
//
//}
