#ifndef EE_GRAPHICSCTEXTUREATLAS
#define EE_GRAPHICSCTEXTUREATLAS

/*
** Copyright (c) 2009 by John W. Ratcliff mailto:jratcliffscarab@gmail.com
**
** The MIT license:
**
** Permission is hereby granted, free of charge, to any person obtaining a copy
** of this software and associated documentation files (the "Software"), to deal
** in the Software without restriction, including without limitation the rights
** to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
** copies of the Software, and to permit persons to whom the Software is furnished
** to do so, subject to the following conditions:
**
** The above copyright notice and this permission notice shall be included in all
** copies or substantial portions of the Software.

** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
** IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
** FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
** AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
** WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
** CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*
**	This implementation is based on the John W. Ratcliff texture atlas implementation.
** Implementation differs from the original, but i used the base texture atlas algorithm.
*/

#include <eepp/graphics/base.hpp>
#include <eepp/graphics/image.hpp>
#include <eepp/graphics/packerhelper.hpp>
#include <eepp/graphics/texture.hpp>

namespace EE { namespace Graphics {

namespace Private {
class TexturePackerNode;
class TexturePackerTex;
} // namespace Private

using namespace Private;

/** @brief The Texture Packer class is used to create new Texture Atlases.
 *	Atlases can be created indicating the texture atlas size and adding textures to the atlases.
 */
class EE_API TexturePacker {
  public:
	static TexturePacker* New();

	/** Creates a new instance of the texture packer indicating the maximum size of the texture
	 *atlas.
	 *	@param maxWidth The maximum width that the texture atlas will use.
	 *	@param maxHeight The maximum height that the texture atlas will use.
	 *	@param pixelDensity Indicates the device pixel density that represents the resources that
	 *will the packer pack.
	 *	@param textureFilter Indicates with texture filter should be used when the texture of the
	 *atlas is loaded.
	 *	@param forcePowOfTwo Indicates that if the max with and height must be adjusted to fit a
	 *power of two texture.
	 *	@param scalableSVG True if SVG images are scaled to the packer PixelDensity
	 *	@param pixelBorder Indicates how many pixels will be added to separate one image to another
	 *in the texture atlas. Usefull to avoid artifacts when rendered scaled TextureRegions. Use at
	 *least 1 pixel to separate images if you will scale any TextureRegion.
	 *	@param allowChilds When enabled if the atlas does not have enough space left in the image to
	 *put more resources it will create a new image atlas and add the rest of the images in that
	 *images, repeating this process recursivelly until using all source images.
	 *	@param allowFlipping Indicates if the images can be flipped inside the texture atlas. This
	 *is not compatible with eepp ( since it can't flip the textures back to the original
	 *orientation ). So avoid it for eepp.
	 */
	static TexturePacker* New( const Uint32& maxWidth, const Uint32& maxHeight,
							   const Float& pixelDensity = 1, const bool& forcePowOfTwo = true,
							   const bool& scalableSVG = false, const Uint32& pixelBorder = 2,
							   const Texture::Filter& textureFilter = Texture::Filter::Linear,
							   const bool& allowChilds = false, const bool& allowFlipping = false );

	/** Creates a new texture packer ( you will need to call SetOptions before adding any texture or
	 * image ). */
	TexturePacker();

	/** Creates a new texture packer indicating the maximum size of the texture atlas.
	 *	@param maxWidth The maximum width that the texture atlas will use.
	 *	@param maxHeight The maximum height that the texture atlas will use.
	 *	@param pixelDensity Indicates the device pixel density that represents the resources that
	 *will the packer pack.
	 *	@param textureFilter Indicates with texture filter should be used when the texture of the
	 *atlas is loaded.
	 *	@param forcePowOfTwo Indicates that if the max with and height must be adjusted to fit a
	 *power of two texture.
	 *	@param scalableSVG True if SVG images are scaled to the packer PixelDensity
	 *	@param pixelBorder Indicates how many pixels will be added to separate one image to another
	 *in the texture atlas. Usefull to avoid artifacts when rendered scaled TextureRegions. Use at
	 *least 1 pixel to separate images if you will scale any TextureRegion.
	 *	@param allowChilds When enabled if the atlas does not have enough space left in the image to
	 *put more resources it will create a new image atlas and add the rest of the images in that
	 *images, repeating this process recursivelly until using all source images.
	 *	@param allowFlipping Indicates if the images can be flipped inside the texture atlas. This
	 *is not compatible with eepp ( since it can't flip the textures back to the original
	 *orientation ). So avoid it for eepp.
	 */
	TexturePacker( const Uint32& maxWidth, const Uint32& maxHeight, const Float& pixelDensity = 1,
				   const bool& forcePowOfTwo = true, const bool& scalableSVG = false,
				   const Uint32& pixelBorder = 2,
				   const Texture::Filter& textureFilter = Texture::Filter::Linear,
				   const bool& allowChilds = false, const bool& allowFlipping = false );

	~TexturePacker();

	/** Adds a image/texture from its path to the texture atlas.
	 *	@param TexturePath The image path. */
	bool addTexture( const std::string& TexturePath );

	/** Adds a image to the texture atlas. The image instance must remain in memory until the
	 * texture atlas is saved. */
	bool addImage( Image* Img, const std::string& Name );

	/** Adds a directory with images. It will try to add all the images inside that directory to the
	 * texture atlas.  */
	bool addTexturesPath( std::string TexturesPath );

	/** After adding all the images that will be used to create the texture atlas. Packing the
	 * textures will generate the texture atlas information ( it will fit the images inside the
	 * texture atlas, etc ).
	 * @return The amount of pixels used for the new image or the total area used. 0 means it
	 * failed.
	 */
	Int32 packTextures();

	/** Save the texture atlas to a file, in the indicated format. If PackTexture() has not been
	 *called, it will be called automatically by the function ( so you don't need to call it ).
	 *	@param Filepath The path were it will be saved the new texture atlas.
	 *	@param Format The image format of the new texture atlas.
	 *	@param KeepExtensions Indicates if the extensions of the image files must be saved. Usually
	 * you want to find the TextureRegions by its name without extension, but this can be changed
	 * here.
	 */
	void save( const std::string& Filepath,
			   const Image::SaveType& Format = Image::SaveType::SAVE_TYPE_PNG,
			   const bool& KeepExtensions = false );

	/** Clear all the textures added */
	void close();

	/** First of all you need to set at least the max dimensions of the texture atlas. If the
	 *instance of the texture packer was created without indicating this data, this must be called
	 *before adding any texture or image.
	 *	@param maxWidth The maximum width that the texture atlas will use.
	 *	@param maxHeight The maximum height that the texture atlas will use.
	 *	@param pixelDensity Indicates the device pixel density that represents the resources that
	 *will the packer pack.
	 *	@param textureFilter Indicates with texture filter should be used when the texture of the
	 *atlas is loaded.
	 *	@param forcePowOfTwo Indicates that if the max with and height must be adjusted to fit a
	 *power of two texture.
	 *	@param scalableSVG True if SVG images are scaled to the packer PixelDensity
	 *	@param pixelBorder Indicates how many pixels will be added to separate one image to another
	 *in the texture atlas. Usefull to avoid artifacts when rendered scaled TextureRegions. Use at
	 *least 1 pixel to separate images if you will scale any TextureRegion.
	 *	@param allowChilds When enabled if the atlas does not have enough space left in the image to
	 *put more resources it will create a new image atlas and add the rest of the images in that
	 *images, repeating this process recursivelly until using all source images.
	 *	@param allowFlipping Indicates if the images can be flipped inside the texture atlas. This
	 *is not compatible with eepp ( since it can't flip the textures back to the original
	 *orientation ). So avoid it for eepp.
	 */
	void setOptions( const Uint32& maxWidth, const Uint32& maxHeight, const Float& pixelDensity = 1,
					 const bool& forcePowOfTwo = true, const bool& scalableSVG = false,
					 const Uint32& pixelBorder = 2,
					 const Texture::Filter& textureFilter = Texture::Filter::Linear,
					 const bool& allowChilds = false, const bool& allowFlipping = false );

	/** @return The texture atlas to generate width. */
	const Int32& getWidth() const;

	/** @return The texture atlas to generate height */
	const Int32& getHeight() const;

	/** @return If the texture atlas has already been saved, returns the file path to the texture
	 * atlas. */
	const std::string& getFilepath() const;

  protected:
	enum PackStrategy { PackBig, PackTiny, PackFail };

	std::vector<TexturePackerTex*> mTextures;

	Int32 mTotalArea;
	TexturePackerNode* mFreeList;
	Int32 mWidth;
	Int32 mHeight;
	Sizei mMaxSize;
	bool mPacked;
	bool mAllowFlipping;
	bool mAllowChilds;
	TexturePacker* mChild;
	Int32 mStrategy;
	Int32 mCount;
	TexturePacker* mParent;
	std::string mFilepath;
	Int32 mPlacedCount;
	bool mForcePowOfTwo;
	Int32 mPixelBorder;
	Uint32 mPixelDensity; /* The multiplier value * 100 ( example: PD = 2, here is 200 ) */
	Texture::Filter mTextureFilter;
	bool mKeepExtensions;
	bool mScalableSVG;
	Image::SaveType mFormat;

	TexturePacker* getChild() const;

	TexturePacker* getParent() const;

	std::vector<TexturePackerTex*>* getTexturePackPtr();

	void childSave( const Image::SaveType& Format );

	void saveTextureRegions();

	void newFree( Int32 x, Int32 y, Int32 getWidth, Int32 getHeight );

	bool mergeNodes();

	void validate();

	TexturePackerTex* getLonguestEdge();

	TexturePackerTex* getShortestEdge();

	Int32 getChildCount();

	const Int32& getPlacedCount() const;

	sTextureHdr createTextureHdr( TexturePacker* Packer );

	void createTextureRegionsHdr( TexturePacker* Packer,
								  std::vector<sTextureRegionHdr>& TextureRegions );

	TexturePackerNode* getBestFit( TexturePackerTex* t, TexturePackerNode** prevBestFit,
								   Int32* EdgeCount );

	void insertTexture( TexturePackerTex* t, TexturePackerNode* bestFit, Int32 edgeCount,
						TexturePackerNode* previousBestFit );

	void addBorderToTextures( const Int32& BorderSize );

	void createChild();

	bool addPackerTex( TexturePackerTex* TPack );

	void reset();

	Uint32 getAtlasNumChannels();
};

}} // namespace EE::Graphics

#endif
