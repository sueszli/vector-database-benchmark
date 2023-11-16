#include <eepp/graphics/textureatlas.hpp>

namespace EE { namespace Graphics {

TextureAtlas* TextureAtlas::New( const std::string& name ) {
	return eeNew( TextureAtlas, ( name ) );
}

TextureAtlas::TextureAtlas( const std::string& name ) : ResourceManager<TextureRegion>() {
	setName( name );
}

TextureAtlas::~TextureAtlas() {}

const std::string& TextureAtlas::getName() const {
	return mName;
}

void TextureAtlas::setName( const std::string& name ) {
	mName = name;
	mId = String::hash( mName );
}

const std::string& TextureAtlas::getPath() const {
	return mPath;
}

void TextureAtlas::setPath( const std::string& path ) {
	mPath = path;
}

const String::HashType& TextureAtlas::getId() const {
	return mId;
}

TextureRegion* TextureAtlas::add( TextureRegion* textureRegion ) {
	return ResourceManager<TextureRegion>::add( textureRegion );
}

TextureRegion* TextureAtlas::add( const Uint32& TexId, const std::string& Name ) {
	return add( TextureRegion::New( TexId, Name ) );
}

TextureRegion* TextureAtlas::add( const Uint32& TexId, const Rect& SrcRect,
								  const std::string& Name ) {
	return add( TextureRegion::New( TexId, SrcRect, Name ) );
}

TextureRegion* TextureAtlas::add( const Uint32& TexId, const Rect& SrcRect, const Sizef& DestSize,
								  const std::string& Name ) {
	return add( TextureRegion::New( TexId, SrcRect, DestSize, Name ) );
}

TextureRegion* TextureAtlas::add( const Uint32& TexId, const Rect& SrcRect, const Sizef& DestSize,
								  const Vector2i& Offset, const std::string& Name ) {
	return add( TextureRegion::New( TexId, SrcRect, DestSize, Offset, Name ) );
}

TextureRegion* TextureAtlas::add( Texture* tex, const std::string& Name ) {
	return add( TextureRegion::New( tex, Name ) );
}

TextureRegion* TextureAtlas::add( Texture* tex, const Rect& SrcRect, const std::string& Name ) {
	return add( TextureRegion::New( tex, SrcRect, Name ) );
}

TextureRegion* TextureAtlas::add( Texture* tex, const Rect& SrcRect, const Sizef& DestSize,
								  const std::string& Name ) {
	return add( TextureRegion::New( tex, SrcRect, DestSize, Name ) );
}

TextureRegion* TextureAtlas::add( Texture* tex, const Rect& SrcRect, const Sizef& DestSize,
								  const Vector2i& Offset, const std::string& Name ) {
	return add( TextureRegion::New( tex, SrcRect, DestSize, Offset, Name ) );
}

Uint32 TextureAtlas::getCount() {
	return ResourceManager<TextureRegion>::getCount();
}

void TextureAtlas::setTextures( std::vector<Texture*> textures ) {
	mTextures = textures;
}

Texture* TextureAtlas::getTexture( const Uint32& texnum ) const {
	eeASSERT( texnum < mTextures.size() );
	return mTextures[texnum];
}

Uint32 TextureAtlas::getTexturesCount() {
	return mTextures.size();
}

}} // namespace EE::Graphics
