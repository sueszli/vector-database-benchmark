#include <eepp/window/backend/SDL2/cursorsdl2.hpp>

#ifdef EE_BACKEND_SDL2

#include <eepp/graphics/image.hpp>

namespace EE { namespace Window { namespace Backend { namespace SDL2 {

CursorSDL::CursorSDL( Texture* tex, const Vector2i& hotspot, const std::string& name,
					  EE::Window::Window* window ) :
	Cursor( tex, hotspot, name, window ), mCursor( NULL ) {
	create();
}

CursorSDL::CursorSDL( Graphics::Image* img, const Vector2i& hotspot, const std::string& name,
					  EE::Window::Window* window ) :
	Cursor( img, hotspot, name, window ), mCursor( NULL ) {
	create();
}

CursorSDL::CursorSDL( const std::string& path, const Vector2i& hotspot, const std::string& name,
					  EE::Window::Window* window ) :
	Cursor( path, hotspot, name, window ), mCursor( NULL ) {
	create();
}

CursorSDL::~CursorSDL() {
	if ( NULL != mCursor )
		SDL_FreeCursor( mCursor );
}

void CursorSDL::create() {
	if ( NULL == mImage )
		return;

	Uint32 rmask, gmask, bmask, amask;
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
	rmask = 0xff000000;
	gmask = 0x00ff0000;
	bmask = 0x0000ff00;
	amask = 0x000000ff;
#else
	rmask = 0x000000ff;
	gmask = 0x0000ff00;
	bmask = 0x00ff0000;
	amask = 0xff000000;
#endif
	SDL_Surface* TempGlyphSheet =
		SDL_CreateRGBSurface( SDL_SWSURFACE, mImage->getWidth(), mImage->getHeight(),
							  mImage->getChannels() * 8, rmask, gmask, bmask, amask );

	SDL_LockSurface( TempGlyphSheet );

	Uint32 ssize = TempGlyphSheet->w * TempGlyphSheet->h * mImage->getChannels();

	Uint8* Ptr = mImage->getPixels();

	for ( Uint32 i = 0; i < ssize; i++ ) {
		( static_cast<Uint8*>( TempGlyphSheet->pixels ) )[i] = Ptr[i];
	}

	mCursor = SDL_CreateColorCursor( TempGlyphSheet, mHotSpot.x, mHotSpot.y );

	SDL_UnlockSurface( TempGlyphSheet );

	SDL_FreeSurface( TempGlyphSheet );
}

SDL_Cursor* CursorSDL::GetCursor() const {
	return mCursor;
}

}}}} // namespace EE::Window::Backend::SDL2

#endif
