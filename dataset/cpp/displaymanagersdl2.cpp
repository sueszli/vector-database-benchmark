#include <eepp/system/sys.hpp>
#include <eepp/window/backend/SDL2/base.hpp>
#include <eepp/window/backend/SDL2/displaymanagersdl2.hpp>

#if EE_PLATFORM == EE_PLATFORM_EMSCRIPTEN
#include <emscripten.h>
#endif

namespace EE { namespace Window { namespace Backend { namespace SDL2 {

#if EE_PLATFORM == EE_PLATFORM_WIN
static bool DISPLAY_REQUEST_DPI_AWARENESS = true;

void DisplayManagerSDL2::setDPIAwareness() {
	if ( DISPLAY_REQUEST_DPI_AWARENESS ) {
		void* user32 = Sys::loadObject( "user32.dll" );
		if ( user32 ) {
			int ( *SetProcessDPIAware )() =
				(int ( * )())Sys::loadFunction( user32, "SetProcessDPIAware" );
			if ( SetProcessDPIAware ) {
				SetProcessDPIAware();
				DISPLAY_REQUEST_DPI_AWARENESS = false;
			}
		}
	}
}
#endif

DisplaySDL2::DisplaySDL2( int index ) : Display( index ) {}

std::string DisplaySDL2::getName() const {
	return std::string( SDL_GetDisplayName( index ) );
}

Rect DisplaySDL2::getBounds() const {
	SDL_Rect r;
	if ( SDL_GetDisplayBounds( index, &r ) == 0 )
		return Rect( r.x, r.y, r.w, r.h );
	return Rect();
}

Float DisplaySDL2::getDPI() {
#if EE_PLATFORM == EE_PLATFORM_EMSCRIPTEN
	return 96.f * emscripten_get_device_pixel_ratio();
#else
#if EE_PLATFORM == EE_PLATFORM_WIN
	DisplayManagerSDL2::setDPIAwareness();
#endif
	float ddpi, hdpi, vdpi;
	if ( 0 == SDL_GetDisplayDPI( 0, &ddpi, &hdpi, &vdpi ) )
		return ddpi;
	return 96.f;
#endif
}

const int& DisplaySDL2::getIndex() const {
	return index;
}

const std::vector<DisplayMode>& DisplaySDL2::getModes() const {
	if ( displayModes.empty() ) {
		int count = SDL_GetNumDisplayModes( index );

		if ( count > 0 ) {
			for ( int mode_index = 0; mode_index < count; mode_index++ ) {
				SDL_DisplayMode mode;

				if ( SDL_GetDisplayMode( index, mode_index, &mode ) == 0 ) {
					displayModes.push_back(
						DisplayMode( mode.w, mode.h, mode.refresh_rate, index ) );
				}
			}
		}
	}

	return displayModes;
}

Uint32 DisplaySDL2::getRefreshRate() const {
	return getCurrentMode().RefreshRate;
}

Sizeu DisplaySDL2::getSize() const {
	DisplayMode mode( getCurrentMode() );
	return { mode.Width, mode.Height };
}

DisplayMode DisplaySDL2::getCurrentMode() const {
	SDL_DisplayMode mode;

	if ( SDL_GetCurrentDisplayMode( index, &mode ) == 0 ) {
		return DisplayMode( mode.w, mode.h, mode.refresh_rate, index );
	}

	return DisplayMode( 0, 0, 0, 0 );
}

DisplayMode DisplaySDL2::getClosestDisplayMode( DisplayMode wantedMode ) const {
	SDL_DisplayMode target, mode;

	target.w = wantedMode.Width;
	target.h = wantedMode.Height;
	target.format = 0;
	target.refresh_rate = wantedMode.RefreshRate;
	target.driverdata = 0;

	if ( SDL_GetClosestDisplayMode( 0, &target, &mode ) != NULL ) {
		return DisplayMode( mode.w, mode.h, mode.refresh_rate, index );
	}

	return DisplayMode( 0, 0, 0, 0 );
}

Rect DisplaySDL2::getUsableBounds() const {
	SDL_Rect r;
	if ( SDL_GetDisplayUsableBounds( index, &r ) == 0 )
		return Rect( r.x, r.y, r.w, r.h );
	return Rect();
}

int DisplayManagerSDL2::getDisplayCount() {
#if EE_PLATFORM == EE_PLATFORM_WIN
	setDPIAwareness();
#endif
	if ( !SDL_WasInit( SDL_INIT_VIDEO ) )
		SDL_Init( SDL_INIT_VIDEO );
	return SDL_GetNumVideoDisplays();
}

Display* DisplayManagerSDL2::getDisplayIndex( int index ) {
	if ( displays.empty() ) {
		int count = getDisplayCount();

		if ( count > 0 ) {
			for ( int i = 0; i < count; i++ ) {
				displays.push_back( eeNew( DisplaySDL2, ( i ) ) );
			}
		}
	}

	return index >= 0 && index < (Int32)displays.size() ? displays[index] : NULL;
}

void DisplayManagerSDL2::enableScreenSaver() {
	SDL_EnableScreenSaver();
	SDL_SetHint( SDL_HINT_VIDEO_ALLOW_SCREENSAVER, "1" );
}

void DisplayManagerSDL2::disableScreenSaver() {
	SDL_DisableScreenSaver();
	SDL_SetHint( SDL_HINT_VIDEO_ALLOW_SCREENSAVER, "0" );
}

void DisplayManagerSDL2::enableMouseFocusClickThrough() {
	SDL_SetHint( SDL_HINT_MOUSE_FOCUS_CLICKTHROUGH, "1" );
}

void DisplayManagerSDL2::disableMouseFocusClickThrough() {
	SDL_SetHint( SDL_HINT_MOUSE_FOCUS_CLICKTHROUGH, "0" );
}

void DisplayManagerSDL2::disableBypassCompositor() {
#ifdef SDL_HINT_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR
	SDL_SetHint( SDL_HINT_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR, "0" );
#endif
}

void DisplayManagerSDL2::enableBypassCompositor() {
#ifdef SDL_HINT_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR
	SDL_SetHint( SDL_HINT_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR, "1" );
#endif
}

}}}} // namespace EE::Window::Backend::SDL2
