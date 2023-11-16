#ifndef EE_WINDOWCCURSORSDL2_HPP
#define EE_WINDOWCCURSORSDL2_HPP

#include <eepp/window/backend/SDL2/base.hpp>
#include <eepp/window/cursor.hpp>

#ifdef EE_BACKEND_SDL2

using namespace EE::Window;

namespace EE { namespace Window { namespace Backend { namespace SDL2 {

class CursorSDL : public Cursor {
  public:
	SDL_Cursor* GetCursor() const;

  protected:
	friend class CursorManagerSDL;

	SDL_Cursor* mCursor;

	CursorSDL( Texture* tex, const Vector2i& hotspot, const std::string& name,
			   EE::Window::Window* window );

	CursorSDL( Graphics::Image* img, const Vector2i& hotspot, const std::string& name,
			   EE::Window::Window* window );

	CursorSDL( const std::string& path, const Vector2i& hotspot, const std::string& name,
			   EE::Window::Window* window );

	virtual ~CursorSDL();

	void create();
};

}}}} // namespace EE::Window::Backend::SDL2

#endif

#endif
