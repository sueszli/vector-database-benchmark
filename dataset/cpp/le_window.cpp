#include "le_window.h"
#include "le_hash_util.h"
#include "le_ui_event.h"
#include "le_log.h"
#include "le_backend_vk.h"

#include "assert.h"
#include <mutex>
#include <vector>
#include <array>
#include <forward_list>
#include <atomic>
#include <string>
#include <cstring> // for memcpy

#define GLFW_INCLUDE_VULKAN
#define GLFW_INCLUDE_NONE
#include "GLFW/glfw3.h"

#if defined( __linux__ )
#	define GLFW_EXPOSE_NATIVE_X11
// #	define GLFW_EXPOSE_NATIVE_WAYLAND
#elif defined( _WIN32 )
#	define GLFW_EXPOSE_NATIVE_WIN32
#endif
#include "GLFW/glfw3native.h"

constexpr size_t EVENT_QUEUE_SIZE = ( 4096 * 4 ) / sizeof( LeUiEvent ); // Allocate a few pages for events
constexpr auto   GAMEPAD_SUBSCRIBERS_SINGLETON_ID = hash_64_fnv1a_const( "le_window_gamepad_subscribers" );

struct le_window_settings_o {
	int          width           = 640;
	int          height          = 480;
	std::string  title           = "Island default window title";
	GLFWmonitor* monitor         = nullptr;
	bool         useEventsQueue  = true;           // whether to use an events queue or not
	uint32_t     active_gamepads = ~uint32_t( 0 ); // bitfield; subscribe to gamepad events for gamepads / joysticks with matching id
};

struct WindowGeometry {
	int x      = 0;
	int y      = 0;
	int width  = 0;
	int height = 0;
};

struct le_window_o {

	GLFWwindow*          window   = nullptr;
	VkSurfaceKHR         mSurface = nullptr;
	VkExtent2D           mSurfaceExtent{};
	le_window_settings_o mSettings{};
	size_t               referenceCount = 0;
	void*                user_data      = nullptr;

	std::atomic<uint32_t>                                  eventQueueBack = 0;        // Event queue currently used to record events
	std::array<std::atomic<uint32_t>, 2>                   numEventsForQueue{ 0, 0 }; // Counter for events per queue (works as arena allocator marker for events queue)
	std::array<std::array<LeUiEvent, EVENT_QUEUE_SIZE>, 2> eventQueue;                // Events queue is double-bufferd, flip happens on `get_event_queue`

	std::array<std::forward_list<std::string>, 2>              eventStringData; // string data associated with events per queue. vector of strings gets deleted on flip.
	std::array<std::forward_list<std::vector<char const*>>, 2> eventStringPtr;  // ptrs to string data associated with events per queue. gets deleted on flip.

	WindowGeometry windowGeometry{};
	bool           isFullscreen = false;
};

struct gamepad_events_subscriber_windows_t {
	std::vector<le_window_o*> windows;
	std::mutex                mtx;
};

static gamepad_events_subscriber_windows_t* gamepad_events_subscribers_singleton_get() {

	static auto logger = le::Log( "le_window" );

	static std::mutex mtx;

	// we need a mutex here to prevent a race condition where two or more threads
	// fight for who can first return the singleton - we want to make sure it only
	// gets initialised once.
	//
	std::unique_lock critical_section( mtx );

	// Attempt to fetch the entry for our singleton from the global, persistent storage.
	//
	void** dict = le_core_produce_dictionary_entry( GAMEPAD_SUBSCRIBERS_SINGLETON_ID );

	// If the entry is empty, allocate a new object and update the dictionary entry.
	if ( *dict == nullptr ) {
		*dict = new gamepad_events_subscriber_windows_t{};
		logger.info( "Created gamepad events subscribers singleton" );
	}

	return static_cast<gamepad_events_subscriber_windows_t*>( *dict );
};

// ----------------------------------------------------------------------
// Check if there is an available index to write at, given an event counter,
// and set eventIdx as a side-effect.
// Returns false if no index in the queue is available for writing.
// Returns true and an available index as a side-effect in eventIdx otherwise.
//
// Note that we limit the return value `eventIdx` to EVENT_QUEUE_SIZE-1
bool event_queue_idx_available( std::atomic<uint32_t>& atomicCounter, uint32_t& eventIdx ) {

	// We post-increment here, so that eventIdx receives the current value of the atomic
	// counter, just before the counter gets incremented.
	eventIdx = atomicCounter++;

	if ( eventIdx >= EVENT_QUEUE_SIZE ) {
		eventIdx = atomicCounter = EVENT_QUEUE_SIZE - 1;
		return false;
	} else {
		return true;
	}
}

// ----------------------------------------------------------------------
static void le_window_gamepad_callback( le_window_o* window, le::UiEvent::GamepadEvent const& ev ) {

	if ( window->mSettings.useEventsQueue ) {

		uint32_t queueIdx = window->eventQueueBack;
		uint32_t eventIdx = 0;

		if ( event_queue_idx_available( window->numEventsForQueue[ queueIdx ], eventIdx ) ) {
			auto& event = window->eventQueue[ queueIdx ][ eventIdx ];
			event.event = le::UiEvent::Type::eGamepad;
			auto& e     = event.gamepad;
			e           = ev;

			static auto logger = le::Log( "le_window" );

		} else {
			// we're over the high - watermark for events, we should probably print a warning.
		}
	}
}

// ----------------------------------------------------------------------
static void glfw_window_key_callback( GLFWwindow* glfwWindow, int key, int scancode, int action, int mods ) {

	auto window = static_cast<le_window_o*>( glfwGetWindowUserPointer( glfwWindow ) );

	if ( window->mSettings.useEventsQueue ) {

		uint32_t queueIdx = window->eventQueueBack;
		uint32_t eventIdx = 0;

		if ( event_queue_idx_available( window->numEventsForQueue[ queueIdx ], eventIdx ) ) {
			auto& event = window->eventQueue[ queueIdx ][ eventIdx ];
			event.event = LeUiEvent::Type::eKey;
			auto& e     = event.key;
			e.key       = LeUiEvent::NamedKey( key );
			e.action    = LeUiEvent::ButtonAction( action );
			e.scancode  = scancode;
			e.mods      = mods;
		} else {
			// we're over the high - watermark for events, we should probably print a warning.
		}
	}
}

// ----------------------------------------------------------------------
static void glfw_window_character_callback( GLFWwindow* glfwWindow, unsigned int codepoint ) {

	auto window = static_cast<le_window_o*>( glfwGetWindowUserPointer( glfwWindow ) );

	if ( window->mSettings.useEventsQueue ) {

		uint32_t queueIdx = window->eventQueueBack;
		uint32_t eventIdx = 0;

		if ( event_queue_idx_available( window->numEventsForQueue[ queueIdx ], eventIdx ) ) {
			auto& event = window->eventQueue[ queueIdx ][ eventIdx ];
			event.event = LeUiEvent::Type::eCharacter;
			auto& e     = event.character;
			e.codepoint = codepoint;
		} else {
			// we're over the high - watermark for events, we should probably print a warning.
		}
	}
}

// ----------------------------------------------------------------------
static void glfw_window_cursor_position_callback( GLFWwindow* glfwWindow, double xpos, double ypos ) {

	auto window = static_cast<le_window_o*>( glfwGetWindowUserPointer( glfwWindow ) );

	if ( window->mSettings.useEventsQueue ) {

		uint32_t queueIdx = window->eventQueueBack;
		uint32_t eventIdx = 0;

		if ( event_queue_idx_available( window->numEventsForQueue[ queueIdx ], eventIdx ) ) {
			auto& event = window->eventQueue[ queueIdx ][ eventIdx ];
			event.event = LeUiEvent::Type::eCursorPosition;
			auto& e     = event.cursorPosition;
			e.x         = xpos;
			e.y         = ypos;
		} else {
			// we're over the high - watermark for events, we should probably print a warning.
		}
	}
}

// ----------------------------------------------------------------------
static void glfw_window_cursor_enter_callback( GLFWwindow* glfwWindow, int entered ) {

	auto window = static_cast<le_window_o*>( glfwGetWindowUserPointer( glfwWindow ) );

	if ( window->mSettings.useEventsQueue ) {

		uint32_t queueIdx = window->eventQueueBack;
		uint32_t eventIdx = 0;

		if ( event_queue_idx_available( window->numEventsForQueue[ queueIdx ], eventIdx ) ) {
			auto& event = window->eventQueue[ queueIdx ][ eventIdx ];
			event.event = LeUiEvent::Type::eCursorEnter;
			auto& e     = event.cursorEnter;
			e.entered   = uint32_t( entered );
		} else {
			// we're over the high - watermark for events, we should probably print a warning.
		}
	}
}

// ----------------------------------------------------------------------
static void glfw_window_mouse_button_callback( GLFWwindow* glfwWindow, int button, int action, int mods ) {

	auto window = static_cast<le_window_o*>( glfwGetWindowUserPointer( glfwWindow ) );

	if ( window->mSettings.useEventsQueue ) {

		uint32_t queueIdx = window->eventQueueBack;
		uint32_t eventIdx = 0;

		if ( event_queue_idx_available( window->numEventsForQueue[ queueIdx ], eventIdx ) ) {
			auto& event = window->eventQueue[ queueIdx ][ eventIdx ];
			event.event = LeUiEvent::Type::eMouseButton;
			auto& e     = event.mouseButton;
			e.button    = button;
			e.action    = le::UiEvent::ButtonAction( action );
			e.mods      = mods;
		} else {
			// we're over the high - watermark for events, we should probably print a warning.
		}
	}
}

// ----------------------------------------------------------------------
static void glfw_window_scroll_callback( GLFWwindow* glfwWindow, double xoffset, double yoffset ) {

	auto window = static_cast<le_window_o*>( glfwGetWindowUserPointer( glfwWindow ) );

	if ( window->mSettings.useEventsQueue ) {

		uint32_t queueIdx = window->eventQueueBack;
		uint32_t eventIdx = 0;

		if ( event_queue_idx_available( window->numEventsForQueue[ queueIdx ], eventIdx ) ) {
			auto& event = window->eventQueue[ queueIdx ][ eventIdx ];
			event.event = LeUiEvent::Type::eScroll;
			auto& e     = event.scroll;
			e.x_offset  = xoffset;
			e.y_offset  = yoffset;
		} else {
			// we're over the high - watermark for events, we should probably print a warning.
		}
	}
}

// ----------------------------------------------------------------------
static void glfw_window_drop_callback( GLFWwindow* glfwWindow, int count_paths, char const** utf8_paths ) {

	auto        window = static_cast<le_window_o*>( glfwGetWindowUserPointer( glfwWindow ) );
	static auto logger = LeLog( "le_window" );

	if ( window->mSettings.useEventsQueue ) {

		uint32_t queueIdx = window->eventQueueBack;
		uint32_t eventIdx = 0;

		if ( event_queue_idx_available( window->numEventsForQueue[ queueIdx ], eventIdx ) ) {
			auto& event = window->eventQueue[ queueIdx ][ eventIdx ];
			event.event = LeUiEvent::Type::eDrop;

			auto& drop = event.drop;

			drop.paths_count = count_paths;

			// Allocate strings for the current event, store them in a container tied
			// to and with the same lifetime as the back event queue:
			// eventStringData[queueIdx].
			//
			// Then collect pointers for all strings tied to the current event.
			//
			// This is so that we can hand out an array of `char const *`.
			// the vector of pointers itself needs to live somewhere: it
			// will get moved to a container tied to and with the same
			// lifetime as the back event queue: `eventStringPtr[queueIdx]`
			//
			std::vector<char const*> str_ptrs;

			for ( int i = 0; i != count_paths; i++ ) {

				std::string str = utf8_paths[ i ];

				logger.debug( "dropped path [%d]: '%s'", i, str.c_str() );

				window->eventStringData[ queueIdx ].push_front( str );
				str_ptrs.push_back( window->eventStringData[ queueIdx ].front().c_str() );
			}

			window->eventStringPtr[ queueIdx ].push_front( str_ptrs );
			drop.paths_utf8 = window->eventStringPtr[ queueIdx ].front().data();

		} else {
			logger.warn( "surpassed high watermark" );
			// we're over the high - watermark for events, we should probably print a warning.
		}
	}
}
// ----------------------------------------------------------------------

static void glfw_framebuffer_resize_callback( GLFWwindow* glfwWindow, int width_px, int height_px ) {

	auto        window = static_cast<le_window_o*>( glfwGetWindowUserPointer( glfwWindow ) );
	static auto logger = LeLog( "le_window" );

	int w = width_px;
	int h = height_px;

	glfwGetFramebufferSize( glfwWindow, &w, &h );

	window->mSurfaceExtent.width  = uint32_t( w );
	window->mSurfaceExtent.height = uint32_t( h );

};

// ----------------------------------------------------------------------

static size_t window_get_reference_count( le_window_o* self ) {
	return self->referenceCount;
}

// ----------------------------------------------------------------------

static void window_increase_reference_count( le_window_o* self ) {
	++self->referenceCount;
}

// ----------------------------------------------------------------------
static void window_destroy( le_window_o* self ); // ffdecl.

static void window_decrease_reference_count( le_window_o* self ) {
	--self->referenceCount;
	if ( self->referenceCount == 0 ) {
		window_destroy( self );
	}
}

// ----------------------------------------------------------------------

static bool pt2_inside_rect( int x, int y, int left, int top, int width, int height ) {
	return ( x > left &&
	         x < ( left + width ) &&
	         y > top &&
	         y < ( top + height ) );
}

// ----------------------------------------------------------------------

// Goes fullscreen (or returns to windowed mode) on the monitor which contains
// the current window.
//
// If more than one monitor is available, the monitor which contains the current
// window's centre receives the fullscreen window.
//
static void window_toggle_fullscreen( le_window_o* self ) {

	auto& g = self->windowGeometry;

	if ( self->isFullscreen ) {
		// restore previous window state

		glfwSetWindowMonitor( self->window, nullptr, g.x, g.y, g.width, g.height, 0 );

		self->isFullscreen = false;
	} else {
		// note that monitor vidmode width and window width and height are both given in screen coordinates
		// screen coordinates have y go from top to bottom (y-axis points down, origin is at the top left)

		// First store current window geometry to restore state later.
		glfwGetWindowPos( self->window, &g.x, &g.y );
		glfwGetWindowSize( self->window, &g.width, &g.height );

		int                 monitorCount      = 0;
		GLFWmonitor**       monitors          = glfwGetMonitors( &monitorCount );
		GLFWmonitor** const monitors_end      = monitors + monitorCount;
		GLFWmonitor*        fullscreenMonitor = *monitors;

		// Iterate over all monitors, and find out if our window is inside any of them.
		// If so, that monitor will be our fullscreen monitor.

		for ( auto monitor = monitors; monitor != monitors_end; monitor++ ) {

			int x_m{};
			int y_m{};
			glfwGetMonitorPos( *monitor, &x_m, &y_m );

			// Get monitor extents (in screen corrds) by querying its GLFWvidmode
			GLFWvidmode const* mode = glfwGetVideoMode( *monitor );

			// Check if the current window's centre point is inside the monitor in question.
			// If yes, we have found our target monitor for going fullscreen.

			if ( pt2_inside_rect( g.x + g.width / 2, g.y + g.height / 2, x_m, y_m, mode->width, mode->height ) ) {
				fullscreenMonitor = *monitor;
				break;
			}
		}

		auto videoMode = glfwGetVideoMode( fullscreenMonitor );

		glfwSetWindowMonitor( self->window, fullscreenMonitor, 0, 0, videoMode->width, videoMode->height, videoMode->refreshRate );
		self->isFullscreen = true;
	}
}

static void window_set_window_size( le_window_o* self, uint32_t width, uint32_t height ) {

	glfwSetWindowSize( self->window, width, height );
	glfwGetWindowSize( self->window, &self->windowGeometry.width, &self->windowGeometry.height );
}
// ----------------------------------------------------------------------

static le_window_settings_o* window_settings_create() {
	le_window_settings_o* obj = new ( le_window_settings_o );
	return obj;
}

// ----------------------------------------------------------------------

static void window_settings_set_title( le_window_settings_o* self_, const char* title_ ) {
	self_->title = std::string( title_ );
}

// ----------------------------------------------------------------------

static void window_settings_set_width( le_window_settings_o* self_, int width_ ) {
	self_->width = width_;
}

// ----------------------------------------------------------------------

static void window_settings_set_height( le_window_settings_o* self_, int height_ ) {
	self_->height = height_;
}

// ----------------------------------------------------------------------

static void window_settings_set_gamepads_active( le_window_settings_o* self, uint32_t gamepads_bitfield ) {
	self->active_gamepads = gamepads_bitfield;
}

// ----------------------------------------------------------------------

static void window_settings_destroy( le_window_settings_o* self_ ) {
	delete self_;
}

// ----------------------------------------------------------------------
// Creates a khr surface using glfw - note that ownership is handed over to
// the caller which must outlive this le_window_o, and take responsibility
// of deleting  the surface.
static VkSurfaceKHR_T* window_create_surface( le_window_o* self, VkInstance vkInstance ) {
	auto        result = glfwCreateWindowSurface( vkInstance, self->window, nullptr, &self->mSurface );
	static auto logger = LeLog( "le_window" );
	if ( result == VK_SUCCESS ) {
		int tmp_w = 0;
		int tmp_h = 0;
		glfwGetFramebufferSize( self->window, &tmp_w, &tmp_h );
		self->mSurfaceExtent.height = uint32_t( tmp_h );
		self->mSurfaceExtent.width  = uint32_t( tmp_w );
		logger.debug( "Created surface" );
	} else {
		logger.debug( "Error creating surface" );
		return nullptr;
	}
	return self->mSurface;
}

// ----------------------------------------------------------------------

static uint32_t window_get_surface_width( le_window_o* self ) {
	if ( self->mSurface ) {
		return self->mSurfaceExtent.width;
	}
	return 0;
}

// ----------------------------------------------------------------------

static uint32_t window_get_surface_height( le_window_o* self ) {
	if ( self->mSurface ) {
		return self->mSurfaceExtent.height;
	}
	return 0;
}

// ----------------------------------------------------------------------

static void window_set_callbacks( le_window_o* self ) {
	glfwSetKeyCallback( self->window, ( GLFWkeyfun )le_core_forward_callback( le_window_api_i->window_callbacks_i.glfw_key_callback_addr ) );
	glfwSetCharCallback( self->window, ( GLFWcharfun )le_core_forward_callback( le_window_api_i->window_callbacks_i.glfw_char_callback_addr ) );
	glfwSetCursorPosCallback( self->window, ( GLFWcursorposfun )le_core_forward_callback( le_window_api_i->window_callbacks_i.glfw_cursor_pos_callback_addr ) );
	glfwSetCursorEnterCallback( self->window, ( GLFWcursorenterfun )le_core_forward_callback( le_window_api_i->window_callbacks_i.glfw_cursor_enter_callback_addr ) );
	glfwSetMouseButtonCallback( self->window, ( GLFWmousebuttonfun )le_core_forward_callback( le_window_api_i->window_callbacks_i.glfw_mouse_button_callback_addr ) );
	glfwSetScrollCallback( self->window, ( GLFWscrollfun )le_core_forward_callback( le_window_api_i->window_callbacks_i.glfw_scroll_callback_addr ) );
	glfwSetFramebufferSizeCallback( self->window, ( GLFWframebuffersizefun )le_core_forward_callback( le_window_api_i->window_callbacks_i.glfw_framebuffer_size_callback_addr ) );
	glfwSetDropCallback( self->window, ( GLFWdropfun )le_core_forward_callback( le_window_api_i->window_callbacks_i.glfw_drop_callback_addr ) );
}

// ----------------------------------------------------------------------

static void window_remove_callbacks( le_window_o* self ) {
	glfwSetKeyCallback( self->window, nullptr );
	glfwSetCharCallback( self->window, nullptr );
	glfwSetCursorPosCallback( self->window, nullptr );
	glfwSetCursorEnterCallback( self->window, nullptr );
	glfwSetMouseButtonCallback( self->window, nullptr );
	glfwSetScrollCallback( self->window, nullptr );
	glfwSetFramebufferSizeCallback( self->window, nullptr );
}

// ----------------------------------------------------------------------
// Returns an array of events pending since the last call to this method.
// Note that calling this method invalidates any values returned from the previous call to this method.
//
// You must only call this method once per Frame.
static void window_get_ui_event_queue( le_window_o* self, LeUiEvent const** events, uint32_t* numEvents ) {
	static auto logger = LeLog( "le_window" );

	if ( false == self->mSettings.useEventsQueue ) {
		*events   = nullptr;
		numEvents = 0;
		logger.warn( "Querying ui event queue when event queue not in use. Use window.settings to enable events queue." );
		return;
	}

	// ----------| Invariant: Event queue is in use.

	// Flip front and back event queue
	// - store old value for queue back in queue front
	// - bitwise xor value for queue back with 1 - this performs a flip-flop from 0->1->0
	uint32_t eventQueueFront = self->eventQueueBack.fetch_xor( 1 );

	// Note: In the unlikely event that any LeUiEvent will be added asynchronously in between
	// these two calls it will be added to the very end of the back queue and then implicitly
	// released, as the event counter for the back queue is reset at the next step.
	// This is not elegant, but otherwise we must mutex for all event callbacks...

	// Clear new back event queue so that ist may accept new event data.
	{
		self->numEventsForQueue[ self->eventQueueBack ] = 0;

		// Free any strings that were associated with events from this queue.
		self->eventStringData[ self->eventQueueBack ].clear();
		self->eventStringPtr[ self->eventQueueBack ].clear();
	}

	// Hand out front events queue
	*events    = self->eventQueue[ eventQueueFront ].data();
	*numEvents = self->numEventsForQueue[ eventQueueFront ];
}

// ----------------------------------------------------------------------
// Remove the the window from the list of gamepad events subscribers
// in case it is subscribed to gamepad events.
static void window_unsubscribe_from_gamepad_events( le_window_o* window ) {

	static auto gamepad_subscribers = gamepad_events_subscribers_singleton_get();

	if ( gamepad_subscribers ) {
		std::unique_lock lock( gamepad_subscribers->mtx );
		size_t           id = 0;

		for ( auto const& w : gamepad_subscribers->windows ) {
			if ( w == window ) {
				break;
			}
			id++;
		}

		if ( id != gamepad_subscribers->windows.size() ) {
			gamepad_subscribers->windows.erase( gamepad_subscribers->windows.begin() + id );
		}
	}
}

// ----------------------------------------------------------------------
// add window to list of gamepad event subscribers if it is not yet on that
// list
static void window_subscribe_to_gamepad_events( le_window_o* window ) {

	static auto gamepad_subscribers = gamepad_events_subscribers_singleton_get();

	if ( gamepad_subscribers ) {
		std::unique_lock lock( gamepad_subscribers->mtx );

		// First, make sure that the given window is not already a subscriber
		size_t id = 0;
		for ( auto const& w : gamepad_subscribers->windows ) {
			if ( w == window ) {
				break;
			}
			id++;
		}

		if ( id == gamepad_subscribers->windows.size() ) {
			gamepad_subscribers->windows.push_back( window );
		}
	}
}

// ----------------------------------------------------------------------

static le_window_o* window_create() {
	auto obj = new le_window_o();
	return obj;
}

// ----------------------------------------------------------------------

static void window_setup( le_window_o* self, const le_window_settings_o* settings ) {

	if ( settings ) {
		self->mSettings = *settings;
	}

	// TODO: implement GLFW window hints, based on settings.
	// See: http://www.glfw.org/docs/latest/window_guide.html#window_hints
	glfwWindowHint( GLFW_FLOATING, GLFW_TRUE ); // < window is created so that it is always on top.
	glfwWindowHint( GLFW_VISIBLE, GLFW_FALSE ); // < window is initially not visible
	glfwWindowHint( GLFW_CLIENT_API, GLFW_NO_API );

	self->window = glfwCreateWindow( self->mSettings.width, self->mSettings.height, self->mSettings.title.c_str(), self->mSettings.monitor, nullptr );

#ifndef NDEBUG

	int           monitorCount = 0;
	GLFWmonitor** monitors     = glfwGetMonitors( &monitorCount );

	int windowX = 100;
	int windowY = 100;

	if ( monitorCount > 1 ) {
		// if there is more than one monitor, we want our window to appear on the secondary monitor by default.
		glfwGetMonitorPos( monitors[ 1 ], &windowX, &windowY );

		int left, top, right, bottom;
		glfwGetWindowFrameSize( self->window, &left, &top, &right, &bottom );

		windowX += left;
		windowY += top;
	}

	glfwSetWindowPos( self->window, windowX, windowY );

#endif

	glfwShowWindow( self->window );

	// Set the user pointer so callbacks know which window they belong to
	glfwSetWindowUserPointer( self->window, self );

	window_set_callbacks( self );

	// If window settings subscribes to any gamepad,
	// we must add window to gamepad subscribers
	if ( self->mSettings.active_gamepads != 0 ) {
		window_subscribe_to_gamepad_events( self );
	}
}

// ----------------------------------------------------------------------

static void window_destroy( le_window_o* self ) {

	if ( self->window ) {
		window_remove_callbacks( self );
	}

	if ( self->window ) {
		glfwDestroyWindow( self->window );
	}

	window_unsubscribe_from_gamepad_events( self );

	delete self;
}

// ----------------------------------------------------------------------

static bool window_should_close( le_window_o* self ) {
	return glfwWindowShouldClose( self->window );
}

// ----------------------------------------------------------------------

static GLFWwindow* window_get_glfw_window( le_window_o* self ) {
	return self->window;
}

// Returns a void *. It is the platform specific:
// + Windows HWND,
// + Xcb xcb_window_t,
// + Xlib Window / Drawable,
// + Wayland wl_surface*,
// + or Android ANativeWindow*.
static void* window_get_os_native_window_handle( le_window_o* self ) {

#if defined( __linux__ )
	//	void *window = ( void * )glfwGetWaylandWindow( self->window );
	void* window = ( void* )glfwGetX11Window( self->window );
	return window;
#elif defined( _WIN32 )
	return ( void* )glfwGetWin32Window( self->window );
#endif
}

// Callback for when joystick is connected or disconnected, triggered by
// GLFW
static void glfw_joystick_connection_callback( int jid, int event ) {

	static auto logger = le::Log( "glfw" );

	if ( event == GLFW_CONNECTED ) {
		logger.info( "Joystick connected: %d, Name: '%s', GUID: '%s'", jid, glfwGetJoystickName( jid ), glfwGetJoystickGUID( jid ) );

		// test whether the joystick has a gamepad mapping

		if ( glfwJoystickIsGamepad( jid ) ) {
			logger.info( "Joystick has gamepad mapping." );
		} else {
			logger.warn( "Joystick does not have gamepad mapping." );
		}

	} else if ( event == GLFW_DISCONNECTED ) {
		logger.info( "Joystick disconnected: %d", jid );
	}
}

// ----------------------------------------------------------------------

static int init() {

	int result = glfwInit();
	assert( result == GLFW_TRUE );

	static auto logger = LeLog( "le_window" );

	if ( glfwVulkanSupported() ) {
		logger.debug( "Vulkan supported." );
		uint32_t     ext_count = 0;
		const char** exts      = glfwGetRequiredInstanceExtensions( &ext_count );
		for ( uint32_t i = 0; i < ext_count; i++ ) {
			le_backend_vk::settings_i.add_required_instance_extension( exts[ i ] );
		}

	} else {
		logger.error( "Vulkan not supported." );
	}

	// initialise gamepad subscriber singleton - we call this method here
	// for its side-effect, which is to allocate the subscribers singleton
	// if it doesn't exist already.
	gamepad_events_subscribers_singleton_get();

	glfwSetJoystickCallback( ( GLFWjoystickfun )le_core_forward_callback( le_window_api_i->window_callbacks_i.glfw_joystick_connection_callback_addr ) );

	// We add a manual mapping as there doesn't seem to be a mapping
	// for our particular controller in the database.
	// you may have to add further mappings for your device in case it is not present

	char const* xbox_series_x_controller_mapping = "050000005e040000130b000015050000,Xbox Wireless Controller,a:b0,b:b1,back:b10,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b12,leftshoulder:b6,leftstick:b13,lefttrigger:a5,leftx:a0,lefty:a1,misc1:b15,rightshoulder:b7,rightstick:b14,righttrigger:a4,rightx:a2,righty:a3,start:b11,x:b3,y:b4,platform:Linux";

	glfwUpdateGamepadMappings( xbox_series_x_controller_mapping );

	if ( glfwJoystickPresent( GLFW_JOYSTICK_1 ) ) {
		glfw_joystick_connection_callback( GLFW_JOYSTICK_1, GLFW_CONNECTED );
	}

	return result;
}

// ----------------------------------------------------------------------
// Polls Window events via GLFW - this will trigger glfw callbacks for
// any events raised via glfw. Since this triggers polling for any window
// associated with this glfw instance, we must find a way to communicate
// to all windows that their event queue is stale at this moment.
static void pollEvents() {
	glfwPollEvents();

	static auto logger = le::Log( "le_window" );

	static le::UiEvent::GamepadEvent gamepad_data[ 15 ];
	uint32_t                         has_gamepad_data = {};

	GLFWgamepadstate js_state;

	// First iterate over all joysticks and find if there are any joysticks
	// which report gamepad data.
	//
	// If they do, tag at the corresponding position
	// we do this in two passes so that we can minimize the time that
	// we are holding the lock guarding access to the window callback
	// vector in gamepad_subscribers.
	for ( auto i = GLFW_JOYSTICK_1; i != GLFW_JOYSTICK_LAST; i++ ) {

		if ( glfwGetGamepadState( i, &js_state ) ) {

			gamepad_data[ i ].gamepad_id = i;
			memcpy( gamepad_data[ i ].axes, js_state.axes, sizeof( js_state.axes ) );

			gamepad_data[ i ].buttons = 0;

			for ( uint8_t b = 0; b != 15; b++ ) {
				gamepad_data[ i ].buttons |= ( uint16_t( js_state.buttons[ b ] ) << b );
			}
			has_gamepad_data |= ( 1 << i );
		}
	}

	{
		// Trigger callbacks on any windows who subscripe to a particular gamepad.
		static gamepad_events_subscriber_windows_t* gamepad_subscribers = gamepad_events_subscribers_singleton_get();

		// critical section
		std::unique_lock lock( gamepad_subscribers->mtx );

		for ( auto& w : gamepad_subscribers->windows ) {

			uint32_t overlap = w->mSettings.active_gamepads & has_gamepad_data;

			uint32_t gamepad_index = 0;
			while ( overlap ) {

				// find next 1, beginning from the least significant bit
				while ( 0 == ( overlap & ( uint32_t( 1 ) << gamepad_index ) ) ) {
					gamepad_index++;
				}

				// we must propagate to the window the gamepad state at
				// gamepad index

				le_window_gamepad_callback( w, gamepad_data[ gamepad_index ] );

				// flip that particular entry to mark it as processed for this window
				overlap &= ~( 1 << gamepad_index );
			}
		}
	}
}

// ----------------------------------------------------------------------

static void le_terminate() {
	static auto logger = LeLog( "le_window" );
	glfwTerminate();
	{
		// destroy list of subscribers
		void** dict = le_core_produce_dictionary_entry( GAMEPAD_SUBSCRIBERS_SINGLETON_ID );
		// this must produce an entry, and we can cast what is located at this entry to a
		// pointer-to gamepad_events_subscriber_windows_t, which we may delete.
		// if that pointer is not set, delete has no effect, as we can delete a nullptr guilt-free.
		delete ( static_cast<gamepad_events_subscriber_windows_t*>( *dict ) );
		*dict = nullptr;
		logger.debug( "Destroyed GamePad events subscribers singleton" );
	}
	logger.debug( "Glfw was terminated." );
}

// ----------------------------------------------------------------------

static const char* get_clipboard_string() {
	return glfwGetClipboardString( nullptr );
}

// ----------------------------------------------------------------------

static void set_clipboard_string( char const* str ) {
	glfwSetClipboardString( nullptr, str );
}

// ----------------------------------------------------------------------

LE_MODULE_REGISTER_IMPL( le_window, api ) {
	auto window_api_i = static_cast<le_window_api*>( api );

	window_api_i->init                 = init;
	window_api_i->terminate            = le_terminate;
	window_api_i->pollEvents           = pollEvents;
	window_api_i->get_clipboard_string = get_clipboard_string;
	window_api_i->set_clipboard_string = set_clipboard_string;

	auto& window_i                       = window_api_i->window_i;
	window_i.create                      = window_create;
	window_i.destroy                     = window_destroy;
	window_i.setup                       = window_setup;
	window_i.should_close                = window_should_close;
	window_i.get_surface_width           = window_get_surface_width;
	window_i.get_surface_height          = window_get_surface_height;
	window_i.create_surface              = window_create_surface;
	window_i.increase_reference_count    = window_increase_reference_count;
	window_i.decrease_reference_count    = window_decrease_reference_count;
	window_i.get_reference_count         = window_get_reference_count;
	window_i.get_glfw_window             = window_get_glfw_window;
	window_i.get_os_native_window_handle = window_get_os_native_window_handle;

	window_i.toggle_fullscreen  = window_toggle_fullscreen;
	window_i.set_window_size    = window_set_window_size;
	window_i.get_ui_event_queue = window_get_ui_event_queue;

	auto& window_settings_i               = window_api_i->window_settings_i;
	window_settings_i.create              = window_settings_create;
	window_settings_i.destroy             = window_settings_destroy;
	window_settings_i.set_title           = window_settings_set_title;
	window_settings_i.set_width           = window_settings_set_width;
	window_settings_i.set_height          = window_settings_set_height;
	window_settings_i.set_gamepads_active = window_settings_set_gamepads_active;

	auto& callbacks_i                               = window_api_i->window_callbacks_i;
	callbacks_i.glfw_key_callback_addr              = ( void* )glfw_window_key_callback;
	callbacks_i.glfw_char_callback_addr             = ( void* )glfw_window_character_callback;
	callbacks_i.glfw_cursor_pos_callback_addr       = ( void* )glfw_window_cursor_position_callback;
	callbacks_i.glfw_cursor_enter_callback_addr     = ( void* )glfw_window_cursor_enter_callback;
	callbacks_i.glfw_mouse_button_callback_addr     = ( void* )glfw_window_mouse_button_callback;
	callbacks_i.glfw_scroll_callback_addr           = ( void* )glfw_window_scroll_callback;
	callbacks_i.glfw_framebuffer_size_callback_addr = ( void* )glfw_framebuffer_resize_callback;
	callbacks_i.glfw_drop_callback_addr             = ( void* )glfw_window_drop_callback;
	callbacks_i.glfw_joystick_connection_callback_addr = ( void* )glfw_joystick_connection_callback;

#if defined PLUGINS_DYNAMIC
	le_core_load_library_persistently( "./modules/libglfw.so" );
#endif
}
