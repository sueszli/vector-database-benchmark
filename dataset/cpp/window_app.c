#ifndef APP_H
#define APP_H

#include "../render/render_opengl.c" // OPENGL

#define SDL_MAIN_HANDLED
#include "../3rd/SDL2/SDL.h" // eSDL2
// #include "../3rd/glfast.h" // SDL2+OPENGL(subset)

extern SDL_Window *window;

enum {
    WINDOW_LEGACY_OPENGL = 0x4,
    WINDOW_SQUARED = 0x8,
    WINDOW_NO_MOUSE = 0x10,
    WINDOW_MSAA4 = 0x20,
};

API int   window_create(float zoom /* 10.0f */, int flags);
API int   window_update();
API void  window_swap(void **pixels); // split into window_capture() and window_swap();
API void  window_destroy(void);

API void  window_capture(void **pixels);

API void  window_title( const char *title );
API void  window_fullscreen(bool enabled);

API vec2  window_size();
API int   window_width();
API int   window_height();
API float window_aspect();
API char* window_stats();

API int   window_is_minimized();
API int   window_is_maximized();
API int   window_is_fullscreen();
API int   window_is_fullscreen_desktop();
API int   window_is_visible();
API int   window_is_resizable();
API int   window_is_borderless();
API int   window_has_input_focus();
API int   window_has_input_grabbed();
API int   window_has_mouse_focus();

API void  window_load_opengl();

#endif


#ifdef APP_C
#pragma once
#include "../input/input.c"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>


int (*printf_handler)(const char *fmt, ...) = printf;

static void die_callback( const char *text ) {
    fprintf(stderr, "%s\n", text);
    exit(-1);
}
static void error_callback(int error, const char* description) {
    int whitelisted = !!strstr(description, "Failed to create OpenGL context");
    if(whitelisted) return;
    fprintf(stderr, "app error %#x: %s\n", error, description);
}
/*
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}
static void* glfwGetProcAddressExtraCompat(const char *name) {
    void *ptr = glfwGetProcAddress(name);
    if( !ptr ) {
        char buf[128];
        sprintf(buf, "%sARB", name);
        ptr = glfwGetProcAddress(buf);
        if( ptr ) printf("%s -> %s", name, buf );
    }
    if( !ptr ) {
        char buf[128];
        sprintf(buf, "%sEXT", name);
        ptr = glfwGetProcAddress(buf);
        if( ptr ) printf("%s -> %s", name, buf );
    }
    if( !ptr ) {
        printf("%s not found\n", name);
    }
    return ptr;
}
*/

/*
#define GLFAST_EXTERNAL_LOADER
#define GLFAST_IMPLEMENTATION
#include "../3rd/glfast.h" // OPENGL(subset)
*/

#define GL_DEBUG_OUTPUT                   0x92E0
#define GL_DEBUG_OUTPUT_SYNCHRONOUS       0x8242

#define GL_DEBUG_SEVERITY_HIGH            0x9146
#define GL_DEBUG_SEVERITY_LOW             0x9148
#define GL_DEBUG_SEVERITY_MEDIUM          0x9147
#define GL_DEBUG_SEVERITY_NOTIFICATION    0x826B
#define GL_DEBUG_SOURCE_API               0x8246
#define GL_DEBUG_SOURCE_APPLICATION       0x824A
#define GL_DEBUG_SOURCE_OTHER             0x824B
#define GL_DEBUG_SOURCE_SHADER_COMPILER   0x8248
#define GL_DEBUG_SOURCE_THIRD_PARTY       0x8249
#define GL_DEBUG_SOURCE_WINDOW_SYSTEM     0x8247
#define GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR 0x824D
#define GL_DEBUG_TYPE_ERROR               0x824C
#define GL_DEBUG_TYPE_MARKER              0x8268
#define GL_DEBUG_TYPE_OTHER               0x8251
#define GL_DEBUG_TYPE_PERFORMANCE         0x8250
#define GL_DEBUG_TYPE_POP_GROUP           0x826A
#define GL_DEBUG_TYPE_PORTABILITY         0x824F
#define GL_DEBUG_TYPE_PUSH_GROUP          0x8269
#define GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR  0x824E


void glDebug(uint32_t source, uint32_t type, uint32_t id, uint32_t severity, int32_t length, const char * message, void * userdata) {

    // whitelisted codes
    if( id == 131154 ) return; // Pixel-path performance warning: Pixel transfer is synchronized with 3D rendering.
    // if( id == 131169 ) return;
    if( id == 131185 ) return; // Buffer object 2 (bound to GL_ELEMENT_ARRAY_BUFFER_ARB, usage hint is GL_STATIC_DRAW) will use VIDEO memory as the source for buffer object operations
    // if( id == 131204 ) return;
    if( id == 131218 ) return; // Program/shader state performance warning: Vertex shader in program 9 is being recompiled based on GL state.
    if( id == 2 ) return; // INFO: API_ID_RECOMPILE_FRAGMENT_SHADER performance warning has been generated. Fragment shader recompiled due to state change. [ID: 2]

    const char * GL_ERROR_SOURCE[] = { "API", "WINDOW SYSTEM", "SHADER COMPILER", "THIRD PARTY", "APPLICATION", "OTHER" };
    const char * GL_ERROR_SEVERITY[] = { "HIGH", "MEDIUM", "LOW", "NOTIFICATION" };
    const char * GL_ERROR_TYPE[] = { "ERROR", "DEPRECATED BEHAVIOR", "UNDEFINED DEHAVIOUR", "PORTABILITY", "PERFORMANCE", "OTHER" };

    severity = severity == GL_DEBUG_SEVERITY_NOTIFICATION ? 3 : severity - GL_DEBUG_SEVERITY_HIGH;
    source = source - GL_DEBUG_SOURCE_API;
    type = type - GL_DEBUG_TYPE_ERROR;

    SDL_Log( "%s [ID: %u]\n", message, id );
    /* "[SEVERITY: %s] [SOURCE: %s] [TYPE: %s]", GL_ERROR_SEVERITY[severity],
        GL_ERROR_SOURCE[source],
        GL_ERROR_TYPE[type], */

#ifdef _WIN32
    if (type <= 2 && debugging()) {
        breakpoint();
    }
#endif
}

void glDebugEnable() {
    // Enable the debug callback
    // #ifndef RELEASE
    typedef void (*GLDEBUGPROC)(uint32_t source, uint32_t type, uint32_t id, uint32_t severity, int32_t length, const char * message, const void * userParam);
    typedef void (*GLDEBUGMESSAGECALLBACKPROC)(GLDEBUGPROC callback, const void * userParam);
    void (*glDebugMessageCallback)(GLDEBUGPROC callback, const void * userParam) = (GLDEBUGMESSAGECALLBACKPROC)SDL_GL_GetProcAddress("glDebugMessageCallback");
    // glEnable(GL_DEBUG_OUTPUT);
    // glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    // glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, /*NULL*/0, GL_TRUE);
    glDebugMessageCallback((GLDEBUGPROC)glDebug, NULL);
    // #endif
}


#ifdef __GNUC__ // also, clang
    int __argc;
    char **__argv;
    __attribute__((constructor)) void init_argcv(int argc, char **argv) {
        __argc = argc;
        __argv = argv;
    }
#else
    // MINGW: _argc, _argv
    // HP-UX:  __argc_value, __argv_value 
    // libc:  __libc_argc, __libc_argv
#endif

/*static*/ SDL_Window  * window = NULL;
/*static*/ SDL_GLContext glcontext;
/*static*/ char title[128] = {0};
static int should_quit = 0;
static int window_flags = 0;

void window_title( const char *title_ ) {
    if( title_ ) strcpy(title, title_);
}

void window_load_opengl(void) {
#  if defined __gl3w_h_
    gl3w_init();
#elif defined __glad_h_
    // gladLoadGLLoader((GLADloadproc) glfwGetProcAddressExtraCompat);

    // Initialize OpenGL loader
    // bool err = false;
    // bool err = gl3wInit() != 0;
    // bool err = glewInit() != GLEW_OK;
    bool err = gladLoadGL() == 0;
    if (err) {
        die_callback("Error: Cannot initialize OpenGL loader");
    }
#else
    puts("warning: no opengl loader found");
#endif

    // intel gma hd
    // if(!glGenVertexArrays) glGenVertexArrays = (void*)glfwGetProcAddressExtraCompat("glGenVertexArrays");
    // if(!glBindVertexArray) glBindVertexArray = (void*)glfwGetProcAddressExtraCompat("glBindVertexArray");

    SDL_GL_SetSwapInterval(1); // Enable vsync, also check -1

    glDebugEnable();

    // int end; glGetIntegerv(GL_NUM_EXTENSIONS, &end);
    // const char *sep = ""; for( int i = 0; i < end; ++i) printf("%s%s", sep, glGetStringi(GL_EXTENSIONS,i)), sep = ","; puts("");
    printf("; version: %.*s, glsl: %.*s, vendor: %s (%s)\n",
        3, glGetString(GL_VERSION),
        3, glGetString(GL_SHADING_LANGUAGE_VERSION),
        glGetString(GL_RENDERER), glGetString(GL_VENDOR));
}

int window_create( float zoom, int flags ) {
    if( window ) {
        return 0;
    } else {
        // engine init
        init();

/* #if !EDITOR
#ifdef _MSC_VER
        _chdir( SDL_GetBasePath() );
#else
        chdir( SDL_GetBasePath() );
#endif
*/

        atexit(window_destroy);
    }

    // remove extension
    int arg0len = strlen( __argv[0] );
    if( arg0len > 4 ) {
        char *dot = &__argv[0][arg0len - 4];
        if( 0 == strcmp( dot, ".exe" ) || 0 == strcmp( dot, ".EXE" ) ) {
            *dot = '\0';
        }
    }
    // truncate paths
    char *s1 = strrchr(__argv[0], '/'), *s2 = strrchr(__argv[0], '\\');
    __argv[0] = s1 > s2 ? s1+1 : s2 > s1 ? s2+1 : __argv[0];
    //
    const char *wtitle = __argv[0];

    flags |= WINDOW_LEGACY_OPENGL; // O:)

    window_flags = flags;

    // detect monitors
    SDL_Rect bounds;
    int monitor = 0;
    for( ; monitor < SDL_GetNumVideoDisplays(); ++monitor ) {
        SDL_GetDisplayBounds( monitor, &bounds );
        if(!bounds.x && !bounds.y) break; // primary monitor found
    }

    // window screen coverage (zoom)
    // zoom = 1..100%, or [0.f .. 1.f] ; fullscreen if zoom = 100% or 1.f ; borderless-fullscreen if zoom > 100% (and also if 1.1f or negative zoom)
    // @todo: remove [0..1] range?
    int fullscreen = 0;
    zoom = zoom > 1.1 ? zoom / 100.f : zoom;
    zoom = zoom > 0 && zoom < 1 ? zoom : (fullscreen = 1, 1);

    int appw = (int)(bounds.w * zoom), apph = (int)(bounds.h * zoom);
    if( flags & WINDOW_SQUARED ) {
        if( appw > apph ) appw = apph;
        else apph = appw;
    }
    int w = appw, h = apph;

    /* flags
    SDL_WINDOW_FULLSCREEN - fullscreen window
    SDL_WINDOW_FULLSCREEN_DESKTOP - fullscreen window at the current desktop resolution
    SDL_WINDOW_OPENGL - window usable with OpenGL context
    SDL_WINDOW_VULKAN - window usable with a Vulkan instance
    SDL_WINDOW_HIDDEN - window is not visible
    SDL_WINDOW_BORDERLESS - no window decoration
    SDL_WINDOW_RESIZABLE - window can be resized
    SDL_WINDOW_MINIMIZED - window is minimized
    SDL_WINDOW_MAXIMIZED - window is maximized
    SDL_WINDOW_INPUT_GRABBED - window has grabbed input focus
    SDL_WINDOW_ALLOW_HIGHDPI - window should be created in high-DPI mode if supported (>= SDL 2.0.1)
    */
    int sdl_window_flags = SDL_WINDOW_OPENGL;
    sdl_window_flags |= SDL_WINDOW_INPUT_FOCUS | SDL_WINDOW_MOUSE_FOCUS;
    sdl_window_flags |= fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0; // SDL_WINDOW_FULLSCREEN
    sdl_window_flags |= fullscreen ? 0 : SDL_WINDOW_RESIZABLE;
    sdl_window_flags |= fullscreen ? SDL_WINDOW_INPUT_GRABBED : 0; // grabbed -> mouse confined to window

    if( flags & WINDOW_MSAA4 ) {
        int msaa_samples = 4;
        SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
        SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, msaa_samples);
    }

    if(1) {
        SDL_GL_SetAttribute(SDL_GL_FRAMEBUFFER_SRGB_CAPABLE, 1);
    }

    /*
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 5);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 5);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 5);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 0);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    glfwWindowHint(GLFW_STENCIL_BITS, 32); // 0 == GLFW_DONT_CARE
    // glfwWindowHint(GLFW_REFRESH_RATE, desktop->refreshRate);
    */

    SDL_GL_SetAttribute( SDL_GL_DEPTH_SIZE, 24 ); // ALSO 32, OR 0 == DONT care
    SDL_GL_SetAttribute( SDL_GL_STENCIL_SIZE, 8 );

    // #ifndef RELEASE
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);
    // #endif

    int have_debug = GL_FALSE;
    int have_core = flags & WINDOW_LEGACY_OPENGL ? SDL_GL_CONTEXT_PROFILE_COMPATIBILITY : SDL_GL_CONTEXT_PROFILE_CORE;

    // try core first, unless compat stated
    if( !(flags & WINDOW_LEGACY_OPENGL) ) {
        int majors[] = { 4, 4, 4, 4, 4, 3, 3 };
        int minors[] = { 4, 3, 2, 1, 0, 3, 2 };
        for( int i = 0; !window && i < sizeof(majors) / sizeof(majors[0]); ++i ) {
            int majv = majors[i], minv = minors[i];
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, majv);
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, minv);
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG); // Always required on Mac
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, have_core ); // SDL_GL_CONTEXT_PROFILE_CORE);

        #if 1 // !SHIPPING
            // only in +4.3
            // SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG); have_debug = GL_TRUE;
        #endif

            window = SDL_CreateWindow(
                wtitle,
                bounds.x + (bounds.w - appw) / 2, bounds.y + (bounds.h - apph) / 2,
                appw, apph, sdl_window_flags
            );

            // if(window) printf("opengl %d.%d%s%s context created\n", majv, minv, have_core ? "-core" : "", have_debug ? "-debug" : "");
        }
    }

    // else 2.1 compat
    if( !window ) {
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);

        // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_FALSE);
        // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE ); // GLFW_OPENGL_COMPAT_PROFILE );
        /* SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG); */ have_debug = GL_FALSE;

        window = SDL_CreateWindow(
            wtitle,
            bounds.x + (bounds.w - appw) / 2, bounds.y + (bounds.h - apph) / 2,
            appw, apph, sdl_window_flags
        );

        // if(window) printf("opengl 2.1%s%s context created\n", 0 ? "-core" : "", have_debug ? "-debug" : "");
    }

    if (!window) {
        die_callback("Error: Cannot create window (SDL_CreateWindow)");
    }

    // assign icon (if exists)
    window_icon( vfs_find(va("%s.ico", __argv[0])) );

    glcontext = SDL_GL_CreateContext(window);
    if( !glcontext ) {
        die_callback("Error: Cannot create GL context (SDL_GL_CreateContext)"); //, SDL_GetError());
    }

    SDL_GL_MakeCurrent(window, glcontext);

    window_load_opengl();

    //glEnable(GL_FRAMEBUFFER_SRGB);

#if 0
    // center
    if( desktop ) {
        glfwSetWindowPos(window , (desktop->width-appw)/2 , (desktop->height-apph)/2);
    }
#endif

    // renderer_init();
        GLuint vao = 0;
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glEnableVertexAttribArray(0);
        viewport_color(vec3(0.1/2,0.1/2,0.1));

    //glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &max_anisotropy);
    //glGetIntegerv(GL_MAX_SAMPLES, &max_supported_samples);

    extern void set_mouse(int);
    set_mouse( flags & WINDOW_NO_MOUSE ? 'hide' : 'show' );

    ui_create();

    SDL_RaiseWindow(window);

    return 1;
}
vec2 window_size() {
    int rect[2];
    SDL_GL_GetDrawableSize(window, &rect[0], &rect[1]);
    return vec2(rect[0],rect[1]);
}
int window_update() {
    if( should_quit ) {
        return 0;
    }
    vec2 rect = window_size();
    int width = (int)rect.x, height = (int)rect.y;
    float ratio = width / (height + 1.f);

    //renderer_update(width, height);
    viewport_clip(vec2(0, 0), window_size());
    viewport_clear(true, true);

    mouse_update();

    return 1;
}

void window_capture( void **pixels ) {
    if( pixels ) {
        int w = window_width(), h = window_height(), comps = 3;
        *pixels = (unsigned char *)REALLOC(*pixels, w * h * comps);

        // @todo, bench against http://roxlu.com/2014/048/fast-pixel-transfers-with-pixel-buffer-objects
        // should we switch tech?
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, w, h, comps == 3 ? GL_RGB : GL_BGRA, GL_UNSIGNED_BYTE, *pixels);
    }
}

void window_swap( void **pixels ) {
    vec2 rect = window_size();

    // renderer_post(rect.w, rect.h)
    {
        //  text_draw(width, height);
        extern int ddraw_printf_line;
        ddraw_printf_line = 0;
        ddraw_render2d();
        static material m, *init = 0;
        if( !init ) material_create(init = &m);
        material_enable(&m, 0);
        // glDisable(GL_BLEND); // @fixme
    }

    ui_render();

    {
        static material m = {0}, *mi = 0; if( !mi ) {
            material_create(mi = &m);
        }
        material_enable(mi, 0);
    }

    if( pixels && !should_quit ) {
        window_capture(pixels);
    }

#if !defined SHIPPING || !SHIPPING
    static uint64_t num_frame = 0;
    if( !num_frame++ ) {
        void *pixels = 0;
        window_capture(&pixels);
        if( pixels ) {
            stbi_flip_vertically_on_write(true);
            stbi_write_png(va("%s.png", __argv[0]), window_width(), window_height(), 3, pixels, window_width()*3);
            FREE(pixels);
        }
    }
#endif

    if( title[0] ) {
        SDL_SetWindowTitle(window, title);
    }

    SDL_GL_SwapWindow(window);
    glFinish();

    // SDL_PumpEvents();
    SDL_Event event;
    nk_input_begin(ui_ctx);
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT)
            should_quit = 1;
        if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE && event.window.windowID == SDL_GetWindowID(window))
            should_quit = 1;
        if (event.type == SDL_MOUSEMOTION && (window_flags & WINDOW_NO_MOUSE) ) {
            /*
            extern double mx,my;
            mx += event.motion.xrel;
            my += event.motion.yrel;
            int *rects = window_size();
            SDL_WarpMouseInWindow(window, rects[0]/2, rects[1]/2);
            */
        }
        nk_sdl_handle_event(&event);
    } nk_input_end(ui_ctx);

    // input
    memcpy(scancodes_old, scancodes_now, SDL_NUM_SCANCODES * sizeof(uint8_t));
    memcpy(scancodes_now, SDL_GetKeyboardState(NULL), SDL_NUM_SCANCODES * sizeof(uint8_t));
}

void window_destroy(void) {
    if(window) {
        // ui_destroy(); // <-- commented because we want to exit quickier O:)
        SDL_DestroyWindow(window);
        window = 0;
        should_quit = 0;
    }
}

void window_fullscreen(bool enabled) {
    SDL_SetWindowFullscreen(window, enabled ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0);
}

char* window_stats() {
    static double num_frames = 0, begin = FLT_MAX, fps = 60, prev_frame = 0, boot_time = 0;

    double now = SDL_GetTicks() / 1000.0;
    if( !boot_time ) {
        boot_time = now;
        prev_frame = now;
    }
    if( begin > now ) {
        begin = now;
        num_frames = 0;
    }
    if( (now - begin) >= 0.25f ) {
        fps = num_frames * (1.f / (now - begin));
    }
    if( (now - begin) > 1 ) {
        begin = now + ((now - begin) - 1);
        num_frames = 0;
    }

    const char *appname = __argv[0]; // @todo: print %used/%avail kib mem, %used/%avail objs as well
    char *buf = va("%s - boot %.2fs %5.2ffps %5.2fms", appname, boot_time, fps, (now - prev_frame) * 1000.f);
    buf += (buf[0] == ' ');

    prev_frame = now;
    ++num_frames;

    return buf;
}

int window_width() { return window_size().x; }
int window_height() { return window_size().y; }
float window_aspect() { vec2 rect = window_size(); return rect.x/(rect.y+0.001f); }

int window_is_minimized() { return !!(SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED ); }
int window_is_maximized() { return !!(SDL_GetWindowFlags(window) & SDL_WINDOW_MAXIMIZED ); }
int window_is_fullscreen() { return !!(SDL_GetWindowFlags(window) & SDL_WINDOW_FULLSCREEN ); }
int window_is_fullscreen_desktop() { return !!(SDL_GetWindowFlags(window) & SDL_WINDOW_FULLSCREEN_DESKTOP ); }
int window_is_visible() { return !!(SDL_GetWindowFlags(window) & SDL_WINDOW_SHOWN ); }
int window_is_resizable() { return !!(SDL_GetWindowFlags(window) & SDL_WINDOW_RESIZABLE ); }
int window_is_borderless() { return !!(SDL_GetWindowFlags(window) & SDL_WINDOW_BORDERLESS ); }
int window_has_input_focus() { return !!(SDL_GetWindowFlags(window) & SDL_WINDOW_INPUT_FOCUS ); }
int window_has_input_grabbed() { return !!(SDL_GetWindowFlags(window) & SDL_WINDOW_INPUT_GRABBED ); }
int window_has_mouse_focus() { return !!(SDL_GetWindowFlags(window) & SDL_WINDOW_MOUSE_FOCUS ); }

#endif
