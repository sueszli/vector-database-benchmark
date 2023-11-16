#ifndef VKHR_PATHS_HH
#define VKHR_PATHS_HH

#ifndef VKHR_ASSETS_PATH
#define VKHR_ASSETS_PATH "share/"
#endif

// e.g. shared path could be: /usr/share/vkhr/
// need to supply SHARED_PATH at compile time!

#define ASSET(PATH)  VKHR_ASSETS_PATH PATH

#define IMAGE(PATH)  ASSET("images/"  PATH)
#define MODEL(PATH)  ASSET("models/"  PATH)
#define SCENE(PATH)  ASSET("scenes/"  PATH)
#define SHADER(PATH) ASSET("shaders/" PATH)
#define STYLE(PATH)  ASSET("styles/"  PATH)

#endif
