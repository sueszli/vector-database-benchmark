"""
Example of embedding CEF browser using PySDL2 library.

Requires PySDL2 and SDL2 libraries, see install instructions further
down.

This example is incomplete and has some issues, see the "Known issues"
section further down. Pull requests with fixes are welcome.

Usage:

    python pysdl2.py [-v] [-h] [-r {software|hardware}]

    -v  turn on debug messages
    -r  specify hardware (default) or software rendering
    -h  display help info

Tested configurations:
- Windows 7: SDL 2.0.7 and PySDL2 0.9.6
- Mac 10.9: SDL 2.0.7 and PySDL2 0.9.6
- Fedora 26: SDL2 2.0.7 with PySDL2 0.9.6
- Ubuntu 14.04: SDL2 with PySDL2 0.9.6

Install instructions:
1. Install SDL libraries for your OS, e.g:
   - Windows: Download SDL2.dll from http://www.libsdl.org/download-2.0.php
              and put SDL2.dll in C:\\Python27\\ (where you've installed Python)
   - Mac: Install Homebrew from https://brew.sh/
          and then type "brew install sdl2"
   - Fedora: sudo dnf install SDL2 SDL2_ttf SDL2_image SDL2_gfx SDL2_mixer
   - Ubuntu: sudo apt-get install libsdl2-dev
2. Install PySDL2 using pip package manager:
   pip install PySDL2

Known issues (pull requests are welcome):
- There are issues when running on slow machine - key events are being
  lost (noticed on Mac only), see Issue #324 for more details
- Performance is still not perfect, see Issue #324 for further details
- Keyboard modifiers that are not yet handled in this example:
  ctrl, marking text inputs with the shift key.
- Dragging with mouse not implemented
- Window size is fixed, cannot be resized

GUI controls:
  Due to SDL2's lack of GUI widgets there are no GUI controls
  for the user. However, as an exercise this example could
  be extended by create some simple SDL2 widgets. An example of
  widgets made using PySDL2 can be found as part of the Pi
  Entertainment System at:
  https://github.com/neilmunday/pes/blob/master/lib/pes/ui.py
"""
import argparse
import logging
import sys

def die(msg):
    if False:
        print('Hello World!')
    '\n    Helper function to exit application on failed imports etc.\n    '
    sys.stderr.write('%s\n' % msg)
    sys.exit(1)
try:
    from cefpython3 import cefpython as cef
except ImportError:
    die('ERROR: cefpython3 package not found\n       To install type: pip install cefpython3')
try:
    import sdl2
    import sdl2.ext
except ImportError as exc:
    excstr = repr(exc)
    if 'No module named sdl2' in excstr:
        die('ERROR: PySDL2 package not found\n       To install type: pip install PySDL2')
    elif 'could not find any library for SDL2 (PYSDL2_DLL_PATH: unset)' in excstr:
        die('ERROR: SDL2 package not found.\n       See install instructions in top comment in sources.')
    else:
        die(excstr)
try:
    from PIL import Image
except ImportError:
    die('ERROR: PIL package not found\n       To install type: pip install Pillow')
if sys.platform == 'darwin':
    try:
        import AppKit
    except ImportError:
        die('ERROR: pyobjc package not found\n       To install type: pip install pyobjc')

def main():
    if False:
        while True:
            i = 10
    '\n    Parses input, initializes everything and then runs the main loop of the\n    program, which handles input and draws the scene.\n    '
    parser = argparse.ArgumentParser(description='PySDL2 / cefpython example', add_help=True)
    parser.add_argument('-v', '--verbose', help='Turn on debug info', dest='verbose', action='store_true')
    parser.add_argument('-r', '--renderer', help='Specify hardware or software rendering', default='hardware', dest='renderer', choices=['software', 'hardware'])
    args = parser.parse_args()
    logLevel = logging.INFO
    if args.verbose:
        logLevel = logging.DEBUG
    logging.basicConfig(format='[%(filename)s %(levelname)s]: %(message)s', level=logLevel)
    logging.info('Using PySDL2 %s' % sdl2.__version__)
    version = sdl2.SDL_version()
    sdl2.SDL_GetVersion(version)
    logging.info('Using SDL2 %s.%s.%s' % (version.major, version.minor, version.patch))
    width = 800
    height = 600
    headerHeight = 0
    browserHeight = height - headerHeight
    browserWidth = width
    scrollEnhance = 40
    frameRate = 100
    sys.excepthook = cef.ExceptHook
    switches = {'disable-surfaces': '', 'disable-gpu': '', 'disable-gpu-compositing': '', 'enable-begin-frame-scheduling': ''}
    browser_settings = {'windowless_frame_rate': frameRate}
    cef.Initialize(settings={'windowless_rendering_enabled': True}, switches=switches)
    if sys.platform == 'darwin':
        AppKit.NSApplication.sharedApplication().setActivationPolicy_(AppKit.NSApplicationActivationPolicyRegular)
    logging.debug('cef initialised')
    window_info = cef.WindowInfo()
    window_info.SetAsOffscreen(0)
    sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO)
    logging.debug('SDL2 initialised')
    window = sdl2.video.SDL_CreateWindow(b'cefpython3 SDL2 Demo', sdl2.video.SDL_WINDOWPOS_UNDEFINED, sdl2.video.SDL_WINDOWPOS_UNDEFINED, width, height, 0)
    backgroundColour = sdl2.SDL_Color(0, 0, 0)
    renderer = None
    if args.renderer == 'hardware':
        logging.info('Using hardware rendering')
        renderer = sdl2.SDL_CreateRenderer(window, -1, sdl2.render.SDL_RENDERER_ACCELERATED)
    else:
        logging.info('Using software rendering')
        renderer = sdl2.SDL_CreateRenderer(window, -1, sdl2.render.SDL_RENDERER_SOFTWARE)
    renderHandler = RenderHandler(renderer, width, height - headerHeight)
    browser = cef.CreateBrowserSync(window_info, url='https://www.google.com/', settings=browser_settings)
    browser.SetClientHandler(LoadHandler())
    browser.SetClientHandler(renderHandler)
    browser.SendFocusEvent(True)
    browser.WasResized()
    running = True
    frames = 0
    logging.debug('beginning rendering loop')
    resetFpsTime = True
    fpsTime = 0
    while running:
        startTime = sdl2.timer.SDL_GetTicks()
        if resetFpsTime:
            fpsTime = sdl2.timer.SDL_GetTicks()
            resetFpsTime = False
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT or (event.type == sdl2.SDL_KEYDOWN and event.key.keysym.sym == sdl2.SDLK_ESCAPE):
                running = False
                logging.debug('SDL2 QUIT event')
                break
            if event.type == sdl2.SDL_MOUSEBUTTONDOWN:
                if event.button.button == sdl2.SDL_BUTTON_LEFT:
                    if event.button.y > headerHeight:
                        logging.debug('SDL2 MOUSEBUTTONDOWN event (left button)')
                        browser.SendMouseClickEvent(event.button.x, event.button.y - headerHeight, cef.MOUSEBUTTON_LEFT, False, 1)
            elif event.type == sdl2.SDL_MOUSEBUTTONUP:
                if event.button.button == sdl2.SDL_BUTTON_LEFT:
                    if event.button.y > headerHeight:
                        logging.debug('SDL2 MOUSEBUTTONUP event (left button)')
                        browser.SendMouseClickEvent(event.button.x, event.button.y - headerHeight, cef.MOUSEBUTTON_LEFT, True, 1)
            elif event.type == sdl2.SDL_MOUSEMOTION:
                if event.motion.y > headerHeight:
                    browser.SendMouseMoveEvent(event.motion.x, event.motion.y - headerHeight, False)
            elif event.type == sdl2.SDL_MOUSEWHEEL:
                logging.debug('SDL2 MOUSEWHEEL event')
                x = event.wheel.x
                if x < 0:
                    x -= scrollEnhance
                else:
                    x += scrollEnhance
                y = event.wheel.y
                if y < 0:
                    y -= scrollEnhance
                else:
                    y += scrollEnhance
                browser.SendMouseWheelEvent(0, 0, x, y)
            elif event.type == sdl2.SDL_TEXTINPUT:
                logging.debug('SDL2 TEXTINPUT event: %s' % event.text.text)
                keycode = ord(event.text.text)
                key_event = {'type': cef.KEYEVENT_CHAR, 'windows_key_code': keycode, 'character': keycode, 'unmodified_character': keycode, 'modifiers': cef.EVENTFLAG_NONE}
                browser.SendKeyEvent(key_event)
                key_event = {'type': cef.KEYEVENT_KEYUP, 'windows_key_code': keycode, 'character': keycode, 'unmodified_character': keycode, 'modifiers': cef.EVENTFLAG_NONE}
                browser.SendKeyEvent(key_event)
            elif event.type == sdl2.SDL_KEYDOWN:
                logging.debug('SDL2 KEYDOWN event')
                if event.key.keysym.sym == sdl2.SDLK_RETURN:
                    keycode = event.key.keysym.sym
                    key_event = {'type': cef.KEYEVENT_CHAR, 'windows_key_code': keycode, 'character': keycode, 'unmodified_character': keycode, 'modifiers': cef.EVENTFLAG_NONE}
                    browser.SendKeyEvent(key_event)
                elif event.key.keysym.sym in [sdl2.SDLK_BACKSPACE, sdl2.SDLK_DELETE, sdl2.SDLK_LEFT, sdl2.SDLK_RIGHT, sdl2.SDLK_UP, sdl2.SDLK_DOWN, sdl2.SDLK_HOME, sdl2.SDLK_END]:
                    keycode = get_key_code(event.key.keysym.sym)
                    if keycode is not None:
                        key_event = {'type': cef.KEYEVENT_RAWKEYDOWN, 'windows_key_code': keycode, 'native_key_code': get_native_key(keycode), 'character': 0, 'unmodified_character': 0, 'modifiers': cef.EVENTFLAG_NONE}
                        browser.SendKeyEvent(key_event)
            elif event.type == sdl2.SDL_KEYUP:
                logging.debug('SDL2 KEYUP event')
                if event.key.keysym.sym in [sdl2.SDLK_RETURN, sdl2.SDLK_BACKSPACE, sdl2.SDLK_DELETE, sdl2.SDLK_LEFT, sdl2.SDLK_RIGHT, sdl2.SDLK_UP, sdl2.SDLK_DOWN, sdl2.SDLK_HOME, sdl2.SDLK_END]:
                    keycode = get_key_code(event.key.keysym.sym)
                    if keycode is not None:
                        key_event = {'type': cef.KEYEVENT_KEYUP, 'windows_key_code': keycode, 'native_key_code': get_native_key(keycode), 'character': keycode, 'unmodified_character': keycode, 'modifiers': cef.EVENTFLAG_NONE}
                        browser.SendKeyEvent(key_event)
        sdl2.SDL_SetRenderDrawColor(renderer, backgroundColour.r, backgroundColour.g, backgroundColour.b, 255)
        sdl2.SDL_RenderClear(renderer)
        cef.MessageLoopWork()
        sdl2.SDL_RenderCopy(renderer, renderHandler.texture, None, sdl2.SDL_Rect(0, headerHeight, browserWidth, browserHeight))
        sdl2.SDL_RenderPresent(renderer)
        frames += 1
        if sdl2.timer.SDL_GetTicks() - fpsTime > 1000:
            logging.debug('FPS: %d' % frames)
            frames = 0
            resetFpsTime = True
        if sdl2.timer.SDL_GetTicks() - startTime < 1000.0 / frameRate:
            sdl2.timer.SDL_Delay(1000 // frameRate - (sdl2.timer.SDL_GetTicks() - startTime))
    exit_app()

def get_key_code(key):
    if False:
        while True:
            i = 10
    'Helper function to convert SDL2 key codes to cef ones'
    key_map = {sdl2.SDLK_RETURN: 13, sdl2.SDLK_DELETE: 46, sdl2.SDLK_BACKSPACE: 8, sdl2.SDLK_LEFT: 37, sdl2.SDLK_RIGHT: 39, sdl2.SDLK_UP: 38, sdl2.SDLK_DOWN: 40, sdl2.SDLK_HOME: 36, sdl2.SDLK_END: 35}
    if key in key_map:
        return key_map[key]
    logging.error('\n        Keyboard mapping incomplete: unsupported SDL key %d.\n        See https://wiki.libsdl.org/SDLKeycodeLookup for mapping.\n        ' % key)
    return None
MACOS_TRANSLATION_TABLE = {8: 51, 37: 123, 38: 126, 39: 124, 40: 125}

def get_native_key(key):
    if False:
        print('Hello World!')
    '\n    Helper function for returning the correct native key map for the operating\n    system.\n    '
    if sys.platform == 'darwin':
        return MACOS_TRANSLATION_TABLE.get(key, key)
    return key

class LoadHandler(object):
    """Simple handler for loading URLs."""

    def OnLoadingStateChange(self, is_loading, **_):
        if False:
            for i in range(10):
                print('nop')
        if not is_loading:
            logging.info('Page loading complete')

    def OnLoadError(self, frame, failed_url, **_):
        if False:
            while True:
                i = 10
        if not frame.IsMain():
            return
        logging.error('Failed to load %s' % failed_url)

class RenderHandler(object):
    """
    Handler for rendering web pages to the
    screen via SDL2.

    The object's texture property is exposed
    to allow the main rendering loop to access
    the SDL2 texture.
    """

    def __init__(self, renderer, width, height):
        if False:
            i = 10
            return i + 15
        self.__width = width
        self.__height = height
        self.__renderer = renderer
        self.texture = None

    def GetViewRect(self, rect_out, **_):
        if False:
            return 10
        rect_out.extend([0, 0, self.__width, self.__height])
        return True

    def OnPaint(self, element_type, paint_buffer, **_):
        if False:
            i = 10
            return i + 15
        "\n        Using the pixel data from CEF's offscreen rendering\n        the data is converted by PIL into a SDL2 surface\n        which can then be rendered as a SDL2 texture.\n        "
        if element_type == cef.PET_VIEW:
            image = Image.frombuffer('RGBA', (self.__width, self.__height), paint_buffer.GetString(mode='rgba', origin='top-left'), 'raw', 'BGRA')
            mode = image.mode
            rmask = gmask = bmask = amask = 0
            depth = None
            pitch = None
            if mode == 'RGB':
                if sdl2.endian.SDL_BYTEORDER == sdl2.endian.SDL_LIL_ENDIAN:
                    rmask = 255
                    gmask = 65280
                    bmask = 16711680
                else:
                    rmask = 16711680
                    gmask = 65280
                    bmask = 255
                depth = 24
                pitch = self.__width * 3
            elif mode in ('RGBA', 'RGBX'):
                if sdl2.endian.SDL_BYTEORDER == sdl2.endian.SDL_LIL_ENDIAN:
                    rmask = 0
                    gmask = 65280
                    bmask = 16711680
                    if mode == 'RGBA':
                        amask = 4278190080
                else:
                    rmask = 4278190080
                    gmask = 16711680
                    bmask = 65280
                    if mode == 'RGBA':
                        amask = 255
                depth = 32
                pitch = self.__width * 4
            else:
                logging.error('ERROR: Unsupported mode: %s' % mode)
                exit_app()
            pxbuf = image.tobytes()
            surface = sdl2.SDL_CreateRGBSurfaceFrom(pxbuf, self.__width, self.__height, depth, pitch, rmask, gmask, bmask, amask)
            if self.texture:
                sdl2.SDL_DestroyTexture(self.texture)
            self.texture = sdl2.SDL_CreateTextureFromSurface(self.__renderer, surface)
            sdl2.SDL_FreeSurface(surface)
        else:
            logging.warning('Unsupport element_type in OnPaint')

def exit_app():
    if False:
        i = 10
        return i + 15
    'Tidy up SDL2 and CEF before exiting.'
    sdl2.SDL_Quit()
    cef.Shutdown()
    logging.info('Exited gracefully')
if __name__ == '__main__':
    main()