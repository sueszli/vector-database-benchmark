/*
 *  SimpleRenderEngine (https://github.com/mortennobel/SimpleRenderEngine)
 *
 *  Created by Morten Nobel-Jørgensen ( http://www.nobel-joergensen.com/ )
 *  License: MIT
 */

#include <chrono>
#include <iostream>
#include <sre/imgui_sre.hpp>
#include <sre/Log.hpp>
#include <sre/VR.hpp>
#include "sre/SDLRenderer.hpp"
#define SDL_MAIN_HANDLED

#ifdef EMSCRIPTEN
#include "emscripten.h"
#endif
#include "sre/impl/GL.hpp"


#ifdef SRE_DEBUG_CONTEXT
void GLAPIENTRY openglCallbackFunction(GLenum source,
	GLenum type,
	GLuint id,
	GLenum severity,
	GLsizei length,
	const GLchar* message,
	const void* userParam) {
	using namespace std;
	const char* typeStr;
	switch (type) {
	case GL_DEBUG_TYPE_ERROR:
		typeStr = "ERROR";
		break;
	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
		typeStr = "DEPRECATED_BEHAVIOR";
		break;
	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
		typeStr = "UNDEFINED_BEHAVIOR";
		break;
	case GL_DEBUG_TYPE_PORTABILITY:
		typeStr = "PORTABILITY";
		break;
	case GL_DEBUG_TYPE_PERFORMANCE:
		typeStr = "PERFORMANCE";
		break;
	case GL_DEBUG_TYPE_OTHER:
	default:
		typeStr = "OTHER";
		break;
		}
	const char* severityStr;
	switch (severity) {
	case GL_DEBUG_SEVERITY_LOW:
		severityStr = "LOW";
		break;
	case GL_DEBUG_SEVERITY_MEDIUM:
		severityStr = "MEDIUM";
		break;
	case GL_DEBUG_SEVERITY_HIGH:
		severityStr = "HIGH";
		break;
	default:
		severityStr = "Unknown";
		break;
	}
    LOG_ERROR("---------------------opengl-callback-start------------\n"
              "message: %s\n"
              "type: %s\n"
              "id: %i\n"
              "severity: %s\n"
              "---------------------opengl-callback-end--------------"
              ,message,typeStr, id ,severityStr
    );

		}
#endif

namespace sre{

    SDLRenderer* SDLRenderer::instance = nullptr;

    struct SDLRendererInternal{
        static void update(float f){
            SDLRenderer::instance->frame(f);
        }
    };

    void update(){
        typedef std::chrono::high_resolution_clock Clock;
        using FpSeconds = std::chrono::duration<float, std::chrono::seconds::period>;
        static auto lastTick = Clock::now();
        auto tick = Clock::now();
        float deltaTime = std::chrono::duration_cast<FpSeconds>(tick - lastTick).count();
        lastTick = tick;
        SDLRendererInternal::update(deltaTime);
    }

    SDLRenderer::SDLRenderer()
    :frameUpdate ([](float){}),
     frameRender ([](){}),
     keyEvent ([](SDL_Event&){}),
     mouseEvent ([](SDL_Event&){}),
     controllerEvent ([](SDL_Event&){}),
     joystickEvent ([](SDL_Event&){}),
     touchEvent ([](SDL_Event&){}),
     otherEvent([](SDL_Event&){}),
     windowTitle( std::string("SimpleRenderEngine ")+std::to_string(Renderer::sre_version_major)+"."+std::to_string(Renderer::sre_version_minor )+"."+std::to_string(Renderer::sre_version_point))
    {

        instance = this;

    }

    SDLRenderer::~SDLRenderer() {
        delete r;
        r = nullptr;

        instance = nullptr;

        SDL_DestroyWindow(window);
        SDL_Quit();
    }

    void SDLRenderer::frame(float deltaTimeSec){
        typedef std::chrono::high_resolution_clock Clock;
        using MilliSeconds = std::chrono::duration<float, std::chrono::milliseconds::period>;
        auto lastTick = Clock::now();

        SDL_Event e;
        //Handle events on queue
        while( SDL_PollEvent( &e ) != 0 )
        {
            ImGui_SRE_ProcessEvent(&e);
            switch (e.type) {
                case SDL_QUIT:
                    running = false;
                    break;
                case SDL_KEYDOWN:
                case SDL_KEYUP:
                    keyEvent(e);
                    break;
                case SDL_MOUSEMOTION:
                case SDL_MOUSEBUTTONDOWN:
                case SDL_MOUSEBUTTONUP:
                case SDL_MOUSEWHEEL:
                    mouseEvent(e);
                    break;
                case SDL_CONTROLLERAXISMOTION:
                case SDL_CONTROLLERBUTTONDOWN:
                case SDL_CONTROLLERBUTTONUP:
                case SDL_CONTROLLERDEVICEADDED:
                case SDL_CONTROLLERDEVICEREMOVED:
                case SDL_CONTROLLERDEVICEREMAPPED:
                    controllerEvent(e);
                    break;
                case SDL_JOYAXISMOTION:
                case SDL_JOYBALLMOTION:
                case SDL_JOYHATMOTION:
                case SDL_JOYBUTTONDOWN:
                case SDL_JOYBUTTONUP:
                case SDL_JOYDEVICEADDED:
                case SDL_JOYDEVICEREMOVED:
                    joystickEvent(e);
                    break;
                case SDL_FINGERDOWN:
                case SDL_FINGERUP:
                case SDL_FINGERMOTION:
                    touchEvent(e);
                    break;
                default:
                    otherEvent(e);
                    break;
            }
        }
        {   // time meassure
            auto tick = Clock::now();
            deltaTimeEvent = std::chrono::duration_cast<MilliSeconds>(tick - lastTick).count();
            lastTick = tick;
        }
        frameUpdate(deltaTimeSec);
        {   // time meassure
            auto tick = Clock::now();
            deltaTimeUpdate = std::chrono::duration_cast<MilliSeconds>(tick - lastTick).count();
            lastTick = tick;
        }
        frameRender();
        {   // time meassure
            auto tick = Clock::now();
            deltaTimeRender = std::chrono::duration_cast<MilliSeconds>(tick - lastTick).count();
            lastTick = tick;
        }

        r->swapWindow();
    }

    void SDLRenderer::startEventLoop() {
        if (!window){
            LOG_INFO("SDLRenderer::init() not called");
        }

        running = true;
#ifdef EMSCRIPTEN
        emscripten_set_main_loop(update, 0, 1);
#else
        typedef std::chrono::high_resolution_clock Clock;
        using FpSeconds = std::chrono::duration<float, std::chrono::seconds::period>;
        auto lastTick = Clock::now();
        float deltaTime = 0;

        while (running){
			frame(deltaTime);

            auto tick = Clock::now();
            deltaTime = std::chrono::duration_cast<FpSeconds>(tick - lastTick).count();

            // warn potential busy wait (SDL_Delay may truncate small numbers)
            // https://forum.lazarus.freepascal.org/index.php?topic=35689.0
            while (deltaTime < timePerFrame){
                Uint32 delayMs = static_cast<Uint32>((timePerFrame - deltaTime) / 1000);
                SDL_Delay(delayMs);
                tick = Clock::now();
                deltaTime = std::chrono::duration_cast<FpSeconds>(tick - lastTick).count();
            }
            lastTick = tick;
        }
#endif
    }

    void SDLRenderer::startEventLoop(std::shared_ptr<VR> vr) {
        if (!window){
            LOG_INFO("SDLRenderer::init() not called");
        }

        running = true;

        typedef std::chrono::high_resolution_clock Clock;
        using FpSeconds = std::chrono::duration<float, std::chrono::seconds::period>;
        auto lastTick = Clock::now();
        float deltaTime = 0;

        while (running){
            vr->render();
			frame(deltaTime);

            auto tick = Clock::now();
            deltaTime = std::chrono::duration_cast<FpSeconds>(tick - lastTick).count();
            lastTick = tick;
        }
    }

    void SDLRenderer::stopEventLoop() {
        running = false;
    }

    void SDLRenderer::setWindowSize(glm::ivec2 size) {
        int width = size.x;
        int height = size.y;
        windowWidth = width;
        windowHeight = height;
        if (window!= nullptr){
            SDL_SetWindowSize(window, width, height);
        }
    }

    void SDLRenderer::setWindowTitle(std::string title) {
        windowTitle = title;
        if (window != nullptr) {
            SDL_SetWindowTitle(window, title.c_str());
        }
    }

    SDL_Window *SDLRenderer::getSDLWindow() {
        return window;
    }

    void SDLRenderer::setFullscreen(bool enabled) {
#ifndef EMSCRIPTEN
        if (isFullscreen() != enabled){
            Uint32 flags = (SDL_GetWindowFlags(window) ^ SDL_WINDOW_FULLSCREEN_DESKTOP);
            if (SDL_SetWindowFullscreen(window, flags) < 0) // NOTE: this takes FLAGS as the second param, NOT true/false!
            {
                std::cout << "Toggling fullscreen mode failed: " << SDL_GetError() << std::endl;
                return;
            }
        }
#endif
    }

    bool SDLRenderer::isFullscreen() {
        return ((SDL_GetWindowFlags(window)&(SDL_WINDOW_FULLSCREEN|SDL_WINDOW_FULLSCREEN_DESKTOP)) != 0);
    }

    void SDLRenderer::setMouseCursorVisible(bool enabled) {
        SDL_ShowCursor(enabled?SDL_ENABLE:SDL_DISABLE);
    }

    bool SDLRenderer::isMouseCursorVisible() {
        return SDL_ShowCursor(SDL_QUERY)==SDL_ENABLE;
    }

    bool SDLRenderer::setMouseCursorLocked(bool enabled) {
        if (enabled){
            setMouseCursorVisible(false);
        }
        return SDL_SetRelativeMouseMode(enabled?SDL_TRUE:SDL_FALSE) == 0;
    }

    bool SDLRenderer::isMouseCursorLocked() {
        return SDL_GetRelativeMouseMode() == SDL_TRUE;
    }

    SDLRenderer::InitBuilder SDLRenderer::init() {
        return SDLRenderer::InitBuilder(this);
    }

    glm::vec3 SDLRenderer::getLastFrameStats() {
        return {
                deltaTimeEvent,deltaTimeUpdate,deltaTimeRender
        };
    }

    SDLRenderer::InitBuilder::~InitBuilder() {
        build();
    }

    SDLRenderer::InitBuilder::InitBuilder(SDLRenderer *sdlRenderer)
            :sdlRenderer(sdlRenderer) {
    }

    SDLRenderer::InitBuilder &SDLRenderer::InitBuilder::withSdlInitFlags(uint32_t sdlInitFlag) {
        this->sdlInitFlag = sdlInitFlag;
        return *this;
    }

    SDLRenderer::InitBuilder &SDLRenderer::InitBuilder::withSdlWindowFlags(uint32_t sdlWindowFlags) {
        this->sdlWindowFlags = sdlWindowFlags;
        return *this;
    }

    SDLRenderer::InitBuilder &SDLRenderer::InitBuilder::withVSync(bool vsync) {
        this->vsync = vsync;
        return *this;
    }

    SDLRenderer::InitBuilder &SDLRenderer::InitBuilder::withGLVersion(int majorVersion, int minorVersion) {
        this->glMajorVersion = majorVersion;
        this->glMinorVersion = minorVersion;
        return *this;
    }

    void SDLRenderer::InitBuilder::build() {
        if (sdlRenderer->running){
            return;
        }
        if (!sdlRenderer->window){
#ifdef EMSCRIPTEN
            SDL_Renderer *renderer = nullptr;
            SDL_CreateWindowAndRenderer(sdlRenderer->windowWidth, sdlRenderer->windowHeight, SDL_WINDOW_OPENGL, &sdlRenderer->window, &renderer);
#else
            SDL_Init( sdlInitFlag  );
            SDL_GL_SetAttribute(SDL_GL_FRAMEBUFFER_SRGB_CAPABLE, 1);
            SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
            SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, glMajorVersion);
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, glMinorVersion);
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
#ifdef SRE_DEBUG_CONTEXT
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);
#endif
            sdlRenderer->window = SDL_CreateWindow(sdlRenderer->windowTitle.c_str(), SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, sdlRenderer->windowWidth, sdlRenderer->windowHeight,sdlWindowFlags);
#endif
            sdlRenderer->r = new Renderer(sdlRenderer->window, vsync, maxSceneLights);

#ifdef SRE_DEBUG_CONTEXT
            if (glDebugMessageCallback) {
				LOG_INFO("Register OpenGL debug callback ");

				std::cout << "Register OpenGL debug callback " << std::endl;
				glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
				glDebugMessageCallback(openglCallbackFunction, nullptr);
				GLuint unusedIds = 0;
				glDebugMessageControl(GL_DONT_CARE,
					GL_DONT_CARE,
					GL_DONT_CARE,
					0,
					&unusedIds,
					true);

			}
#endif
        }
    }

    SDLRenderer::InitBuilder &SDLRenderer::InitBuilder::withMaxSceneLights(int maxSceneLights) {
        this->maxSceneLights = maxSceneLights;
        return *this;
    }

    void SDLRenderer::setWindowIcon(std::shared_ptr<Texture> tex){
        auto texRaw = tex->getRawImage();
        auto surface = SDL_CreateRGBSurfaceFrom(texRaw.data(),tex->getWidth(),tex->getHeight(),32,tex->getWidth()*4,0x00ff0000,0x0000ff00,0x000000ff,0xff000000);

        // The icon is attached to the window pointer
        SDL_SetWindowIcon(window, surface);

        // ...and the surface containing the icon pixel data is no longer required.
        SDL_FreeSurface(surface);

    }
}
