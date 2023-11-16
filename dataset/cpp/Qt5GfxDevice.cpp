#if defined(WITH_QT5)

#if defined(WITH_GLEW)
#	define GLEW_NO_GLU
#	include <GL/glew.h>
#endif

#include "../../Common.h"
#include "../Graphics/GL/GLTexture.h"
#include "../Graphics/GL/GLFramebuffer.h"
#include "Qt5GfxDevice.h"
#include "../MainApplication.h"
#include "Qt5Widget.h"

#include <QWindow>
#include <QApplication>
#include <QScreen>

namespace nCine
{
	Qt5GfxDevice::Qt5GfxDevice(const WindowMode& windowMode, const GLContextInfo& glContextInfo, const DisplayMode& displayMode, Qt5Widget& widget)
		: IGfxDevice(windowMode, glContextInfo, displayMode), widget_(widget), isResizable_(windowMode.isResizable)
	{
		initDevice(windowMode.isFullscreen);
	}

	void Qt5GfxDevice::setSwapInterval(int interval)
	{
		widget_.format().setSwapInterval(interval);
	}

	void Qt5GfxDevice::setResolution(bool fullscreen, int width, int height)
	{
		width_ = width;
		height_ = height;
		theApplication().resizeRootViewport(width, height);

		QRect rect = widget_.geometry();
		rect.setWidth(width);
		rect.setHeight(height);
		widget_.setGeometry(rect);

		if (fullscreen) {
			widget_.setWindowState(widget_.windowState() | Qt::WindowFullScreen);
		} else {
			widget_.setWindowState(widget_.windowState() & ~Qt::WindowFullScreen);
			if (!isResizable_) {
				widget_.setMinimumSize(width, height);
				widget_.setMaximumSize(width, height);
			}
		}
	}

	void Qt5GfxDevice::setResolutionInternal(int width, int height)
	{
		width_ = width;
		height_ = height;

		theApplication().resizeRootViewport(width, height);

		QRect rect = widget_.geometry();
		rect.setWidth(width);
		rect.setHeight(height);
		widget_.setGeometry(rect);
	}

	void Qt5GfxDevice::setWindowTitle(const char* windowTitle)
	{
		widget_.setWindowTitle(windowTitle);
	}

	void Qt5GfxDevice::setWindowIcon(const char* windowIconFilename)
	{
		widget_.setWindowIcon(QIcon(windowIconFilename));
	}

	int Qt5GfxDevice::windowPositionX() const
	{
		return widget_.pos().x();
	}

	int Qt5GfxDevice::windowPositionY() const
	{
		return widget_.pos().y();
	}

	const Vector2i Qt5GfxDevice::windowPosition() const
	{
		return Vector2i(widget_.pos().x(), widget_.pos().y());
	}

	void Qt5GfxDevice::setWindowPosition(int x, int y)
	{
		QRect geometry = widget_.geometry();
		geometry.setX(x);
		geometry.setY(y);
		widget_.setGeometry(geometry);
	}

	void Qt5GfxDevice::flashWindow() const
	{
		QApplication::alert(&widget_, 0);
	}

	const Qt5GfxDevice::VideoMode& Qt5GfxDevice::currentVideoMode() const
	{
		return videoModes_[0];
	}

	void Qt5GfxDevice::updateVideoModes()
	{
		QScreen* screen = nullptr;
		if (widget_.window() && widget_.window()->windowHandle()) {
			screen = widget_.window()->windowHandle()->screen();
		} else {
			screen = QApplication::primaryScreen();
		}

		videoModes_.resize_for_overwrite(1);
		if (screen) {
			videoModes_[0].width = screen->size().width();
			videoModes_[0].height = screen->size().height();
			videoModes_[0].refreshRate = screen->refreshRate();

			if (screen->depth() >= 24) {
				videoModes_[0].redBits = 8;
				videoModes_[0].greenBits = 8;
				videoModes_[0].blueBits = 8;
			}
		}
	}

#if defined(WITH_GLEW)
	void Qt5GfxDevice::initGlew()
	{
		const GLenum err = glewInit();
		FATAL_ASSERT_MSG(err == GLEW_OK, "GLEW error: %s", glewGetErrorString(err));

		glContextInfo_.debugContext = glContextInfo_.debugContext && glewIsSupported("GL_ARB_debug_output");
	}
#endif

	void Qt5GfxDevice::resetTextureBinding()
	{
		GLTexture::bindHandle(GL_TEXTURE_2D, 0);
	}

	void Qt5GfxDevice::bindDefaultDrawFramebufferObject()
	{
		const GLuint glHandle = widget_.defaultFramebufferObject();
		GLFramebuffer::bindHandle(GL_DRAW_FRAMEBUFFER, glHandle);
	}

	void Qt5GfxDevice::initDevice(bool isFullscreen)
	{
		QSurfaceFormat format;
		format.setRedBufferSize(displayMode_.redBits());
		format.setGreenBufferSize(displayMode_.greenBits());
		format.setBlueBufferSize(displayMode_.blueBits());
		//format.setAlphaBufferSize(displayMode_.alphaBits());
		format.setSwapBehavior(displayMode_.isDoubleBuffered() ? QSurfaceFormat::DoubleBuffer : QSurfaceFormat::SingleBuffer);
		format.setDepthBufferSize(displayMode_.depthBits());
		format.setStencilBufferSize(displayMode_.stencilBits());
		format.setVersion(glContextInfo_.majorVersion, glContextInfo_.minorVersion);
#if defined(WITH_OPENGLES)
		format.setRenderableType(QSurfaceFormat::OpenGLES);
#endif
		format.setProfile(glContextInfo_.coreProfile ? QSurfaceFormat::CoreProfile : QSurfaceFormat::CompatibilityProfile);
		if (glContextInfo_.debugContext)
			format.setOptions(QSurfaceFormat::DebugContext);

		if (isFullscreen) {
			widget_.setWindowState(widget_.windowState() | Qt::WindowFullScreen);
		}

		const int interval = displayMode_.hasVSync() ? 1 : 0;
		format.setSwapInterval(interval);

		widget_.setFormat(format);
		QSurfaceFormat::setDefaultFormat(format);

		updateVideoModes();
	}
}

#endif