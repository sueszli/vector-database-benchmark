from PyQt5.QtCore import Qt, QMetaObject, pyqtSlot
from PyQt5.QtWidgets import QOpenGLWidget, QApplication
from PyQt5.QtOpenGL import QGLContext
from OpenGL import GL
from mpv import MPV, _mpv_get_sub_api, _mpv_opengl_cb_set_update_callback, _mpv_opengl_cb_init_gl, OpenGlCbGetProcAddrFn, _mpv_opengl_cb_draw, _mpv_opengl_cb_report_flip, MpvSubApi, OpenGlCbUpdateFn, _mpv_opengl_cb_uninit_gl, MpvRenderContext

def get_proc_addr(_, name):
    if False:
        i = 10
        return i + 15
    glctx = QGLContext.currentContext()
    if glctx is None:
        return 0
    addr = int(glctx.getProcAddress(name.decode('utf-8')))
    return addr

class MpvWidget(QOpenGLWidget):

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(parent=parent)
        self.mpv = MPV(ytdl=True)
        self.ctx = None
        self.get_proc_addr_c = OpenGlCbGetProcAddrFn(get_proc_addr)

    def initializeGL(self):
        if False:
            i = 10
            return i + 15
        params = {'get_proc_address': self.get_proc_addr_c}
        self.ctx = MpvRenderContext(self.mpv, 'opengl', opengl_init_params=params)
        self.ctx.update_cb = self.on_update

    def paintGL(self):
        if False:
            while True:
                i = 10
        ratio = self.windowHandle().devicePixelRatio()
        w = int(self.width() * ratio)
        h = int(self.height() * ratio)
        opengl_fbo = {'w': w, 'h': h, 'fbo': self.defaultFramebufferObject()}
        self.ctx.render(flip_y=True, opengl_fbo=opengl_fbo)

    @pyqtSlot()
    def maybe_update(self):
        if False:
            while True:
                i = 10
        if self.window().isMinimized():
            self.makeCurrent()
            self.paintGL()
            self.context().swapBuffers(self.context().surface())
            self.doneCurrent()
        else:
            self.update()

    def on_update(self, ctx=None):
        if False:
            return 10
        QMetaObject.invokeMethod(self, 'maybe_update')

    def on_update_fake(self, ctx=None):
        if False:
            return 10
        pass

    def closeEvent(self, _):
        if False:
            while True:
                i = 10
        pass
if __name__ == '__main__':
    import locale
    app = QApplication([])
    locale.setlocale(locale.LC_NUMERIC, 'C')
    widget = MpvWidget()
    widget.show()
    url = 'data/test.webm'
    widget.mpv.play(url)
    app.exec()