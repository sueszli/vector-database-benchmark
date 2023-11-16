import sys
from pywin.mfc import docview
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
except ImportError:
    print('The OpenGL extensions do not appear to be installed.')
    print('This Pythonwin demo can not run')
    sys.exit(1)
import timer
import win32api
import win32con
import win32ui
PFD_TYPE_RGBA = 0
PFD_TYPE_COLORINDEX = 1
PFD_MAIN_PLANE = 0
PFD_OVERLAY_PLANE = 1
PFD_UNDERLAY_PLANE = -1
PFD_DOUBLEBUFFER = 1
PFD_STEREO = 2
PFD_DRAW_TO_WINDOW = 4
PFD_DRAW_TO_BITMAP = 8
PFD_SUPPORT_GDI = 16
PFD_SUPPORT_OPENGL = 32
PFD_GENERIC_FORMAT = 64
PFD_NEED_PALETTE = 128
PFD_NEED_SYSTEM_PALETTE = 256
PFD_SWAP_EXCHANGE = 512
PFD_SWAP_COPY = 1024
PFD_SWAP_LAYER_BUFFERS = 2048
PFD_GENERIC_ACCELERATED = 4096
PFD_DEPTH_DONTCARE = 536870912
PFD_DOUBLEBUFFER_DONTCARE = 1073741824
PFD_STEREO_DONTCARE = 2147483648
threeto8 = [0, 73 >> 1, 146 >> 1, 219 >> 1, 292 >> 1, 365 >> 1, 438 >> 1, 255]
twoto8 = [0, 85, 170, 255]
oneto8 = [0, 255]

def ComponentFromIndex(i, nbits, shift):
    if False:
        while True:
            i = 10
    val = i >> shift & 15
    if nbits == 1:
        val = val & 1
        return oneto8[val]
    elif nbits == 2:
        val = val & 3
        return twoto8[val]
    elif nbits == 3:
        val = val & 7
        return threeto8[val]
    else:
        return 0
OpenGLViewParent = docview.ScrollView

class OpenGLView(OpenGLViewParent):

    def PreCreateWindow(self, cc):
        if False:
            while True:
                i = 10
        self.HookMessage(self.OnSize, win32con.WM_SIZE)
        style = cc[5]
        style = style | win32con.WS_CLIPSIBLINGS | win32con.WS_CLIPCHILDREN
        cc = (cc[0], cc[1], cc[2], cc[3], cc[4], style, cc[6], cc[7], cc[8])
        cc = self._obj_.PreCreateWindow(cc)
        return cc

    def OnSize(self, params):
        if False:
            while True:
                i = 10
        lParam = params[3]
        cx = win32api.LOWORD(lParam)
        cy = win32api.HIWORD(lParam)
        glViewport(0, 0, cx, cy)
        if self.oldrect[2] > cx or self.oldrect[3] > cy:
            self.RedrawWindow()
        self.OnSizeChange(cx, cy)
        self.oldrect = (self.oldrect[0], self.oldrect[1], cx, cy)

    def OnInitialUpdate(self):
        if False:
            return 10
        self.SetScaleToFitSize((100, 100))
        return self._obj_.OnInitialUpdate()

    def OnCreate(self, cs):
        if False:
            i = 10
            return i + 15
        self.oldrect = self.GetClientRect()
        self._InitContexts()
        self.Init()

    def OnDestroy(self, msg):
        if False:
            for i in range(10):
                print('nop')
        self.Term()
        self._DestroyContexts()
        return OpenGLViewParent.OnDestroy(self, msg)

    def OnDraw(self, dc):
        if False:
            for i in range(10):
                print('nop')
        self.DrawScene()

    def OnEraseBkgnd(self, dc):
        if False:
            i = 10
            return i + 15
        return 1

    def _SetupPixelFormat(self):
        if False:
            for i in range(10):
                print('nop')
        dc = self.dc.GetSafeHdc()
        pfd = CreatePIXELFORMATDESCRIPTOR()
        pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER
        pfd.iPixelType = PFD_TYPE_RGBA
        pfd.cColorBits = 24
        pfd.cDepthBits = 32
        pfd.iLayerType = PFD_MAIN_PLANE
        pixelformat = ChoosePixelFormat(dc, pfd)
        SetPixelFormat(dc, pixelformat, pfd)
        self._CreateRGBPalette()

    def _CreateRGBPalette(self):
        if False:
            while True:
                i = 10
        dc = self.dc.GetSafeHdc()
        n = GetPixelFormat(dc)
        pfd = DescribePixelFormat(dc, n)
        if pfd.dwFlags & PFD_NEED_PALETTE:
            n = 1 << pfd.cColorBits
            pal = []
            for i in range(n):
                this = (ComponentFromIndex(i, pfd.cRedBits, pfd.cRedShift), ComponentFromIndex(i, pfd.cGreenBits, pfd.cGreenShift), ComponentFromIndex(i, pfd.cBlueBits, pfd.cBlueShift), 0)
                pal.append(this)
            hpal = win32ui.CreatePalette(pal)
            self.dc.SelectPalette(hpal, 0)
            self.dc.RealizePalette()

    def _InitContexts(self):
        if False:
            while True:
                i = 10
        self.dc = self.GetDC()
        self._SetupPixelFormat()
        hrc = wglCreateContext(self.dc.GetSafeHdc())
        wglMakeCurrent(self.dc.GetSafeHdc(), hrc)

    def _DestroyContexts(self):
        if False:
            while True:
                i = 10
        hrc = wglGetCurrentContext()
        wglMakeCurrent(0, 0)
        if hrc:
            wglDeleteContext(hrc)

    def DrawScene(self):
        if False:
            while True:
                i = 10
        assert 0, 'You must override this method'

    def Init(self):
        if False:
            while True:
                i = 10
        assert 0, 'You must override this method'

    def OnSizeChange(self, cx, cy):
        if False:
            print('Hello World!')
        pass

    def Term(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class TestView(OpenGLView):

    def OnSizeChange(self, right, bottom):
        if False:
            print('Hello World!')
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClearDepth(1.0)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        if bottom:
            aspect = right / bottom
        else:
            aspect = 0
        glLoadIdentity()
        gluPerspective(45.0, aspect, 3.0, 7.0)
        glMatrixMode(GL_MODELVIEW)
        near_plane = 3.0
        far_plane = 7.0
        maxObjectSize = 3.0
        self.radius = near_plane + maxObjectSize / 2.0

    def Init(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def DrawScene(self):
        if False:
            i = 10
            return i + 15
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPushMatrix()
        glTranslatef(0.0, 0.0, -self.radius)
        self._DrawCone()
        self._DrawPyramid()
        glPopMatrix()
        glFinish()
        SwapBuffers(wglGetCurrentDC())

    def _DrawCone(self):
        if False:
            for i in range(10):
                print('nop')
        glColor3f(0.0, 1.0, 0.0)
        glPushMatrix()
        glTranslatef(-1.0, 0.0, 0.0)
        quadObj = gluNewQuadric()
        gluQuadricDrawStyle(quadObj, GLU_FILL)
        gluQuadricNormals(quadObj, GLU_SMOOTH)
        gluCylinder(quadObj, 1.0, 0.0, 1.0, 20, 10)
        glPopMatrix()

    def _DrawPyramid(self):
        if False:
            while True:
                i = 10
        glPushMatrix()
        glTranslatef(1.0, 0.0, 0.0)
        glBegin(GL_TRIANGLE_FAN)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 1.0, 0.0)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(-1.0, 0.0, 0.0)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 1.0)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(1.0, 0.0, 0.0)
        glEnd()
        glPopMatrix()

class CubeView(OpenGLView):

    def OnSizeChange(self, right, bottom):
        if False:
            print('Hello World!')
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClearDepth(1.0)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        if bottom:
            aspect = right / bottom
        else:
            aspect = 0
        glLoadIdentity()
        gluPerspective(45.0, aspect, 3.0, 7.0)
        glMatrixMode(GL_MODELVIEW)
        near_plane = 3.0
        far_plane = 7.0
        maxObjectSize = 3.0
        self.radius = near_plane + maxObjectSize / 2.0

    def Init(self):
        if False:
            while True:
                i = 10
        self.busy = 0
        self.wAngleY = 10.0
        self.wAngleX = 1.0
        self.wAngleZ = 5.0
        self.timerid = timer.set_timer(150, self.OnTimer)

    def OnTimer(self, id, timeVal):
        if False:
            return 10
        self.DrawScene()

    def Term(self):
        if False:
            for i in range(10):
                print('nop')
        timer.kill_timer(self.timerid)

    def DrawScene(self):
        if False:
            i = 10
            return i + 15
        if self.busy:
            return
        self.busy = 1
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPushMatrix()
        glTranslatef(0.0, 0.0, -self.radius)
        glRotatef(self.wAngleX, 1.0, 0.0, 0.0)
        glRotatef(self.wAngleY, 0.0, 1.0, 0.0)
        glRotatef(self.wAngleZ, 0.0, 0.0, 1.0)
        self.wAngleX = self.wAngleX + 1.0
        self.wAngleY = self.wAngleY + 10.0
        self.wAngleZ = self.wAngleZ + 5.0
        glBegin(GL_QUAD_STRIP)
        glColor3f(1.0, 0.0, 1.0)
        glVertex3f(-0.5, 0.5, 0.5)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(-0.5, -0.5, 0.5)
        glColor3f(1.0, 1.0, 1.0)
        glVertex3f(0.5, 0.5, 0.5)
        glColor3f(1.0, 1.0, 0.0)
        glVertex3f(0.5, -0.5, 0.5)
        glColor3f(0.0, 1.0, 1.0)
        glVertex3f(0.5, 0.5, -0.5)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.5, -0.5, -0.5)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(-0.5, 0.5, -0.5)
        glColor3f(0.0, 0.0, 0.0)
        glVertex3f(-0.5, -0.5, -0.5)
        glColor3f(1.0, 0.0, 1.0)
        glVertex3f(-0.5, 0.5, 0.5)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(-0.5, -0.5, 0.5)
        glEnd()
        glBegin(GL_QUADS)
        glColor3f(1.0, 0.0, 1.0)
        glVertex3f(-0.5, 0.5, 0.5)
        glColor3f(1.0, 1.0, 1.0)
        glVertex3f(0.5, 0.5, 0.5)
        glColor3f(0.0, 1.0, 1.0)
        glVertex3f(0.5, 0.5, -0.5)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(-0.5, 0.5, -0.5)
        glEnd()
        glBegin(GL_QUADS)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(-0.5, -0.5, 0.5)
        glColor3f(1.0, 1.0, 0.0)
        glVertex3f(0.5, -0.5, 0.5)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.5, -0.5, -0.5)
        glColor3f(0.0, 0.0, 0.0)
        glVertex3f(-0.5, -0.5, -0.5)
        glEnd()
        glPopMatrix()
        glFinish()
        SwapBuffers(wglGetCurrentDC())
        self.busy = 0

def test():
    if False:
        print('Hello World!')
    template = docview.DocTemplate(None, None, None, CubeView)
    template.OpenDocumentFile(None)
if __name__ == '__main__':
    test()