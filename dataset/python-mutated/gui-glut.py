"""Simple GLUT example to manually test event loop integration.

This is meant to run tests manually in ipython as:

In [5]: %gui glut

In [6]: %run gui-glut.py

In [7]: gl.glClearColor(1,1,1,1)
"""
import sys
import OpenGL.GL as gl
import OpenGL.GLUT as glut

def close():
    if False:
        i = 10
        return i + 15
    glut.glutDestroyWindow(glut.glutGetWindow())

def display():
    if False:
        return 10
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    glut.glutSwapBuffers()

def resize(width, height):
    if False:
        while True:
            i = 10
    gl.glViewport(0, 0, width, height + 4)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glOrtho(0, width, 0, height + 4, -1, 1)
    gl.glMatrixMode(gl.GL_MODELVIEW)
if glut.glutGetWindow() > 0:
    interactive = True
    glut.glutInit(sys.argv)
    glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA | glut.GLUT_DEPTH)
else:
    interactive = False
glut.glutCreateWindow(b'gui-glut')
glut.glutDisplayFunc(display)
glut.glutReshapeFunc(resize)
if sys.platform == 'darwin' and (not bool(glut.HAVE_FREEGLUT)):
    glut.glutWMCloseFunc(close)
gl.glClearColor(0, 0, 0, 1)
if not interactive:
    glut.glutMainLoop()