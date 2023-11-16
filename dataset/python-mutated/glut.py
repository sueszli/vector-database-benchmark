"""GLUT Input hook for interactive use with prompt_toolkit
"""
import sys
import time
import signal
import OpenGL.GLUT as glut
import OpenGL.platform as platform
from timeit import default_timer as clock
glut_fps = 60
glut_display_mode = glut.GLUT_DOUBLE | glut.GLUT_RGBA | glut.GLUT_DEPTH
glutMainLoopEvent = None
if sys.platform == 'darwin':
    try:
        glutCheckLoop = platform.createBaseFunction('glutCheckLoop', dll=platform.GLUT, resultType=None, argTypes=[], doc='glutCheckLoop(  ) -> None', argNames=())
    except AttributeError as e:
        raise RuntimeError('Your glut implementation does not allow interactive sessions. Consider installing freeglut.') from e
    glutMainLoopEvent = glutCheckLoop
elif glut.HAVE_FREEGLUT:
    glutMainLoopEvent = glut.glutMainLoopEvent
else:
    raise RuntimeError('Your glut implementation does not allow interactive sessions. Consider installing freeglut.')

def glut_display():
    if False:
        print('Hello World!')
    pass

def glut_idle():
    if False:
        for i in range(10):
            print('nop')
    pass

def glut_close():
    if False:
        print('Hello World!')
    glut.glutHideWindow()
    glutMainLoopEvent()

def glut_int_handler(signum, frame):
    if False:
        for i in range(10):
            print('nop')
    signal.signal(signal.SIGINT, signal.default_int_handler)
    print('\nKeyboardInterrupt')
glut.glutInit(sys.argv)
glut.glutInitDisplayMode(glut_display_mode)
if bool(glut.glutSetOption):
    glut.glutSetOption(glut.GLUT_ACTION_ON_WINDOW_CLOSE, glut.GLUT_ACTION_GLUTMAINLOOP_RETURNS)
glut.glutCreateWindow(b'ipython')
glut.glutReshapeWindow(1, 1)
glut.glutHideWindow()
glut.glutWMCloseFunc(glut_close)
glut.glutDisplayFunc(glut_display)
glut.glutIdleFunc(glut_idle)

def inputhook(context):
    if False:
        while True:
            i = 10
    'Run the pyglet event loop by processing pending events only.\n\n    This keeps processing pending events until stdin is ready.  After\n    processing all pending events, a call to time.sleep is inserted.  This is\n    needed, otherwise, CPU usage is at 100%.  This sleep time should be tuned\n    though for best performance.\n    '
    signal.signal(signal.SIGINT, glut_int_handler)
    try:
        t = clock()
        if glut.glutGetWindow() == 0:
            glut.glutSetWindow(1)
            glutMainLoopEvent()
            return 0
        while not context.input_is_ready():
            glutMainLoopEvent()
            used_time = clock() - t
            if used_time > 10.0:
                time.sleep(1.0)
            elif used_time > 0.1:
                time.sleep(0.05)
            else:
                time.sleep(0.001)
    except KeyboardInterrupt:
        pass