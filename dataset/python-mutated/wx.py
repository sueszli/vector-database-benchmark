"""Enable wxPython to be used interactively in prompt_toolkit
"""
import sys
import signal
import time
from timeit import default_timer as clock
import wx

def ignore_keyboardinterrupts(func):
    if False:
        for i in range(10):
            print('nop')
    'Decorator which causes KeyboardInterrupt exceptions to be ignored during\n    execution of the decorated function.\n\n    This is used by the inputhook functions to handle the event where the user\n    presses CTRL+C while IPython is idle, and the inputhook loop is running. In\n    this case, we want to ignore interrupts.\n    '

    def wrapper(*args, **kwargs):
        if False:
            print('Hello World!')
        try:
            func(*args, **kwargs)
        except KeyboardInterrupt:
            pass
    return wrapper

@ignore_keyboardinterrupts
def inputhook_wx1(context):
    if False:
        while True:
            i = 10
    'Run the wx event loop by processing pending events only.\n\n    This approach seems to work, but its performance is not great as it\n    relies on having PyOS_InputHook called regularly.\n    '
    app = wx.GetApp()
    if app is not None:
        assert wx.Thread_IsMain()
        evtloop = wx.EventLoop()
        ea = wx.EventLoopActivator(evtloop)
        while evtloop.Pending():
            evtloop.Dispatch()
        app.ProcessIdle()
        del ea
    return 0

class EventLoopTimer(wx.Timer):

    def __init__(self, func):
        if False:
            for i in range(10):
                print('nop')
        self.func = func
        wx.Timer.__init__(self)

    def Notify(self):
        if False:
            while True:
                i = 10
        self.func()

class EventLoopRunner(object):

    def Run(self, time, input_is_ready):
        if False:
            while True:
                i = 10
        self.input_is_ready = input_is_ready
        self.evtloop = wx.EventLoop()
        self.timer = EventLoopTimer(self.check_stdin)
        self.timer.Start(time)
        self.evtloop.Run()

    def check_stdin(self):
        if False:
            print('Hello World!')
        if self.input_is_ready():
            self.timer.Stop()
            self.evtloop.Exit()

@ignore_keyboardinterrupts
def inputhook_wx2(context):
    if False:
        return 10
    'Run the wx event loop, polling for stdin.\n\n    This version runs the wx eventloop for an undetermined amount of time,\n    during which it periodically checks to see if anything is ready on\n    stdin.  If anything is ready on stdin, the event loop exits.\n\n    The argument to elr.Run controls how often the event loop looks at stdin.\n    This determines the responsiveness at the keyboard.  A setting of 1000\n    enables a user to type at most 1 char per second.  I have found that a\n    setting of 10 gives good keyboard response.  We can shorten it further,\n    but eventually performance would suffer from calling select/kbhit too\n    often.\n    '
    app = wx.GetApp()
    if app is not None:
        assert wx.Thread_IsMain()
        elr = EventLoopRunner()
        elr.Run(time=10, input_is_ready=context.input_is_ready)
    return 0

@ignore_keyboardinterrupts
def inputhook_wx3(context):
    if False:
        return 10
    'Run the wx event loop by processing pending events only.\n\n    This is like inputhook_wx1, but it keeps processing pending events\n    until stdin is ready.  After processing all pending events, a call to\n    time.sleep is inserted.  This is needed, otherwise, CPU usage is at 100%.\n    This sleep time should be tuned though for best performance.\n    '
    app = wx.GetApp()
    if app is not None:
        assert wx.Thread_IsMain()
        if not callable(signal.getsignal(signal.SIGINT)):
            signal.signal(signal.SIGINT, signal.default_int_handler)
        evtloop = wx.EventLoop()
        ea = wx.EventLoopActivator(evtloop)
        t = clock()
        while not context.input_is_ready():
            while evtloop.Pending():
                t = clock()
                evtloop.Dispatch()
            app.ProcessIdle()
            used_time = clock() - t
            if used_time > 10.0:
                time.sleep(1.0)
            elif used_time > 0.1:
                time.sleep(0.05)
            else:
                time.sleep(0.001)
        del ea
    return 0

@ignore_keyboardinterrupts
def inputhook_wxphoenix(context):
    if False:
        return 10
    'Run the wx event loop until the user provides more input.\n\n    This input hook is suitable for use with wxPython >= 4 (a.k.a. Phoenix).\n\n    It uses the same approach to that used in\n    ipykernel.eventloops.loop_wx. The wx.MainLoop is executed, and a wx.Timer\n    is used to periodically poll the context for input. As soon as input is\n    ready, the wx.MainLoop is stopped.\n    '
    app = wx.GetApp()
    if app is None:
        return
    if context.input_is_ready():
        return
    assert wx.IsMainThread()
    poll_interval = 100
    timer = wx.Timer()

    def poll(ev):
        if False:
            print('Hello World!')
        if context.input_is_ready():
            timer.Stop()
            app.ExitMainLoop()
    timer.Start(poll_interval)
    timer.Bind(wx.EVT_TIMER, poll)
    if not callable(signal.getsignal(signal.SIGINT)):
        signal.signal(signal.SIGINT, signal.default_int_handler)
    app.SetExitOnFrameDelete(False)
    app.MainLoop()
major_version = 3
try:
    major_version = int(wx.__version__[0])
except Exception:
    pass
if major_version >= 4:
    inputhook = inputhook_wxphoenix
elif sys.platform == 'darwin':
    inputhook = inputhook_wx2
else:
    inputhook = inputhook_wx3