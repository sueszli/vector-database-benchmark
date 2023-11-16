from typing import Callable, Tuple, Union
from hscommon.jobprogress.performer import ThreadedJobPerformer
from hscommon.gui.base import GUIObject
from hscommon.gui.text_field import TextField

class ProgressWindowView:
    """Expected interface for :class:`ProgressWindow`'s view.

    *Not actually used in the code. For documentation purposes only.*

    Our view, some kind window with a progress bar, two labels and a cancel button, is expected
    to properly respond to its callbacks.

    It's also expected to call :meth:`ProgressWindow.cancel` when the cancel button is clicked.
    """

    def show(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Show the dialog.'

    def close(self) -> None:
        if False:
            i = 10
            return i + 15
        'Close the dialog.'

    def set_progress(self, progress: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set the progress of the progress bar to ``progress``.\n\n        Not all jobs are equally responsive on their job progress report and it is recommended that\n        you put your progressbar in "indeterminate" mode as long as you haven\'t received the first\n        ``set_progress()`` call to avoid letting the user think that the app is frozen.\n\n        :param int progress: a value between ``0`` and ``100``.\n        '

class ProgressWindow(GUIObject, ThreadedJobPerformer):
    """Cross-toolkit GUI-enabled progress window.

    This class allows you to run a long running, job enabled function in a separate thread and
    allow the user to follow its progress with a progress dialog.

    To use it, you start your long-running job with :meth:`run` and then have your UI layer
    regularly call :meth:`pulse` to refresh the job status in the UI. It is advised that you call
    :meth:`pulse` in the main thread because GUI toolkit usually only support calling UI-related
    functions from the main thread.

    We subclass :class:`.GUIObject` and :class:`.ThreadedJobPerformer`.
    Expected view: :class:`ProgressWindowView`.

    :param finish_func: A function ``f(jobid)`` that is called when a job is completed. ``jobid`` is
                        an arbitrary id passed to :meth:`run`.
    :param error_func: A function ``f(jobid, err)`` that is called when an exception is raised and
                       unhandled during the job. If not specified, the error will be raised in the
                       main thread. If it's specified, it's your responsibility to raise the error
                       if you want to. If the function returns ``True``, ``finish_func()`` will be
                       called as if the job terminated normally.
    """

    def __init__(self, finish_func: Callable[[Union[str, None]], None], error_func: Callable[[Union[str, None], Exception], bool]=None) -> None:
        if False:
            return 10
        GUIObject.__init__(self)
        ThreadedJobPerformer.__init__(self)
        self._finish_func = finish_func
        self._error_func = error_func
        self.jobdesc_textfield = TextField()
        self.progressdesc_textfield = TextField()
        self.jobid: Union[str, None] = None

    def cancel(self) -> None:
        if False:
            i = 10
            return i + 15
        'Call for a user-initiated job cancellation.'
        if self._job_running:
            self.job_cancelled = True

    def pulse(self) -> None:
        if False:
            i = 10
            return i + 15
        'Update progress reports in the GUI.\n\n        Call this regularly from the GUI main run loop. The values might change before\n        :meth:`ProgressWindowView.set_progress` happens.\n\n        If the job is finished, ``pulse()`` will take care of closing the window and re-raising any\n        exception that might have been raised during the job (in the main thread this time). If\n        there was no exception, ``finish_func(jobid)`` is called to let you take appropriate action.\n        '
        last_progress = self.last_progress
        last_desc = self.last_desc
        if not self._job_running or last_progress is None:
            self.view.close()
            should_continue = True
            if self.last_error is not None:
                err = self.last_error.with_traceback(self.last_traceback)
                if self._error_func is not None:
                    should_continue = self._error_func(self.jobid, err)
                else:
                    raise err
            if not self.job_cancelled and should_continue:
                self._finish_func(self.jobid)
            return
        if self.job_cancelled:
            return
        if last_desc:
            self.progressdesc_textfield.text = last_desc
        self.view.set_progress(last_progress)

    def run(self, jobid: str, title: str, target: Callable, args: Tuple=()):
        if False:
            while True:
                i = 10
        "Starts a threaded job.\n\n        The ``target`` function will be sent, as its first argument, a :class:`.Job` instance which\n        it can use to report on its progress.\n\n        :param jobid: Arbitrary identifier which will be passed to ``finish_func()`` at the end.\n        :param title: A title for the task you're starting.\n        :param target: The function that does your famous long running job.\n        :param args: additional arguments that you want to send to ``target``.\n        "
        self.jobid = jobid
        self.progressdesc_textfield.text = ''
        j = self.create_job()
        args = tuple([j] + list(args))
        self.run_threaded(target, args)
        self.jobdesc_textfield.text = title
        self.view.show()