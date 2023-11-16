# Copyright 2016 Hardcoded Software (http://www.hardcoded.net)
#
# This software is licensed under the "GPLv3" License as described in the "LICENSE" file,
# which should be included with this package. The terms are also available at
# http://www.gnu.org/licenses/gpl-3.0.html

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
        """Show the dialog."""

    def close(self) -> None:
        """Close the dialog."""

    def set_progress(self, progress: int) -> None:
        """Set the progress of the progress bar to ``progress``.

        Not all jobs are equally responsive on their job progress report and it is recommended that
        you put your progressbar in "indeterminate" mode as long as you haven't received the first
        ``set_progress()`` call to avoid letting the user think that the app is frozen.

        :param int progress: a value between ``0`` and ``100``.
        """


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

    def __init__(
        self,
        finish_func: Callable[[Union[str, None]], None],
        error_func: Callable[[Union[str, None], Exception], bool] = None,
    ) -> None:
        # finish_func(jobid) is the function that is called when a job is completed.
        GUIObject.__init__(self)
        ThreadedJobPerformer.__init__(self)
        self._finish_func = finish_func
        self._error_func = error_func
        #: :class:`.TextField`. It contains that title you gave the job on :meth:`run`.
        self.jobdesc_textfield = TextField()
        #: :class:`.TextField`. It contains the job textual update that the function might yield
        #: during its course.
        self.progressdesc_textfield = TextField()
        self.jobid: Union[str, None] = None

    def cancel(self) -> None:
        """Call for a user-initiated job cancellation."""
        # The UI is sometimes a bit buggy and calls cancel() on self.view.close(). We just want to
        # make sure that this doesn't lead us to think that the user acually cancelled the task, so
        # we verify that the job is still running.
        if self._job_running:
            self.job_cancelled = True

    def pulse(self) -> None:
        """Update progress reports in the GUI.

        Call this regularly from the GUI main run loop. The values might change before
        :meth:`ProgressWindowView.set_progress` happens.

        If the job is finished, ``pulse()`` will take care of closing the window and re-raising any
        exception that might have been raised during the job (in the main thread this time). If
        there was no exception, ``finish_func(jobid)`` is called to let you take appropriate action.
        """
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

    def run(self, jobid: str, title: str, target: Callable, args: Tuple = ()):
        """Starts a threaded job.

        The ``target`` function will be sent, as its first argument, a :class:`.Job` instance which
        it can use to report on its progress.

        :param jobid: Arbitrary identifier which will be passed to ``finish_func()`` at the end.
        :param title: A title for the task you're starting.
        :param target: The function that does your famous long running job.
        :param args: additional arguments that you want to send to ``target``.
        """
        # target is a function with its first argument being a Job. It can then be followed by other
        # arguments which are passed as `args`.
        self.jobid = jobid
        self.progressdesc_textfield.text = ""
        j = self.create_job()
        args = tuple([j] + list(args))
        self.run_threaded(target, args)
        self.jobdesc_textfield.text = title
        self.view.show()
