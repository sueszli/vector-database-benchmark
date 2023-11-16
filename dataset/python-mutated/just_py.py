import inspect
import os
import sys
from subprocess import Popen
from time import sleep
from typing import Callable, Optional
import lightning.app
from lightning.app.frontend.frontend import Frontend
from lightning.app.utilities.log import get_logfile

class JustPyFrontend(Frontend):
    """A frontend for wrapping JustPy code in your LightingFlow.

    Return this in your `LightningFlow.configure_layout()` method if you wish to build the UI with ``justpy``.
    To use this frontend, you must first install the `justpy` package (if running locally):

    .. code-block:: bash

        pip install justpy

    Arguments:
        render_fn: A function that contains your justpy code. This function must accept exactly one argument, the
            ``AppState`` object which you can use to access variables in your flow (see example below).

    Example:

        In your LightningFlow, override the method `configure_layout`:

        .. code-block:: python

            from typing import Callable
            from lightning import LightningApp, LightningFlow
            from lightning.app.frontend import JustPyFrontend


            class Flow(LightningFlow):
                def __init__(self):
                    super().__init__()
                    self.counter = 0

                def run(self):
                    print(self.counter)

                def configure_layout(self):
                    return JustPyFrontend(render_fn=render_fn)


            def render_fn(get_state: Callable) -> Callable:
                import justpy as jp

                def my_click(self, *_):
                    state = get_state()
                    old_counter = state.counter
                    state.counter += 1
                    self.text = f"Click Me ! Old Counter: {old_counter} New Counter: {state.counter}"

                def webpage():
                    wp = jp.WebPage()
                    d = jp.Div(text="Hello ! Click Me!")
                    d.on("click", my_click)
                    wp.add(d)
                    return wp

                return webpage


            app = LightningApp(Flow())

    """

    def __init__(self, render_fn: Callable) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        if inspect.ismethod(render_fn):
            raise TypeError("The `JustPyFrontend` doesn't support `render_fn` being a method. Please, use a pure function.")
        self.render_fn = render_fn
        self._process: Optional[Popen] = None

    def start_server(self, host: str, port: int, root_path: str='') -> None:
        if False:
            while True:
                i = 10
        env = os.environ.copy()
        env['LIGHTNING_FLOW_NAME'] = self.flow.name
        env['LIGHTNING_RENDER_FUNCTION'] = self.render_fn.__name__
        env['LIGHTNING_RENDER_MODULE_FILE'] = inspect.getmodule(self.render_fn).__file__
        env['LIGHTNING_HOST'] = host
        env['LIGHTNING_PORT'] = str(port)
        std_out_out = get_logfile('output.log')
        path = os.path.join(os.path.dirname(lightning.app.frontend.just_py.__file__), 'just_py_base.py')
        with open(std_out_out, 'wb') as stdout:
            self._process = Popen(f'{sys.executable} {path}', env=env, stdout=stdout, stderr=sys.stderr, shell=True)
        sleep(1)

    def stop_server(self) -> None:
        if False:
            while True:
                i = 10
        assert self._process
        self._process.terminate()