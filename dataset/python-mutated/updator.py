"""Has classes that help updating Prompt sections using Threads."""
import concurrent.futures
import threading
import typing as tp
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import PygmentsTokens
from xonsh.built_ins import XSH
from xonsh.prompt.base import ParsedTokens
from xonsh.style_tools import partial_color_tokenize, style_as_faded

class Executor:
    """Caches thread results across prompts."""

    def __init__(self):
        if False:
            return 10
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=XSH.env['ASYNC_PROMPT_THREAD_WORKERS'])
        self.thread_results = {}

    def submit(self, func: tp.Callable, field: str):
        if False:
            while True:
                i = 10
        future = self.thread_pool.submit(self._run_func, func, field)
        place_holder = '{' + field + '}'
        return (future, self.thread_results[field] if field in self.thread_results else place_holder, place_holder)

    def _run_func(self, func, field):
        if False:
            for i in range(10):
                print('nop')
        'Run the callback and store the result.'
        result = func()
        self.thread_results[field] = result if result is None else style_as_faded(result)
        return result

class AsyncPrompt:
    """Represent an asynchronous prompt."""

    def __init__(self, name: str, session: PromptSession, executor: Executor):
        if False:
            i = 10
            return i + 15
        "\n\n        Parameters\n        ----------\n        name: str\n            what prompt to update. One of ['message', 'rprompt', 'bottom_toolbar']\n        session: PromptSession\n            current ptk session\n        "
        self.name = name
        self.tokens: tp.Optional[ParsedTokens] = None
        self.timer = None
        self.session = session
        self.executor = executor
        self.futures: dict[concurrent.futures.Future, tuple[str, tp.Optional[int], tp.Optional[str], tp.Optional[str]]] = {}

    def start_update(self, on_complete):
        if False:
            return 10
        'Listen on futures and update the prompt as each one completed.\n\n        Timer is used to avoid clogging multiple calls at the same time.\n\n        Parameters\n        -----------\n        on_complete:\n            callback to notify after all the futures are completed\n        '
        if not self.tokens:
            print(f'Warn: AsyncPrompt is created without tokens - {self.name}')
            return
        for fut in concurrent.futures.as_completed(self.futures):
            try:
                val = fut.result()
            except concurrent.futures.CancelledError:
                continue
            if fut not in self.futures:
                continue
            (placeholder, idx, spec, conv) = self.futures[fut]
            if isinstance(idx, int):
                self.tokens.update(idx, val, spec, conv)
            else:
                for (idx, ptok) in enumerate(self.tokens.tokens):
                    if placeholder in ptok.value:
                        val = ptok.value.replace(placeholder, val)
                        self.tokens.update(idx, val, spec, conv)
            self.invalidate()
        on_complete(self.name)

    def invalidate(self):
        if False:
            while True:
                i = 10
        'Create a timer to update the prompt. The timing can be configured through env variables.\n        threading.Timer is used to stop calling invalidate frequently.\n        '
        from xonsh.ptk_shell.shell import tokenize_ansi
        if self.timer:
            self.timer.cancel()

        def _invalidate():
            if False:
                for i in range(10):
                    print('nop')
            new_prompt = self.tokens.process()
            formatted_tokens = tokenize_ansi(PygmentsTokens(partial_color_tokenize(new_prompt)))
            setattr(self.session, self.name, formatted_tokens)
            self.session.app.invalidate()
        self.timer = threading.Timer(XSH.env['ASYNC_INVALIDATE_INTERVAL'], _invalidate)
        self.timer.start()

    def stop(self):
        if False:
            return 10
        'Stop any running threads'
        for fut in self.futures:
            fut.cancel()
        self.futures.clear()

    def submit_section(self, func: tp.Callable, field: str, idx: tp.Optional[int]=None, spec: tp.Optional[str]=None, conv=None):
        if False:
            print('Hello World!')
        (future, intermediate_value, placeholder) = self.executor.submit(func, field)
        self.futures[future] = (placeholder, idx, spec, conv)
        return intermediate_value

class PromptUpdator:
    """Handle updating multiple AsyncPrompt instances prompt/rprompt/bottom_toolbar"""

    def __init__(self, shell):
        if False:
            i = 10
            return i + 15
        from xonsh.ptk_shell.shell import PromptToolkitShell
        self.prompts: dict[str, AsyncPrompt] = {}
        self.shell: PromptToolkitShell = shell
        self.executor = Executor()
        self.futures = {}
        self.attrs_loaded = None

    def add(self, prompt_name: tp.Optional[str]) -> tp.Optional[AsyncPrompt]:
        if False:
            for i in range(10):
                print('nop')
        if prompt_name is None:
            return None
        self.stop(prompt_name)
        self.prompts[prompt_name] = AsyncPrompt(prompt_name, self.shell.prompter, self.executor)
        return self.prompts[prompt_name]

    def add_attrs(self):
        if False:
            while True:
                i = 10
        for (attr, val) in self.shell.get_lazy_ptk_kwargs():
            setattr(self.shell.prompter, attr, val)
        self.shell.prompter.app.invalidate()

    def start(self):
        if False:
            while True:
                i = 10
        'after ptk prompt is created, update it in background.'
        if not self.attrs_loaded:
            self.attrs_loaded = self.executor.thread_pool.submit(self.add_attrs)
        prompts = list(self.prompts)
        for pt_name in prompts:
            if pt_name not in self.prompts:
                continue
            prompt = self.prompts[pt_name]
            future = self.executor.thread_pool.submit(prompt.start_update, self.on_complete)
            self.futures[pt_name] = future

    def stop(self, prompt_name: str):
        if False:
            i = 10
            return i + 15
        if prompt_name in self.prompts:
            self.prompts[prompt_name].stop()
        if prompt_name in self.futures:
            self.futures[prompt_name].cancel()

    def on_complete(self, prompt_name):
        if False:
            i = 10
            return i + 15
        self.prompts.pop(prompt_name, None)
        self.futures.pop(prompt_name, None)