import time
from threading import Lock
from ..utils.struct.lock import lock

class CaptchaManager:

    def __init__(self, core):
        if False:
            return 10
        self.lock = Lock()
        self.pyload = core
        self._ = core._
        self.tasks = []
        self.ids = 0

    def new_task(self, format, params, result_type):
        if False:
            return 10
        task = CaptchaTask(self.ids, format, params, result_type)
        self.ids += 1
        return task

    @lock
    def remove_task(self, task):
        if False:
            i = 10
            return i + 15
        if task in self.tasks:
            self.tasks.remove(task)

    @lock
    def get_task(self):
        if False:
            print('Hello World!')
        for task in self.tasks:
            if task.status in ('waiting', 'shared-user'):
                return task
        return None

    @lock
    def get_task_by_id(self, tid):
        if False:
            for i in range(10):
                print('nop')
        for task in self.tasks:
            if task.id == str(tid):
                return task
        return None

    def handle_captcha(self, task, timeout):
        if False:
            while True:
                i = 10
        cli = self.pyload.is_client_connected()
        task.set_waiting(timeout)
        for plugin in self.pyload.addon_manager.active_plugins():
            try:
                plugin.captcha_task(task)
            except Exception:
                self.pyload.log.warning(self.pyload._('Unable to create captcha task'), exc_info=self.pyload.debug > 1, stack_info=self.pyload.debug > 2)
        if task.handler or cli:
            self.tasks.append(task)
            return True
        task.error = self._('No Client connected for captcha decrypting')
        return False

class CaptchaTask:

    def __init__(self, id, format, params={}, result_type='textual'):
        if False:
            i = 10
            return i + 15
        self.id = str(id)
        self.captcha_params = params
        self.captcha_format = format
        self.captcha_result_type = result_type
        self.handler = []
        self.result = None
        self.wait_until = 0
        self.error = None
        self.status = 'init'
        self.data = {}

    def get_captcha(self):
        if False:
            print('Hello World!')
        return (self.captcha_params, self.captcha_format, self.captcha_result_type)

    def set_result(self, result):
        if False:
            while True:
                i = 10
        if self.is_textual() or self.is_interactive() or self.is_invisible():
            self.result = result
        elif self.is_positional():
            try:
                parts = result.split(',')
                self.result = (int(parts[0]), int(parts[1]))
            except Exception:
                self.result = None

    def get_result(self):
        if False:
            return 10
        return self.result

    def get_status(self):
        if False:
            while True:
                i = 10
        return self.status

    def set_waiting(self, sec):
        if False:
            for i in range(10):
                print('nop')
        '\n        let the captcha wait secs for the solution.\n        '
        self.wait_until = max(time.time() + sec, self.wait_until)
        self.status = 'waiting'

    def is_waiting(self):
        if False:
            while True:
                i = 10
        if self.result or self.error or time.time() > self.wait_until:
            return False
        return True

    def is_textual(self):
        if False:
            return 10
        '\n        returns if text is written on the captcha.\n        '
        return self.captcha_result_type == 'textual'

    def is_positional(self):
        if False:
            print('Hello World!')
        '\n        returns if user have to click a specific region on the captcha.\n        '
        return self.captcha_result_type == 'positional'

    def is_interactive(self):
        if False:
            while True:
                i = 10
        '\n        returns if user has to solve the captcha in an interactive iframe.\n        '
        return self.captcha_result_type == 'interactive'

    def is_invisible(self):
        if False:
            print('Hello World!')
        '\n        returns if invisible (browser only, no user interaction) captcha.\n        '
        return self.captcha_result_type == 'invisible'

    def set_waiting_for_user(self, exclusive):
        if False:
            i = 10
            return i + 15
        if exclusive:
            self.status = 'user'
        else:
            self.status = 'shared-user'

    def timed_out(self):
        if False:
            while True:
                i = 10
        return time.time() > self.wait_until

    def invalid(self):
        if False:
            return 10
        '\n        indicates the captcha was not correct.\n        '
        [x.captcha_invalid(self) for x in self.handler]

    def correct(self):
        if False:
            print('Hello World!')
        [x.captcha_correct(self) for x in self.handler]

    def __str__(self):
        if False:
            print('Hello World!')
        return f"<CaptchaTask '{self.id}'>"