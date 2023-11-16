from datetime import datetime
from functools import wraps
from typing import Callable
from dateutil.tz import tzlocal
from django.core.management.commands.runserver import Command as DjangoCommand
from typing_extensions import override

def output_styler(style_func: Callable[[str], str]) -> Callable[[str], str]:
    if False:
        while True:
            i = 10
    date_prefix = datetime.now(tzlocal()).strftime('%B %d, %Y - ')

    @wraps(style_func)
    def _wrapped_style_func(message: str) -> str:
        if False:
            while True:
                i = 10
        if message == 'Performing system checks...\n\n' or message.startswith(('System check identified no issues', date_prefix)):
            message = ''
        elif 'Quit the server with ' in message:
            message = 'Django process (re)started. ' + message[message.index('Quit the server with '):]
        return style_func(message)
    return _wrapped_style_func

class Command(DjangoCommand):

    @override
    def inner_run(self, *args: object, **options: object) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.stdout.style_func = output_styler(self.stdout.style_func)
        super().inner_run(*args, **options)