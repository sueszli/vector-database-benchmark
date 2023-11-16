import os
from contextlib import contextmanager
from kitty.utils import get_editor
from . import BaseTest

@contextmanager
def patch_env(**kw):
    if False:
        print('Hello World!')
    orig = os.environ.copy()
    for (k, v) in kw.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    yield
    os.environ.clear()
    os.environ.update(orig)

class TestOpenActions(BaseTest):

    def test_parsing_of_open_actions(self):
        if False:
            for i in range(10):
                print('nop')
        from kitty.open_actions import KeyAction, actions_for_url
        self.set_options()
        spec = '\nprotocol file\nmime text/*\nfragment_matches .\nAcTion launch $EDITOR $FILE_PATH $FRAGMENT\naction\n\nprotocol file\nmime text/*\naction ignored\n\next py,txt\naction one\naction two\n'

        def actions(url):
            if False:
                while True:
                    i = 10
            with patch_env(FILE_PATH='notgood'):
                return tuple(actions_for_url(url, spec))

        def single(url, func, *args):
            if False:
                for i in range(10):
                    print('nop')
            acts = actions(url)
            self.ae(len(acts), 1)
            self.ae(acts[0].func, func)
            self.ae(acts[0].args, args)
        single('file://hostname/tmp/moo.txt#23', 'launch', *get_editor(), '/tmp/moo.txt', '23')
        single('some thing.txt', 'ignored')
        self.ae(actions('x:///a.txt'), (KeyAction('one', ()), KeyAction('two', ())))