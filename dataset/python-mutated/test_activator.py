from __future__ import annotations
from argparse import Namespace
from virtualenv.activation.activator import Activator

def test_activator_prompt_cwd(monkeypatch, tmp_path):
    if False:
        print('Hello World!')

    class FakeActivator(Activator):

        def generate(self, creator):
            if False:
                while True:
                    i = 10
            raise NotImplementedError
    cwd = tmp_path / 'magic'
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    activator = FakeActivator(Namespace(prompt='.'))
    assert activator.flag_prompt == 'magic'