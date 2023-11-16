import dataclasses
from pathlib import Path
from typing import List
import pytest
import gradio
import gradio as gr
from gradio.cli.commands.reload import _setup_config
from gradio.networking import Server

def build_demo():
    if False:
        return 10
    with gr.Blocks() as demo:
        gr.Textbox('')
    return demo

@dataclasses.dataclass
class Config:
    filename: str
    path: Path
    watch_dirs: List[str]
    demo_name: str

class TestReload:

    @pytest.fixture(autouse=True)
    def argv(self):
        if False:
            print('Hello World!')
        return ['demo/calculator/run.py']

    @pytest.fixture
    def config(self, monkeypatch, argv) -> Config:
        if False:
            for i in range(10):
                print('nop')
        monkeypatch.setattr('sys.argv', ['gradio'] + argv)
        name = argv[1].replace('--demo-name', '').strip() if len(argv) > 1 else 'demo'
        return Config(*_setup_config(argv[0], name))

    @pytest.fixture(params=[{}])
    def reloader(self, config):
        if False:
            print('Hello World!')
        reloader = Server(config)
        reloader.should_exit = True
        yield reloader
        reloader.close()

    def test_config_default_app(self, config):
        if False:
            while True:
                i = 10
        assert config.filename == 'run'

    @pytest.mark.parametrize('argv', [['demo/calculator/run.py', '--demo-name test']])
    def test_config_custom_app(self, config):
        if False:
            while True:
                i = 10
        assert config.filename == 'run'
        assert config.demo_name == 'test'

    def test_config_watch_gradio(self, config):
        if False:
            i = 10
            return i + 15
        gradio_dir = str(Path(gradio.__file__).parent)
        assert gradio_dir in config.watch_dirs

    def test_config_watch_app(self, config):
        if False:
            i = 10
            return i + 15
        demo_dir = str(Path('demo/calculator/run.py').resolve().parent)
        assert demo_dir in config.watch_dirs