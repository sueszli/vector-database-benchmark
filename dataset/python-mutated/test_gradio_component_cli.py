import textwrap
from pathlib import Path
import pytest
from gradio.cli.commands.components._create_utils import OVERRIDES
from gradio.cli.commands.components.build import _build
from gradio.cli.commands.components.create import _create
from gradio.cli.commands.components.install_component import _install
from gradio.cli.commands.components.show import _show

@pytest.mark.parametrize('template', ['Row', 'Column', 'Tabs', 'Group', 'Accordion', 'AnnotatedImage', 'HighlightedText', 'BarPlot', 'ClearButton', 'ColorPicker', 'DuplicateButton', 'LinePlot', 'LogoutButton', 'LoginButton', 'ScatterPlot', 'UploadButton', 'JSON', 'FileExplorer', 'Model3D'])
def test_template_override_component(template, tmp_path):
    if False:
        while True:
            i = 10
    _create('MyComponent', tmp_path, template=template, overwrite=True, install=False)
    app = (tmp_path / 'demo' / 'app.py').read_text()
    answer = textwrap.dedent(f"\nimport gradio as gr\nfrom gradio_mycomponent import MyComponent\n\n{OVERRIDES[template].demo_code.format(name='MyComponent')}\n\ndemo.launch()\n")
    assert app.strip() == answer.strip()
    assert (tmp_path / 'backend' / 'gradio_mycomponent' / 'mycomponent.py').exists()

def test_raise_error_component_template_does_not_exist(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError, match='Cannot find NonExistentComponent in gradio.components or gradio.layouts'):
        _create('MyComponent', tmp_path, template='NonExistentComponent', overwrite=True, install=False)

def test_do_not_replace_class_name_in_import_statement(tmp_path):
    if False:
        i = 10
        return i + 15
    _create('MyImage', template='Image', directory=tmp_path, overwrite=True, install=False)
    code = (tmp_path / 'backend' / 'gradio_myimage' / 'myimage.py').read_text()
    assert 'from PIL import Image as _Image' in code
    assert 'class MyImage' in code
    assert '_Image.Image' in code

def test_raises_if_directory_exists(tmp_path):
    if False:
        i = 10
        return i + 15
    with pytest.raises(Exception):
        _create('MyComponent', tmp_path)

def test_show(capsys):
    if False:
        i = 10
        return i + 15
    _show()
    (stdout, _) = capsys.readouterr()
    assert 'Form Component' in stdout
    assert 'Beginner Friendly' in stdout
    assert 'Layout' in stdout
    assert 'Dataframe' not in stdout
    assert 'Dataset' not in stdout

@pytest.mark.xfail
@pytest.mark.parametrize('template', ['Audio', 'Video', 'Image', 'Textbox'])
def test_build(template, tmp_path):
    if False:
        while True:
            i = 10
    _create('TestTextbox', template=template, directory=tmp_path, overwrite=True, install=True)
    _build(tmp_path, build_frontend=True)
    template_dir: Path = tmp_path.resolve() / 'backend' / 'gradio_testtextbox' / 'templates'
    assert template_dir.exists() and template_dir.is_dir()
    assert list(template_dir.glob('**/index.js'))
    assert (tmp_path / 'dist').exists() and list((tmp_path / 'dist').glob('*.whl'))

def test_install(tmp_path):
    if False:
        while True:
            i = 10
    _create('TestTextbox', template='Textbox', directory=tmp_path, overwrite=True, install=False)
    assert not (tmp_path / 'frontend' / 'node_modules').exists()
    _install(tmp_path)
    assert (tmp_path / 'frontend' / 'node_modules').exists()

def test_fallback_template_app(tmp_path):
    if False:
        i = 10
        return i + 15
    _create('SimpleComponent2', directory=tmp_path, overwrite=True, install=False)
    app = (tmp_path / 'demo' / 'app.py').read_text()
    answer = textwrap.dedent('\n\nimport gradio as gr\nfrom gradio_simplecomponent2 import SimpleComponent2\n\n\nwith gr.Blocks() as demo:\n    gr.Markdown("# Change the value (keep it JSON) and the front-end will update automatically.")\n    SimpleComponent2(value={"message": "Hello from Gradio!"}, label="Static")\n\n\ndemo.launch()\n\n')
    assert app.strip() == answer.strip()