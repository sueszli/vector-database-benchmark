from panda3d.core import Filename, NodePath, LoaderFileTypeRegistry
from direct.showbase.Loader import Loader
import pytest
import sys

@pytest.fixture
def loader():
    if False:
        print('Hello World!')
    return Loader(base=None)

@pytest.fixture
def temp_model():
    if False:
        return 10
    from panda3d.core import ModelPool, ModelRoot
    root = ModelRoot('model')
    root.fullpath = '/test-model.bam'
    ModelPool.add_model(root.fullpath, root)
    yield root.fullpath
    ModelPool.release_model(root.fullpath)

def test_load_model_filename(loader, temp_model):
    if False:
        while True:
            i = 10
    model = loader.load_model(Filename(temp_model))
    assert model
    assert isinstance(model, NodePath)
    assert model.name == 'model'

def test_load_model_str(loader, temp_model):
    if False:
        for i in range(10):
            print('nop')
    model = loader.load_model(str(temp_model))
    assert model
    assert isinstance(model, NodePath)
    assert model.name == 'model'

def test_load_model_list(loader, temp_model):
    if False:
        print('Hello World!')
    models = loader.load_model([temp_model, temp_model])
    assert models
    assert isinstance(models, list)
    assert len(models) == 2
    assert isinstance(models[0], NodePath)
    assert isinstance(models[1], NodePath)

def test_load_model_tuple(loader, temp_model):
    if False:
        i = 10
        return i + 15
    models = loader.load_model((temp_model, temp_model))
    assert models
    assert isinstance(models, list)
    assert len(models) == 2
    assert isinstance(models[0], NodePath)
    assert isinstance(models[1], NodePath)

def test_load_model_set(loader, temp_model):
    if False:
        i = 10
        return i + 15
    models = loader.load_model({temp_model})
    assert models
    assert isinstance(models, list)
    assert len(models) == 1
    assert isinstance(models[0], NodePath)

def test_load_model_missing(loader):
    if False:
        i = 10
        return i + 15
    with pytest.raises(IOError):
        loader.load_model('/nonexistent.bam')

def test_load_model_okmissing(loader):
    if False:
        i = 10
        return i + 15
    model = loader.load_model('/nonexistent.bam', okMissing=True)
    assert model is None

def test_loader_entry_points(tmp_path):
    if False:
        while True:
            i = 10
    (tmp_path / 'fnargle.py').write_text('\nfrom panda3d.core import ModelRoot\nimport sys\n\nsys._fnargle_loaded = True\n\nclass FnargleLoader:\n    name = "Fnargle"\n    extensions = [\'fnrgl\']\n    supports_compressed = False\n\n    @staticmethod\n    def load_file(path, options, record=None):\n        return ModelRoot("fnargle")\n')
    (tmp_path / 'fnargle.dist-info').mkdir()
    (tmp_path / 'fnargle.dist-info' / 'METADATA').write_text('\nMetadata-Version: 2.0\nName: fnargle\nVersion: 1.0.0\n')
    (tmp_path / 'fnargle.dist-info' / 'entry_points.txt').write_text('\n[panda3d.loaders]\nfnrgl = fnargle:FnargleLoader\n')
    model_path = tmp_path / 'test.fnrgl'
    model_path.write_text('')
    if sys.version_info >= (3, 11):
        import sysconfig
        stdlib = sysconfig.get_path('stdlib')
        platstdlib = sysconfig.get_path('platstdlib')
    else:
        from distutils import sysconfig
        stdlib = sysconfig.get_python_lib(False, True)
        platstdlib = sysconfig.get_python_lib(True, True)
    registry = LoaderFileTypeRegistry.get_global_ptr()
    prev_loaded = Loader._loadedPythonFileTypes
    prev_path = sys.path
    file_type = None
    try:
        sys.path = [str(tmp_path), platstdlib, stdlib]
        Loader._loadedPythonFileTypes = False
        loader = Loader(None)
        assert Loader._loadedPythonFileTypes
        file_type = registry.get_type_from_extension('fnrgl')
        assert file_type is not None
        assert not hasattr(sys, '_fnargle_loaded')
        assert file_type.supports_load()
        assert not file_type.supports_save()
        assert not file_type.supports_compressed()
        assert file_type.get_extension() == 'fnrgl'
        assert sys._fnargle_loaded
        assert 'fnargle' in sys.modules
        model_fn = Filename(model_path)
        model_fn.make_true_case()
        model = loader.load_model(model_fn, noCache=True)
        assert model is not None
        assert model.name == 'fnargle'
    finally:
        Loader._loadedPythonFileTypes = prev_loaded
        sys.path = prev_path
        if hasattr(sys, '_fnargle_loaded'):
            del sys._fnargle_loaded
        if 'fnargle' in sys.modules:
            del sys.modules['fnargle']
        if file_type is not None:
            registry.unregister_type(file_type)