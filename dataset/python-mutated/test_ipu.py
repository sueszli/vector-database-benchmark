from importlib import import_module
import pytest

@pytest.mark.parametrize(('import_path', 'name'), [('lightning.pytorch.accelerators', 'IPUAccelerator'), ('lightning.pytorch.accelerators.ipu', 'IPUAccelerator'), ('lightning.pytorch.strategies', 'IPUStrategy'), ('lightning.pytorch.strategies.ipu', 'IPUStrategy'), ('lightning.pytorch.plugins.precision', 'IPUPrecisionPlugin'), ('lightning.pytorch.plugins.precision.ipu', 'IPUPrecisionPlugin')])
def test_extracted_ipu(import_path, name):
    if False:
        i = 10
        return i + 15
    module = import_module(import_path)
    cls = getattr(module, name)
    with pytest.raises(NotImplementedError, match=f'{name}` class has been moved to an external package.*'):
        cls()