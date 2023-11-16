import pytest
import papermill as pm
import scrapbook as sb
from pathlib import Path
from recommenders.utils.notebook_utils import is_jupyter, is_databricks

@pytest.mark.notebooks
def test_is_jupyter(output_notebook, kernel_name):
    if False:
        for i in range(10):
            print('nop')
    assert is_jupyter() is False
    assert is_databricks() is False
    path = Path(__file__).absolute().parent.joinpath('test_notebook_utils.ipynb')
    pm.execute_notebook(path, output_notebook, kernel_name=kernel_name)
    nb = sb.read_notebook(output_notebook)
    df = nb.scraps.dataframe
    result_is_jupyter = df.loc[df['name'] == 'is_jupyter', 'data'].values[0]
    assert result_is_jupyter
    result_is_databricks = df.loc[df['name'] == 'is_databricks', 'data'].values[0]
    assert not result_is_databricks

@pytest.mark.spark
@pytest.mark.notebooks
@pytest.mark.skip(reason='TODO: Implement this')
def test_is_databricks():
    if False:
        for i in range(10):
            print('nop')
    pass