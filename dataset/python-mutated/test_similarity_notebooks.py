import papermill as pm
import pytest
import scrapbook as sb
KERNEL_NAME = 'python3'
OUTPUT_NOTEBOOK = 'output.ipynb'

@pytest.mark.notebooks
def test_00_notebook_run(similarity_notebooks):
    if False:
        while True:
            i = 10
    notebook_path = similarity_notebooks['00']
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)
    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert len(nb_output.scraps['query_feature'].data) == 512
    assert min(nb_output.scraps['query_feature'].data) >= 0
    assert min([dist for (path, dist) in nb_output.scraps['distances'].data]) < 0.001

@pytest.mark.notebooks
def test_01_notebook_run(similarity_notebooks, tiny_ic_data_path):
    if False:
        print('Hello World!')
    notebook_path = similarity_notebooks['01']
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, parameters=dict(PM_VERSION=pm.__version__, DATA_PATH=tiny_ic_data_path, EPOCHS_HEAD=1, EPOCHS_BODY=1, IM_SIZE=50), kernel_name=KERNEL_NAME)

@pytest.mark.notebooks
def test_02_notebook_run(similarity_notebooks, tiny_is_data_path):
    if False:
        return 10
    notebook_path = similarity_notebooks['02']
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, parameters=dict(PM_VERSION=pm.__version__, DATA_PATH=tiny_is_data_path, EPOCHS_HEAD=1, EPOCHS_BODY=1, BATCH_SIZE=1, IM_SIZE=50), kernel_name=KERNEL_NAME)

@pytest.mark.notebooks
def test_11_notebook_run(similarity_notebooks, tiny_ic_data_path):
    if False:
        for i in range(10):
            print('nop')
    notebook_path = similarity_notebooks['11']
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, parameters=dict(PM_VERSION=pm.__version__, DATA_PATHS=[tiny_ic_data_path], REPS=1, LEARNING_RATES=[0.0001], IM_SIZES=[30], EPOCHS=[1]), kernel_name=KERNEL_NAME)

@pytest.mark.notebooks
def test_12_notebook_run(similarity_notebooks, tiny_ic_data_path):
    if False:
        while True:
            i = 10
    notebook_path = similarity_notebooks['12']
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, parameters=dict(PM_VERSION=pm.__version__, DATA_PATH=tiny_ic_data_path, IM_SIZE=30, NUM_RANK_ITER=5), kernel_name=KERNEL_NAME)