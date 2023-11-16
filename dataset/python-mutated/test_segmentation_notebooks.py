import papermill as pm
import pytest
import scrapbook as sb
KERNEL_NAME = 'python3'
OUTPUT_NOTEBOOK = 'output.ipynb'

@pytest.mark.notebooks
def test_11_notebook_run(segmentation_notebooks, tiny_seg_data_path):
    if False:
        print('Hello World!')
    notebook_path = segmentation_notebooks['11']
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, parameters=dict(PM_VERSION=pm.__version__, REPS=1, EPOCHS=[1], IM_SIZE=[50], LEARNING_RATES=[0.0001], DATA_PATH=[tiny_seg_data_path]), kernel_name=KERNEL_NAME)
    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    nr_elements = nb_output.scraps['nr_elements'].data
    ratio_correct = nb_output.scraps['ratio_correct'].data
    max_duration = nb_output.scraps['max_duration'].data
    min_duration = nb_output.scraps['min_duration'].data
    assert nr_elements == 2