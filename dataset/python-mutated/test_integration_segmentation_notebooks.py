import papermill as pm
import pytest
import numpy as np
import scrapbook as sb
KERNEL_NAME = 'python3'
OUTPUT_NOTEBOOK = 'output.ipynb'

@pytest.mark.notebooks
@pytest.mark.linuxgpu
def test_11_notebook_integration_run(segmentation_notebooks):
    if False:
        print('Hello World!')
    notebook_path = segmentation_notebooks['11']
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, parameters=dict(PM_VERSION=pm.__version__, REPS=1), kernel_name=KERNEL_NAME)
    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    nr_elements = nb_output.scraps['nr_elements'].data
    ratio_correct = nb_output.scraps['ratio_correct'].data
    max_duration = nb_output.scraps['max_duration'].data
    min_duration = nb_output.scraps['min_duration'].data
    assert nr_elements == 12
    assert min_duration <= 0.8 * max_duration
    assert np.max(ratio_correct) > 0.75