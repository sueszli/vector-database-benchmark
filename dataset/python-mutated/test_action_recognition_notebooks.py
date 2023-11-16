import papermill as pm
import pytest
import scrapbook as sb
from utils_cv.common.data import unzip_url
from utils_cv.action_recognition.data import Urls
KERNEL_NAME = 'python3'
OUTPUT_NOTEBOOK = 'output.ipynb'

@pytest.mark.notebooks
def test_00_notebook_run(action_recognition_notebooks):
    if False:
        i = 10
        return i + 15
    notebook_path = action_recognition_notebooks['00']
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, parameters=dict(PM_VERSION=pm.__version__, sample_video_url=Urls.webcam_vid_low_res), kernel_name=KERNEL_NAME)
    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)

@pytest.mark.notebooks
def test_01_notebook_run(action_recognition_notebooks, ar_milk_bottle_path):
    if False:
        i = 10
        return i + 15
    notebook_path = action_recognition_notebooks['01']
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, parameters=dict(PM_VERSION=pm.__version__, DATA_PATH=ar_milk_bottle_path, MODEL_INPUT_SIZE=8, EPOCHS=1, BATCH_SIZE=8, LR=0.001), kernel_name=KERNEL_NAME)
    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert isinstance(nb_output.scraps['vid_pred_accuracy'].data, float)
    assert isinstance(nb_output.scraps['clip_pred_accuracy'].data, float)

@pytest.mark.notebooks
def test_02_notebook_run(action_recognition_notebooks):
    if False:
        return 10
    pass

@pytest.mark.notebooks
def test_10_notebook_run(action_recognition_notebooks):
    if False:
        i = 10
        return i + 15
    notebook_path = action_recognition_notebooks['10']
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, parameters=dict(PM_VERSION=pm.__version__), kernel_name=KERNEL_NAME)
    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)