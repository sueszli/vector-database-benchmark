import glob
import os
import shutil
import uuid
from fuzzydata.clients.modin import ModinWorkflow
from fuzzydata.core.generator import generate_workflow
from modin.config import Engine

def test_fuzzydata_sample_workflow():
    if False:
        print('Hello World!')
    wf_name = str(uuid.uuid4())[:8]
    num_versions = 10
    cols = 33
    rows = 1000
    bfactor = 1.0
    exclude_ops = ['groupby']
    matfreq = 2
    engine = Engine.get().lower()
    base_out_directory = f'/tmp/fuzzydata-test-wf-{engine}/'
    if os.path.exists(base_out_directory):
        shutil.rmtree(base_out_directory)
    output_directory = f'{base_out_directory}/{wf_name}/'
    os.makedirs(output_directory, exist_ok=True)
    workflow = generate_workflow(workflow_class=ModinWorkflow, name=wf_name, num_versions=num_versions, base_shape=(cols, rows), out_directory=output_directory, bfactor=bfactor, exclude_ops=exclude_ops, matfreq=matfreq, wf_options={'modin_engine': engine})
    assert len(workflow) == num_versions
    assert len(list(glob.glob(f'{output_directory}/artifacts/*.csv'))) == len(workflow.artifact_dict)
    assert os.path.exists(f'{output_directory}/{workflow.name}_operations.json')
    assert os.path.getsize(f'{output_directory}/{workflow.name}_operations.json') > 0
    assert os.path.exists(f'{output_directory}/{workflow.name}_gt_graph.csv')