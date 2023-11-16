import os
from google.api_core.exceptions import ResourceExhausted
from google.cloud import enterpriseknowledgegraph as ekg
import create_entity_reconciliation_job_sample
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
location = 'global'
input_dataset = 'ekg_entity_reconciliation'
input_table = 'patients'
mapping_file_uri = 'gs://cloud-samples-data/ekg/quickstart/test_mapping1.yml'
entity_type = ekg.InputConfig.EntityType.PERSON
output_dataset = 'ekg_entity_reconciliation'

def test_create_entity_reconciliation_job(capsys):
    if False:
        while True:
            i = 10
    try:
        create_entity_reconciliation_job_sample.create_entity_reconciliation_job_sample(project_id=project_id, location=location, input_dataset=input_dataset, input_table=input_table, mapping_file_uri=mapping_file_uri, entity_type=entity_type, output_dataset=output_dataset)
    except ResourceExhausted as e:
        print(e.message)
    (out, _) = capsys.readouterr()
    assert 'Job: projects/' in out or 'Resource Exhausted' in out