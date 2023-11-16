from typing import Any
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine

def list_documents_sample(project_id: str, location: str, data_store_id: str) -> Any:
    if False:
        return 10
    client_options = ClientOptions(api_endpoint=f'{location}-discoveryengine.googleapis.com') if location != 'global' else None
    client = discoveryengine.DocumentServiceClient(client_options=client_options)
    parent = client.branch_path(project=project_id, location=location, data_store=data_store_id, branch='default_branch')
    response = client.list_documents(parent=parent)
    print(f'Documents in {data_store_id}:')
    for result in response:
        print(result)
    return response