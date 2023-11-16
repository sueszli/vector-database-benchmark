from typing import Dict

def deidentify_dataset(project_id: str, location: str, dataset_id: str, destination_dataset_id: str) -> Dict[str, str]:
    if False:
        for i in range(10):
            print('nop')
    "Uses a DICOM tag keeplist to create a new dataset containing de-identified DICOM data from the source dataset.\n\n    See\n    https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/datasets\n    before running the sample.\n    See https://googleapis.github.io/google-api-python-client/docs/dyn/healthcare_v1.projects.locations.datasets.html#deidentify\n    for the Python API reference.\n\n    Args:\n      project_id: The project ID or project number of the Google Cloud project you want\n          to use.\n      location: The name of the dataset's location.\n      dataset_id: The ID of the source dataset containing the DICOM store to de-identify.\n      destination_dataset_id: The ID of the dataset where de-identified DICOM data\n        is written.\n\n    Returns:\n      A dictionary representing a long-running operation that results from\n      calling the 'DeidentifyDataset' method. Use the\n      'google.longrunning.Operation'\n      API to poll the operation status.\n    "
    import time
    from googleapiclient import discovery
    from googleapiclient.errors import HttpError
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    source_dataset = 'projects/{}/locations/{}/datasets/{}'.format(project_id, location, dataset_id)
    destination_dataset = 'projects/{}/locations/{}/datasets/{}'.format(project_id, location, destination_dataset_id)
    body = {'destinationDataset': destination_dataset, 'config': {'dicom': {'keepList': {'tags': ['Columns', 'NumberOfFrames', 'PixelRepresentation', 'MediaStorageSOPClassUID', 'MediaStorageSOPInstanceUID', 'Rows', 'SamplesPerPixel', 'BitsAllocated', 'HighBit', 'PhotometricInterpretation', 'BitsStored', 'PatientID', 'TransferSyntaxUID', 'SOPInstanceUID', 'StudyInstanceUID', 'SeriesInstanceUID', 'PixelData']}}}}
    request = client.projects().locations().datasets().deidentify(sourceDataset=source_dataset, body=body)
    start_time = time.time()
    max_time = 600
    try:
        operation = request.execute()
        while not operation.get('done', False):
            print('Waiting for operation to finish...')
            if time.time() - start_time > max_time:
                raise RuntimeError('Timed out waiting for operation to finish.')
            operation = client.projects().locations().datasets().operations().get(name=operation['name']).execute()
            time.sleep(5)
        if operation.get('error'):
            raise TimeoutError(f"De-identify operation failed: {operation['error']}")
        else:
            print(f'De-identified data to dataset: {destination_dataset_id}')
            print(f"Resources succeeded: {operation.get('metadata').get('counter').get('success')}")
            print(f"Resources failed: {operation.get('metadata').get('counter').get('failure')}")
            return operation
    except HttpError as err:
        if err.resp.status == 409:
            raise RuntimeError(f'Destination dataset with ID {destination_dataset_id} already exists.')
        else:
            raise err