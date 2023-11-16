import pytest
import deeplake
from deeplake.constants import HUB_CLOUD_DEV_USERNAME
enabled_datasets = pytest.mark.parametrize('ds', ['memory_ds', 'local_ds', 's3_ds', 'gcs_ds', 'azure_ds', 'gdrive_ds'], indirect=True)
enabled_non_gdrive_datasets = pytest.mark.parametrize('ds', ['memory_ds', 'local_ds', 's3_ds', 'gcs_ds', 'azure_ds'], indirect=True)
enabled_non_gcs_datasets = pytest.mark.parametrize('ds', ['memory_ds', 'local_ds', 's3_ds', 'azure_ds', 'gdrive_ds'], indirect=True)
enabled_non_gcs_gdrive_datasets = pytest.mark.parametrize('ds', ['memory_ds', 'local_ds', 's3_ds', 'azure_ds'], indirect=True)
enabled_persistent_dataset_generators = pytest.mark.parametrize('ds_generator', ['local_ds_generator', 's3_ds_generator', 'gcs_ds_generator', 'azure_ds_generator', 'gdrive_ds_generator'], indirect=True)
enabled_persistent_non_gdrive_dataset_generators = pytest.mark.parametrize('ds_generator', ['local_ds_generator', 's3_ds_generator', 'gcs_ds_generator', 'azure_ds_generator'], indirect=True)
enabled_cloud_dataset_generators = pytest.mark.parametrize('ds_generator', ['s3_ds_generator', 'gcs_ds_generator', 'azure_ds_generator'], indirect=True)

@pytest.fixture
def memory_ds(memory_path):
    if False:
        return 10
    return deeplake.dataset(memory_path)

@pytest.fixture
def local_ds(local_ds_generator):
    if False:
        while True:
            i = 10
    return local_ds_generator()

@pytest.fixture
def local_auth_ds(local_auth_ds_generator):
    if False:
        i = 10
        return i + 15
    return local_auth_ds_generator()

@pytest.fixture
def local_ds_generator(local_path):
    if False:
        i = 10
        return i + 15

    def generate_local_ds(**kwargs):
        if False:
            return 10
        return deeplake.dataset(local_path, **kwargs)
    return generate_local_ds

@pytest.fixture
def local_auth_ds_generator(local_path, hub_cloud_dev_token):
    if False:
        print('Hello World!')

    def generate_local_auth_ds(**kwargs):
        if False:
            while True:
                i = 10
        return deeplake.dataset(local_path, token=hub_cloud_dev_token, **kwargs)
    return generate_local_auth_ds

@pytest.fixture
def s3_ds(s3_ds_generator):
    if False:
        print('Hello World!')
    return s3_ds_generator()

@pytest.fixture
def s3_ds_generator(s3_path):
    if False:
        while True:
            i = 10

    def generate_s3_ds(**kwargs):
        if False:
            i = 10
            return i + 15
        return deeplake.dataset(s3_path, **kwargs)
    return generate_s3_ds

@pytest.fixture
def gdrive_ds(gdrive_ds_generator):
    if False:
        print('Hello World!')
    return gdrive_ds_generator()

@pytest.fixture
def gdrive_ds_generator(gdrive_path, gdrive_creds):
    if False:
        return 10

    def generate_gdrive_ds(**kwargs):
        if False:
            i = 10
            return i + 15
        return deeplake.dataset(gdrive_path, creds=gdrive_creds, **kwargs)
    return generate_gdrive_ds

@pytest.fixture
def gcs_ds(gcs_ds_generator):
    if False:
        print('Hello World!')
    return gcs_ds_generator()

@pytest.fixture
def gcs_ds_generator(gcs_path, gcs_creds):
    if False:
        while True:
            i = 10

    def generate_gcs_ds(**kwargs):
        if False:
            i = 10
            return i + 15
        return deeplake.dataset(gcs_path, creds=gcs_creds, **kwargs)
    return generate_gcs_ds

@pytest.fixture
def azure_ds(azure_ds_generator):
    if False:
        i = 10
        return i + 15
    return azure_ds_generator()

@pytest.fixture
def azure_ds_generator(azure_path):
    if False:
        while True:
            i = 10

    def generate_azure_ds(**kwargs):
        if False:
            return 10
        return deeplake.dataset(azure_path, **kwargs)
    return generate_azure_ds

@pytest.fixture
def hub_cloud_ds(hub_cloud_ds_generator):
    if False:
        return 10
    return hub_cloud_ds_generator()

@pytest.fixture
def hub_cloud_ds_generator(hub_cloud_path, hub_cloud_dev_token):
    if False:
        return 10

    def generate_hub_cloud_ds(**kwargs):
        if False:
            print('Hello World!')
        return deeplake.dataset(hub_cloud_path, token=hub_cloud_dev_token, **kwargs)
    return generate_hub_cloud_ds

@pytest.fixture
def hub_cloud_gcs_ds_generator(gcs_path, gcs_creds, hub_cloud_dev_token):
    if False:
        while True:
            i = 10

    def generate_hub_cloud_gcs_ds(**kwargs):
        if False:
            i = 10
            return i + 15
        ds = deeplake.dataset(gcs_path, creds=gcs_creds, **kwargs)
        ds.connect(org_id=HUB_CLOUD_DEV_USERNAME, token=hub_cloud_dev_token, creds_key='gcp_creds')
        return ds
    return generate_hub_cloud_gcs_ds

@pytest.fixture
def hub_cloud_gcs_ds(hub_cloud_gcs_ds_generator):
    if False:
        while True:
            i = 10
    return hub_cloud_gcs_ds_generator()

@pytest.fixture
def ds(request):
    if False:
        i = 10
        return i + 15
    'Used with parametrize to use all enabled dataset fixtures.'
    return request.getfixturevalue(request.param)

@pytest.fixture
def ds_generator(request):
    if False:
        i = 10
        return i + 15
    'Used with parametrize to use all enabled persistent dataset generator fixtures.'
    return request.getfixturevalue(request.param)

@pytest.fixture
def runtime():
    if False:
        i = 10
        return i + 15
    return {'tensor_db': True}