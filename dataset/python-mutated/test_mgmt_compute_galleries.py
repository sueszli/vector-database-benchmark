import os
import unittest
import pytest
import azure.mgmt.compute
from devtools_testutils import AzureMgmtRecordedTestCase, RandomNameResourceGroupPreparer, recorded_by_proxy
AZURE_LOCATION = 'eastus2'

class TestMgmtCompute(AzureMgmtRecordedTestCase):

    def setup_method(self, method):
        if False:
            while True:
                i = 10
        self.mgmt_client = self.create_mgmt_client(azure.mgmt.compute.ComputeManagementClient)
        if self.is_live:
            from azure.mgmt.network import NetworkManagementClient
            self.network_client = self.create_mgmt_client(NetworkManagementClient)

    def create_snapshot(self, group_name, disk_name, snapshot_name):
        if False:
            while True:
                i = 10
        BODY = {'location': AZURE_LOCATION, 'creation_data': {'create_option': 'Empty'}, 'disk_size_gb': '200'}
        result = self.mgmt_client.disks.begin_create_or_update(group_name, disk_name, BODY)
        disk = result.result()
        BODY = {'location': AZURE_LOCATION, 'creation_data': {'create_option': 'Copy', 'source_uri': disk.id}}
        result = self.mgmt_client.snapshots.begin_create_or_update(group_name, snapshot_name, BODY)
        result = result.result()

    def delete_snapshot(self, group_name, snapshot_name):
        if False:
            for i in range(10):
                print('nop')
        result = self.mgmt_client.snapshots.begin_revoke_access(group_name, snapshot_name)
        result = result.result()
        result = self.mgmt_client.snapshots.begin_delete(group_name, snapshot_name)
        result = result.result()

    @pytest.mark.skipif(os.getenv('AZURE_TEST_RUN_LIVE') not in ('true', 'yes'), reason='only run live test')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_compute_galleries(self, resource_group):
        if False:
            i = 10
            return i + 15
        SUBSCRIPTION_ID = self.get_settings_value('SUBSCRIPTION_ID')
        RESOURCE_GROUP = resource_group.name
        GALLERY_NAME = self.get_resource_name('galleryname')
        APPLICATION_NAME = self.get_resource_name('applicationname')
        IMAGE_NAME = self.get_resource_name('imagex')
        DISK_NAME = self.get_resource_name('diskname')
        SNAPSHOT_NAME = self.get_resource_name('snapshotname')
        VERSION_NAME = '1.0.0'
        if self.is_live:
            self.create_snapshot(RESOURCE_GROUP, DISK_NAME, SNAPSHOT_NAME)
        BODY = {'location': AZURE_LOCATION, 'description': 'This is the gallery description.'}
        result = self.mgmt_client.galleries.begin_create_or_update(resource_group.name, GALLERY_NAME, BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'description': 'This is the gallery application description.', 'eula': 'This is the gallery application EULA.', 'supported_os_type': 'Windows'}
        result = self.mgmt_client.gallery_applications.begin_create_or_update(resource_group.name, GALLERY_NAME, APPLICATION_NAME, BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'os_type': 'Windows', 'os_state': 'Generalized', 'hyper_v_generation': 'V1', 'identifier': {'publisher': 'myPublisherName', 'offer': 'myOfferName', 'sku': 'mySkuName'}}
        result = self.mgmt_client.gallery_images.begin_create_or_update(resource_group.name, GALLERY_NAME, IMAGE_NAME, BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'publishing_profile': {'target_regions': [{'name': AZURE_LOCATION, 'regional_replica_count': '2', 'storage_account_type': 'Standard_ZRS'}]}, 'storage_profile': {'os_disk_image': {'source': {'id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.Compute/snapshots/' + SNAPSHOT_NAME + ''}, 'host_caching': 'ReadOnly'}}}
        result = self.mgmt_client.gallery_image_versions.begin_create_or_update(resource_group.name, GALLERY_NAME, IMAGE_NAME, VERSION_NAME, BODY)
        result = result.result()
        result = self.mgmt_client.gallery_image_versions.get(resource_group.name, GALLERY_NAME, IMAGE_NAME, VERSION_NAME)
        result = self.mgmt_client.gallery_images.get(resource_group.name, GALLERY_NAME, IMAGE_NAME)
        result = self.mgmt_client.gallery_applications.get(resource_group.name, GALLERY_NAME, APPLICATION_NAME)
        result = self.mgmt_client.galleries.get(resource_group.name, GALLERY_NAME)
        result = self.mgmt_client.gallery_image_versions.list_by_gallery_image(resource_group.name, GALLERY_NAME, IMAGE_NAME)
        result = self.mgmt_client.gallery_images.list_by_gallery(resource_group.name, GALLERY_NAME)
        result = self.mgmt_client.gallery_applications.list_by_gallery(resource_group.name, GALLERY_NAME)
        result = self.mgmt_client.galleries.list_by_resource_group(resource_group.name)
        result = self.mgmt_client.galleries.list()
        BODY = {'publishing_profile': {'target_regions': [{'name': AZURE_LOCATION, 'regional_replica_count': '2', 'storage_account_type': 'Standard_ZRS'}]}, 'storage_profile': {'os_disk_image': {'source': {'id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.Compute/snapshots/' + SNAPSHOT_NAME + ''}, 'host_caching': 'ReadOnly'}}}
        result = self.mgmt_client.gallery_image_versions.begin_update(resource_group.name, GALLERY_NAME, IMAGE_NAME, VERSION_NAME, BODY)
        result = result.result()
        BODY = {'os_type': 'Windows', 'os_state': 'Generalized', 'hyper_v_generation': 'V1', 'identifier': {'publisher': 'myPublisherName', 'offer': 'myOfferName', 'sku': 'mySkuName'}}
        result = self.mgmt_client.gallery_images.begin_update(resource_group.name, GALLERY_NAME, IMAGE_NAME, BODY)
        result = result.result()
        BODY = {'description': 'This is the gallery application description.', 'eula': 'This is the gallery application EULA.', 'supported_os_type': 'Windows', 'tags': {'tag1': 'tag1'}}
        result = self.mgmt_client.gallery_applications.begin_update(resource_group.name, GALLERY_NAME, APPLICATION_NAME, BODY)
        result = result.result()
        BODY = {'description': 'This is the gallery description.'}
        result = self.mgmt_client.galleries.begin_update(resource_group.name, GALLERY_NAME, BODY)
        result = result.result()
        result = self.mgmt_client.gallery_image_versions.begin_delete(resource_group.name, GALLERY_NAME, IMAGE_NAME, VERSION_NAME)
        result = result.result()
        result = self.mgmt_client.gallery_applications.begin_delete(resource_group.name, GALLERY_NAME, APPLICATION_NAME)
        result = result.result()
        if self.is_live:
            self.delete_snapshot(RESOURCE_GROUP, SNAPSHOT_NAME)
        result = self.mgmt_client.gallery_images.begin_delete(resource_group.name, GALLERY_NAME, IMAGE_NAME)
        result = result.result()
        result = self.mgmt_client.galleries.begin_delete(resource_group.name, GALLERY_NAME)
        result = result.result()