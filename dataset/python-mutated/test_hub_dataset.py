import pytest
from tests.base import TestBase
from tests.utils import is_package_installed

class TestHubDatasetIntegration(TestBase):

    @pytest.mark.skipif(not is_package_installed('hub'), reason="'hub' is not installed. skipping.")
    def test_dataset_as_run_param(self):
        if False:
            return 10
        import hub
        from aim.sdk.objects.plugins.hub_dataset import HubDataset
        from aim.sdk import Run
        ds = hub.dataset('hub://activeloop/cifar100-test')
        run = Run(system_tracking_interval=None)
        run['hub_ds'] = HubDataset(ds)
        ds_object = run['hub_ds']
        ds_dict = run.get('hub_ds', resolve_objects=True)
        self.assertTrue(isinstance(ds_object, HubDataset))
        self.assertTrue(isinstance(ds_dict, dict))
        self.assertIn('meta', ds_dict['dataset'].keys())
        self.assertIn('source', ds_dict['dataset'].keys())