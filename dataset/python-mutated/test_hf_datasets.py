import pytest
from tests.base import TestBase
from tests.utils import is_package_installed

class TestHFDatasetsIntegration(TestBase):

    @pytest.mark.skipif(not is_package_installed('datasets'), reason="'datasets' is not installed. skipping.")
    def test_datasets_as_run_param(self):
        if False:
            for i in range(10):
                print('nop')
        from datasets import load_dataset
        from aim.sdk.objects.plugins.hf_datasets_metadata import HFDataset
        from aim.sdk import Run
        dataset = load_dataset('rotten_tomatoes')
        run = Run(repo='.hf_datasets', system_tracking_interval=None)
        run['datasets_info'] = HFDataset(dataset)
        ds_object = run['datasets_info']
        ds_dict = run.get('datasets_info', resolve_objects=True)
        self.assertTrue(isinstance(ds_object, HFDataset))
        self.assertTrue(isinstance(ds_dict, dict))
        self.assertIn('meta', ds_dict['dataset'].keys())
        self.assertIn('source', ds_dict['dataset'].keys())