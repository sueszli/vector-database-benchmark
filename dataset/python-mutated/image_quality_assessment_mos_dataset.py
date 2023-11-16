from modelscope.metainfo import Models
from modelscope.msdatasets.dataset_cls.custom_datasets import CUSTOM_DATASETS, TorchCustomDataset
from modelscope.preprocessors.cv import ImageQualityAssessmentMosPreprocessor
from modelscope.utils.constant import Tasks

@CUSTOM_DATASETS.register_module(Tasks.image_quality_assessment_mos, module_name=Models.image_quality_assessment_mos)
class ImageQualityAssessmentMosDataset(TorchCustomDataset):
    """Paired image dataset for image quality assessment mos.
    """

    def __init__(self, dataset, opt, preprocessor=ImageQualityAssessmentMosPreprocessor()):
        if False:
            print('Hello World!')
        self.preprocessor = preprocessor
        self.dataset = dataset
        self.opt = opt
        self.scale = 0.2

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.dataset)

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        item_dict = self.dataset[index]
        iterm_mos = float(item_dict['mos']) * self.scale
        img = self.preprocessor(item_dict['image:FILE'])
        return {'input': img['input'].squeeze(0), 'target': iterm_mos}