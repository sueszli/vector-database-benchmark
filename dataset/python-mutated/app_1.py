import flash
from flash.core.data.utils import download_data
from flash.image import ImageClassificationData, ImageClassifier
from lightning.app import LightningWork, LightningFlow, LightningApp, CloudCompute
from lightning.pytorch.callbacks import ModelCheckpoint

class ImageClassifierTrainWork(LightningWork):

    def __init__(self, max_epochs: int, backbone: str, cloud_compute: CloudCompute):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parallel=True, cloud_compute=cloud_compute)
        self.max_epochs = max_epochs
        self.backbone = backbone
        self.best_model_path = None
        self.best_model_score = None

    def run(self, train_folder):
        if False:
            print('Hello World!')
        datamodule = ImageClassificationData.from_folders(train_folder=train_folder, batch_size=1, val_split=0.5)
        model = ImageClassifier(datamodule.num_classes, backbone=self.backbone)
        trainer = flash.Trainer(max_epochs=self.max_epochs, limit_train_batches=1, limit_val_batches=4, callbacks=[ModelCheckpoint(monitor='val_cross_entropy')])
        trainer.fit(model, datamodule=datamodule)
        self.best_model_path = trainer.checkpoint_callback.best_model_path
        self.best_model_score = trainer.checkpoint_callback.best_model_score.item()

class ImageClassifierServeWork(LightningWork):

    def run(self, best_model_path: str):
        if False:
            while True:
                i = 10
        model = ImageClassifier.load_from_checkpoint(best_model_path)
        model.serve(output='labels')

class RootFlow(LightningFlow):

    def __init__(self, max_epochs: int, data_dir: str):
        if False:
            print('Hello World!')
        super().__init__()
        self.data_dir = data_dir
        self.train_work_1 = ImageClassifierTrainWork(max_epochs, 'resnet18')
        self.train_work_2 = ImageClassifierTrainWork(max_epochs, 'resnet26')
        self.server_work = ImageClassifierServeWork()

    def run(self):
        if False:
            while True:
                i = 10
        self.train_work_1.run(self.data_dir)
        self.train_work_2.run(self.data_dir)
        if self.train_work_1.best_model_score and self.train_work_2.best_model_score:
            self.server_work.run(self.train_work_1.best_model_path if self.train_work_1.best_model_score < self.train_work_2.best_model_score else self.train_work_2.best_model_path)
download_data('https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip', './data')
app = LightningApp(RootFlow(5, './data/hymenoptera_data'))