import pytest
from unittest import TestCase
import torch
from bigdl.nano.pytorch import Trainer
import bigdl.nano.automl.hpo.space as space
from bigdl.nano.automl.pytorch import HPOSearcher
import bigdl.nano.automl.hpo as hpo
from _helper import BoringModel, RandomDataset

class TestHPOSearcher(TestCase):

    def test_simple_model(self):
        if False:
            return 10

        @hpo.plmodel()
        class CustomModel(BoringModel):

            def __init__(self, out_dim1, out_dim2, dropout_1, dropout_2):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                layers = []
                input_dim = 32
                for (out_dim, dropout) in [(out_dim1, dropout_1), (out_dim2, dropout_2)]:
                    layers.append(torch.nn.Linear(input_dim, out_dim))
                    layers.append(torch.nn.Tanh())
                    layers.append(torch.nn.Dropout(dropout))
                    input_dim = out_dim
                layers.append(torch.nn.Linear(input_dim, 2))
                self.layers: torch.nn.Module = torch.nn.Sequential(*layers)
        model = CustomModel(out_dim1=space.Categorical(16, 32), out_dim2=space.Categorical(16, 32), dropout_1=space.Real(0.1, 0.5), dropout_2=0.2)
        trainer = Trainer(logger=True, checkpoint_callback=False, max_epochs=3)
        searcher = HPOSearcher(trainer)
        searcher.search(model, target_metric='val_loss', direction='minimize', n_trials=3, max_epochs=3)
        study = searcher.search_summary()
        assert study
        assert study.best_trial

    def test_simple_model_multi_processes(self):
        if False:
            while True:
                i = 10

        @hpo.plmodel()
        class CustomModel(BoringModel):

            def __init__(self, out_dim1, out_dim2, dropout_1, dropout_2):
                if False:
                    return 10
                super().__init__()
                layers = []
                input_dim = 32
                for (out_dim, dropout) in [(out_dim1, dropout_1), (out_dim2, dropout_2)]:
                    layers.append(torch.nn.Linear(input_dim, out_dim))
                    layers.append(torch.nn.Tanh())
                    layers.append(torch.nn.Dropout(dropout))
                    input_dim = out_dim
                layers.append(torch.nn.Linear(input_dim, 2))
                self.layers: torch.nn.Module = torch.nn.Sequential(*layers)
        model = CustomModel(out_dim1=space.Categorical(16, 32), out_dim2=space.Categorical(16, 32), dropout_1=space.Real(0.1, 0.5), dropout_2=0.2)
        trainer = Trainer(logger=True, checkpoint_callback=True, max_epochs=3, num_processes=2)
        searcher = HPOSearcher(trainer, num_processes=2)
        searcher.search(model, target_metric='val_loss', direction='minimize', n_trials=3, max_epochs=3)
        study = searcher.search_summary()
        assert study
        assert study.best_trial
if __name__ == '__main__':
    pytest.main([__file__])