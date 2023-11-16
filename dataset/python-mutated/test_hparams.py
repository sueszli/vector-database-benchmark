import copy
import functools
import os
import pickle
from argparse import Namespace
from dataclasses import dataclass
from enum import Enum
from unittest import mock
import cloudpickle
import pytest
import torch
from fsspec.implementations.local import LocalFileSystem
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.core.datamodule import LightningDataModule
from lightning.pytorch.core.mixins import HyperparametersMixin
from lightning.pytorch.core.saving import load_hparams_from_yaml, save_hparams_to_yaml
from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel, RandomDataset
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.utilities import AttributeDict, is_picklable
from lightning.pytorch.utilities.imports import _OMEGACONF_AVAILABLE
from lightning_utilities.core.imports import RequirementCache
from lightning_utilities.test.warning import no_warning_call
from torch.utils.data import DataLoader
from tests_pytorch.helpers.runif import RunIf
if _OMEGACONF_AVAILABLE:
    from omegaconf import Container, OmegaConf
    from omegaconf.dictconfig import DictConfig

class SaveHparamsModel(BoringModel):
    """Tests that a model can take an object."""

    def __init__(self, hparams):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.save_hyperparameters(hparams)

def decorate(func):
    if False:
        for i in range(10):
            print('nop')

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            print('Hello World!')
        return func(*args, **kwargs)
    return wrapper

class SaveHparamsDecoratedModel(BoringModel):
    """Tests that a model can take an object."""

    @decorate
    @decorate
    def __init__(self, hparams, *my_args, **my_kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.save_hyperparameters(hparams)

class SaveHparamsDataModule(BoringDataModule):
    """Tests that a model can take an object."""

    def __init__(self, hparams):
        if False:
            return 10
        super().__init__()
        self.save_hyperparameters(hparams)

class SaveHparamsDecoratedDataModule(BoringDataModule):
    """Tests that a model can take an object."""

    @decorate
    @decorate
    def __init__(self, hparams, *my_args, **my_kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.save_hyperparameters(hparams)

def _run_standard_hparams_test(tmpdir, model, cls, datamodule=None, try_overwrite=False):
    if False:
        for i in range(10):
            print('nop')
    "Tests for the existence of an arg 'test_arg=14'."
    obj = datamodule if issubclass(cls, LightningDataModule) else model
    hparam_type = type(obj.hparams)
    assert obj.hparams.test_arg == 14
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, overfit_batches=2)
    trainer.fit(model, datamodule=datamodule if issubclass(cls, LightningDataModule) else None)
    raw_checkpoint_path = _raw_checkpoint_path(trainer)
    raw_checkpoint = torch.load(raw_checkpoint_path)
    assert cls.CHECKPOINT_HYPER_PARAMS_KEY in raw_checkpoint
    assert raw_checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY]['test_arg'] == 14
    obj2 = cls.load_from_checkpoint(raw_checkpoint_path)
    assert obj2.hparams.test_arg == 14
    assert isinstance(obj2.hparams, hparam_type)
    if try_overwrite:
        obj3 = cls.load_from_checkpoint(raw_checkpoint_path, test_arg=78)
        assert obj3.hparams.test_arg == 78
    return raw_checkpoint_path

@pytest.mark.parametrize('cls', [SaveHparamsModel, SaveHparamsDecoratedModel, SaveHparamsDataModule, SaveHparamsDecoratedDataModule])
def test_namespace_hparams(tmpdir, cls):
    if False:
        return 10
    hparams = Namespace(test_arg=14)
    if issubclass(cls, LightningDataModule):
        model = BoringModel()
        datamodule = cls(hparams=hparams)
    else:
        model = cls(hparams=hparams)
        datamodule = None
    _run_standard_hparams_test(tmpdir, model, cls, datamodule=datamodule)

@pytest.mark.parametrize('cls', [SaveHparamsModel, SaveHparamsDecoratedModel, SaveHparamsDataModule, SaveHparamsDecoratedDataModule])
def test_dict_hparams(tmpdir, cls):
    if False:
        return 10
    hparams = {'test_arg': 14}
    if issubclass(cls, LightningDataModule):
        model = BoringModel()
        datamodule = cls(hparams=hparams)
    else:
        model = cls(hparams=hparams)
        datamodule = None
    _run_standard_hparams_test(tmpdir, model, cls, datamodule=datamodule)

@RunIf(omegaconf=True)
@pytest.mark.parametrize('cls', [SaveHparamsModel, SaveHparamsDecoratedModel, SaveHparamsDataModule, SaveHparamsDecoratedDataModule])
def test_omega_conf_hparams(tmpdir, cls):
    if False:
        return 10
    conf = OmegaConf.create({'test_arg': 14, 'mylist': [15.4, {'a': 1, 'b': 2}]})
    if issubclass(cls, LightningDataModule):
        model = BoringModel()
        obj = datamodule = cls(hparams=conf)
    else:
        obj = model = cls(hparams=conf)
        datamodule = None
    assert isinstance(obj.hparams, Container)
    raw_checkpoint_path = _run_standard_hparams_test(tmpdir, model, cls, datamodule=datamodule)
    obj2 = cls.load_from_checkpoint(raw_checkpoint_path)
    assert isinstance(obj2.hparams, Container)
    assert obj2.hparams.test_arg == 14
    assert obj2.hparams.mylist[0] == 15.4

def test_explicit_args_hparams(tmpdir):
    if False:
        return 10
    'Tests that a model can take implicit args and assign.'

    class LocalModel(BoringModel):

        def __init__(self, test_arg, test_arg2):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__()
            self.save_hyperparameters('test_arg', 'test_arg2')
    model = LocalModel(test_arg=14, test_arg2=90)
    raw_checkpoint_path = _run_standard_hparams_test(tmpdir, model, LocalModel)
    model = LocalModel.load_from_checkpoint(raw_checkpoint_path, test_arg2=120)
    assert model.hparams.test_arg2 == 120

def test_implicit_args_hparams(tmpdir):
    if False:
        print('Hello World!')
    'Tests that a model can take regular args and assign.'

    class LocalModel(BoringModel):

        def __init__(self, test_arg, test_arg2):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__()
            self.save_hyperparameters()
    model = LocalModel(test_arg=14, test_arg2=90)
    raw_checkpoint_path = _run_standard_hparams_test(tmpdir, model, LocalModel)
    model = LocalModel.load_from_checkpoint(raw_checkpoint_path, test_arg2=120)
    assert model.hparams.test_arg2 == 120

def test_explicit_missing_args_hparams(tmpdir):
    if False:
        while True:
            i = 10
    'Tests that a model can take regular args and assign.'

    class LocalModel(BoringModel):

        def __init__(self, test_arg, test_arg2):
            if False:
                print('Hello World!')
            super().__init__()
            self.save_hyperparameters('test_arg')
    model = LocalModel(test_arg=14, test_arg2=90)
    assert model.hparams.test_arg == 14
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, overfit_batches=0.5)
    trainer.fit(model)
    raw_checkpoint_path = _raw_checkpoint_path(trainer)
    raw_checkpoint = torch.load(raw_checkpoint_path)
    assert LightningModule.CHECKPOINT_HYPER_PARAMS_KEY in raw_checkpoint
    assert raw_checkpoint[LightningModule.CHECKPOINT_HYPER_PARAMS_KEY]['test_arg'] == 14
    model = LocalModel.load_from_checkpoint(raw_checkpoint_path, test_arg2=123)
    assert model.hparams.test_arg == 14
    assert 'test_arg2' not in model.hparams
    return raw_checkpoint_path

def test_class_nesting():
    if False:
        while True:
            i = 10

    class MyModule(LightningModule):

        def forward(self):
            if False:
                i = 10
                return i + 15
            ...
    a = MyModule()
    assert isinstance(a, torch.nn.Module)

    def test_outside():
        if False:
            i = 10
            return i + 15
        a = MyModule()
        _ = a.hparams

    class A:

        def test(self):
            if False:
                i = 10
                return i + 15
            a = MyModule()
            _ = a.hparams

        def test2(self):
            if False:
                print('Hello World!')
            test_outside()
    test_outside()
    A().test2()
    A().test()

class CustomBoringModel(BoringModel):

    def __init__(self, batch_size=64):
        if False:
            print('Hello World!')
        super().__init__()
        self.save_hyperparameters()

class SubClassBoringModel(CustomBoringModel):
    any_other_loss = torch.nn.CrossEntropyLoss()

    def __init__(self, *args, subclass_arg=1200, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

class MixinForBoringModel:
    any_other_loss = torch.nn.CrossEntropyLoss()

    def __init__(self, *args, subclass_arg=1200, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

class BoringModelWithMixin(MixinForBoringModel, CustomBoringModel):
    pass

class BoringModelWithMixinAndInit(MixinForBoringModel, CustomBoringModel):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

class NonSavingSubClassBoringModel(CustomBoringModel):
    any_other_loss = torch.nn.CrossEntropyLoss()

    def __init__(self, *args, subclass_arg=1200, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)

class SubSubClassBoringModel(SubClassBoringModel):
    pass

class AggSubClassBoringModel(SubClassBoringModel):

    def __init__(self, *args, my_loss=torch.nn.CrossEntropyLoss(), **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

class UnconventionalArgsBoringModel(CustomBoringModel):
    """A model that has unconventional names for "self", "*args" and "**kwargs"."""

    def __init__(obj, *more_args, other_arg=300, **more_kwargs):
        if False:
            print('Hello World!')
        super().__init__(*more_args, **more_kwargs)
        obj.save_hyperparameters()
if _OMEGACONF_AVAILABLE:

    class DictConfSubClassBoringModel(SubClassBoringModel):

        def __init__(self, *args, dict_conf=OmegaConf.create({'my_param': 'something'}), **kwargs):
            if False:
                while True:
                    i = 10
            super().__init__(*args, **kwargs)
            self.save_hyperparameters()
else:

    class DictConfSubClassBoringModel:
        ...

@pytest.mark.parametrize('cls', [CustomBoringModel, SubClassBoringModel, NonSavingSubClassBoringModel, SubSubClassBoringModel, AggSubClassBoringModel, UnconventionalArgsBoringModel, pytest.param(DictConfSubClassBoringModel, marks=RunIf(omegaconf=True)), BoringModelWithMixin, BoringModelWithMixinAndInit])
def test_collect_init_arguments(tmpdir, cls):
    if False:
        return 10
    'Test that the model automatically saves the arguments passed into the constructor.'
    extra_args = {}
    if cls is AggSubClassBoringModel:
        extra_args.update(my_loss=torch.nn.CosineEmbeddingLoss())
    elif cls is DictConfSubClassBoringModel:
        extra_args.update(dict_conf=OmegaConf.create({'my_param': 'anything'}))
    model = cls(**extra_args)
    assert model.hparams.batch_size == 64
    model = cls(batch_size=179, **extra_args)
    assert model.hparams.batch_size == 179
    if isinstance(model, (SubClassBoringModel, NonSavingSubClassBoringModel, MixinForBoringModel)):
        assert model.hparams.subclass_arg == 1200
    if isinstance(model, AggSubClassBoringModel):
        assert isinstance(model.hparams.my_loss, torch.nn.CosineEmbeddingLoss)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, overfit_batches=0.5)
    trainer.fit(model)
    raw_checkpoint_path = _raw_checkpoint_path(trainer)
    raw_checkpoint = torch.load(raw_checkpoint_path)
    assert LightningModule.CHECKPOINT_HYPER_PARAMS_KEY in raw_checkpoint
    assert raw_checkpoint[LightningModule.CHECKPOINT_HYPER_PARAMS_KEY]['batch_size'] == 179
    model = cls.load_from_checkpoint(raw_checkpoint_path)
    assert model.hparams.batch_size == 179
    if isinstance(model, AggSubClassBoringModel):
        assert isinstance(model.hparams.my_loss, torch.nn.CosineEmbeddingLoss)
    if isinstance(model, DictConfSubClassBoringModel):
        assert isinstance(model.hparams.dict_conf, Container)
        assert model.hparams.dict_conf['my_param'] == 'anything'
    model = cls.load_from_checkpoint(raw_checkpoint_path, batch_size=99)
    assert model.hparams.batch_size == 99

def _raw_checkpoint_path(trainer) -> str:
    if False:
        i = 10
        return i + 15
    raw_checkpoint_paths = os.listdir(trainer.checkpoint_callback.dirpath)
    raw_checkpoint_paths = [x for x in raw_checkpoint_paths if '.ckpt' in x]
    assert raw_checkpoint_paths
    raw_checkpoint_path = raw_checkpoint_paths[0]
    raw_checkpoint_path = os.path.join(trainer.checkpoint_callback.dirpath, raw_checkpoint_path)
    return raw_checkpoint_path

@pytest.mark.parametrize('base_class', [HyperparametersMixin, LightningModule, LightningDataModule])
def test_save_hyperparameters_under_composition(base_class):
    if False:
        return 10
    "Test that in a composition where the parent is not a Lightning-like module, the parent's arguments don't get\n    collected."

    class ChildInComposition(base_class):

        def __init__(self, same_arg):
            if False:
                print('Hello World!')
            super().__init__()
            self.save_hyperparameters()

    class NotPLSubclass:

        def __init__(self, same_arg='parent_default', other_arg='other'):
            if False:
                return 10
            self.child = ChildInComposition(same_arg='cocofruit')
    parent = NotPLSubclass()
    assert parent.child.hparams == {'same_arg': 'cocofruit'}

class LocalVariableModelSuperLast(BoringModel):
    """This model has the super().__init__() call at the end."""

    def __init__(self, arg1, arg2, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.argument1 = arg1
        arg1 = 'overwritten'
        local_var = 1234
        super().__init__(*args, **kwargs)

class LocalVariableModelSuperFirst(BoringModel):
    """This model has the save_hyperparameters() call at the end."""

    def __init__(self, arg1, arg2, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        self.argument1 = arg1
        arg1 = 'overwritten'
        local_var = 1234
        self.save_hyperparameters()

@pytest.mark.parametrize('cls', [LocalVariableModelSuperFirst])
def test_collect_init_arguments_with_local_vars(cls):
    if False:
        print('Hello World!')
    'Tests that only the arguments are collected and not local variables.'
    model = cls(arg1=1, arg2=2)
    assert 'local_var' not in model.hparams
    assert model.hparams['arg1'] == 'overwritten'
    assert model.hparams['arg2'] == 2

class AnotherArgModel(BoringModel):

    def __init__(self, arg1):
        if False:
            print('Hello World!')
        super().__init__()
        self.save_hyperparameters(arg1)

class OtherArgsModel(BoringModel):

    def __init__(self, arg1, arg2):
        if False:
            return 10
        super().__init__()
        self.save_hyperparameters(arg1, arg2)

@pytest.mark.parametrize(('cls', 'config'), [(AnotherArgModel, {'arg1': 42}), (OtherArgsModel, {'arg1': 3.14, 'arg2': 'abc'})])
def test_single_config_models_fail(tmpdir, cls, config):
    if False:
        print('Hello World!')
    'Test fail on passing unsupported config type.'
    with pytest.raises(ValueError):
        _ = cls(**config)

@pytest.mark.parametrize('past_key', ['module_arguments'])
def test_load_past_checkpoint(tmpdir, past_key):
    if False:
        print('Hello World!')
    model = CustomBoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)
    trainer.fit(model)
    raw_checkpoint_path = _raw_checkpoint_path(trainer)
    raw_checkpoint = torch.load(raw_checkpoint_path)
    raw_checkpoint[past_key] = raw_checkpoint[LightningModule.CHECKPOINT_HYPER_PARAMS_KEY]
    raw_checkpoint['hparams_type'] = 'Namespace'
    raw_checkpoint[past_key]['batch_size'] = -17
    del raw_checkpoint[LightningModule.CHECKPOINT_HYPER_PARAMS_KEY]
    torch.save(raw_checkpoint, raw_checkpoint_path)
    model2 = CustomBoringModel.load_from_checkpoint(raw_checkpoint_path)
    assert model2.hparams.batch_size == -17

def test_hparams_pickle(tmpdir):
    if False:
        print('Hello World!')
    ad = AttributeDict({'key1': 1, 'key2': 'abc'})
    pkl = pickle.dumps(ad)
    assert ad == pickle.loads(pkl)
    pkl = cloudpickle.dumps(ad)
    assert ad == pickle.loads(pkl)

class UnpickleableArgsBoringModel(BoringModel):
    """A model that has an attribute that cannot be pickled."""

    def __init__(self, foo='bar', pickle_me=lambda x: x + 1, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        assert not is_picklable(pickle_me)
        self.save_hyperparameters()

def test_hparams_pickle_warning(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    model = UnpickleableArgsBoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_steps=1)
    with pytest.warns(UserWarning, match="attribute 'pickle_me' removed from hparams because it cannot be pickled"):
        trainer.fit(model)
    assert 'pickle_me' not in model.hparams

def test_hparams_save_yaml(tmpdir):
    if False:
        for i in range(10):
            print('nop')

    class Options(str, Enum):
        option1name = 'option1val'
        option2name = 'option2val'
        option3name = 'option3val'
    hparams = {'batch_size': 32, 'learning_rate': 0.001, 'data_root': './any/path/here', 'nested': {'any_num': 123, 'anystr': 'abcd'}, 'switch': Options.option3name}
    path_yaml = os.path.join(tmpdir, 'testing-hparams.yaml')

    def _compare_params(loaded_params, default_params: dict):
        if False:
            while True:
                i = 10
        assert isinstance(loaded_params, (dict, DictConfig))
        assert loaded_params.keys() == default_params.keys()
        for (k, v) in default_params.items():
            if isinstance(v, Enum):
                assert v.name == loaded_params[k]
            else:
                assert v == loaded_params[k]
    save_hparams_to_yaml(path_yaml, hparams)
    _compare_params(load_hparams_from_yaml(path_yaml, use_omegaconf=False), hparams)
    save_hparams_to_yaml(path_yaml, Namespace(**hparams))
    _compare_params(load_hparams_from_yaml(path_yaml, use_omegaconf=False), hparams)
    save_hparams_to_yaml(path_yaml, AttributeDict(hparams))
    _compare_params(load_hparams_from_yaml(path_yaml, use_omegaconf=False), hparams)
    if _OMEGACONF_AVAILABLE:
        save_hparams_to_yaml(path_yaml, OmegaConf.create(hparams))
        _compare_params(load_hparams_from_yaml(path_yaml), hparams)

class NoArgsSubClassBoringModel(CustomBoringModel):

    def __init__(self):
        if False:
            return 10
        super().__init__()

@pytest.mark.parametrize('cls', [BoringModel, NoArgsSubClassBoringModel])
def test_model_nohparams_train_test(tmpdir, cls):
    if False:
        return 10
    'Test models that do not take any argument in init.'
    model = cls()
    trainer = Trainer(max_epochs=1, default_root_dir=tmpdir)
    train_loader = DataLoader(RandomDataset(32, 64), batch_size=32)
    trainer.fit(model, train_loader)
    test_loader = DataLoader(RandomDataset(32, 64), batch_size=32)
    trainer.test(dataloaders=test_loader)

def test_model_ignores_non_exist_kwargument(tmpdir):
    if False:
        i = 10
        return i + 15
    'Test that the model takes only valid class arguments.'

    class LocalModel(BoringModel):

        def __init__(self, batch_size=15):
            if False:
                i = 10
                return i + 15
            super().__init__()
            self.save_hyperparameters()
    model = LocalModel()
    assert model.hparams.batch_size == 15
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)
    trainer.fit(model)
    raw_checkpoint_path = _raw_checkpoint_path(trainer)
    model = LocalModel.load_from_checkpoint(raw_checkpoint_path, non_exist_kwarg=99)
    assert 'non_exist_kwarg' not in model.hparams

class SuperClassPositionalArgs(BoringModel):

    def __init__(self, hparams):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._hparams = hparams

class SubClassVarArgs(SuperClassPositionalArgs):
    """Loading this model should accept hparams and init in the super class."""

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)

def test_args(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Test for inheritance: super class takes positional arg, subclass takes varargs.'
    hparams = {'test': 1}
    model = SubClassVarArgs(hparams)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)
    trainer.fit(model)
    raw_checkpoint_path = _raw_checkpoint_path(trainer)
    with pytest.raises(TypeError, match="__init__\\(\\) got an unexpected keyword argument 'test'"):
        SubClassVarArgs.load_from_checkpoint(raw_checkpoint_path)

class RuntimeParamChangeModelSaving(BoringModel):

    def __init__(self, **kwargs):
        if False:
            return 10
        super().__init__()
        self.save_hyperparameters()

@pytest.mark.parametrize('cls', [RuntimeParamChangeModelSaving])
def test_init_arg_with_runtime_change(tmpdir, cls):
    if False:
        return 10
    'Test that we save/export only the initial hparams, no other runtime change allowed.'
    model = cls(running_arg=123)
    assert model.hparams.running_arg == 123
    model.hparams.running_arg = -1
    assert model.hparams.running_arg == -1
    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=2, limit_val_batches=2, limit_test_batches=2, max_epochs=1, logger=TensorBoardLogger(tmpdir))
    trainer.fit(model)
    path_yaml = os.path.join(trainer.logger.log_dir, trainer.logger.NAME_HPARAMS_FILE)
    hparams = load_hparams_from_yaml(path_yaml)
    assert hparams.get('running_arg') == 123

class UnsafeParamModel(BoringModel):

    def __init__(self, my_path, any_param=123):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.save_hyperparameters()

def test_model_with_fsspec_as_parameter(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    model = UnsafeParamModel(LocalFileSystem(tmpdir))
    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=2, limit_val_batches=2, limit_test_batches=2, max_epochs=1)
    trainer.fit(model)
    trainer.test()

@pytest.mark.skipif(RequirementCache('hydra-core<1.1'), reason="Requires Hydra's Compose API")
def test_model_save_hyper_parameters_interpolation_with_hydra(tmpdir):
    if False:
        print('Hello World!')
    'This test relies on configuration saved under tests/models/conf/config.yaml.'
    from hydra import compose, initialize

    class TestHydraModel(BoringModel):

        def __init__(self, args_0, args_1, args_2, kwarg_1=None):
            if False:
                print('Hello World!')
            self.save_hyperparameters()
            assert self.hparams.args_0.log == 'Something'
            assert self.hparams.args_1['cfg'].log == 'Something'
            assert self.hparams.args_2[0].log == 'Something'
            assert self.hparams.kwarg_1['cfg'][0].log == 'Something'
            super().__init__()
    with initialize(config_path='conf'):
        args_0 = compose(config_name='config')
        args_1 = {'cfg': compose(config_name='config')}
        args_2 = [compose(config_name='config')]
        kwarg_1 = {'cfg': [compose(config_name='config')]}
        model = TestHydraModel(args_0, args_1, args_2, kwarg_1=kwarg_1)
        epochs = 2
        checkpoint_callback = ModelCheckpoint(monitor=None, dirpath=tmpdir, save_top_k=-1)
        trainer = Trainer(default_root_dir=tmpdir, callbacks=[checkpoint_callback], limit_train_batches=10, limit_val_batches=10, max_epochs=epochs, logger=False)
        trainer.fit(model)
        _ = TestHydraModel.load_from_checkpoint(checkpoint_callback.best_model_path)

@pytest.mark.parametrize('ignore', ['arg2', ('arg2', 'arg3')])
def test_ignore_args_list_hparams(tmpdir, ignore):
    if False:
        return 10
    'Tests that args can be ignored in save_hyperparameters.'

    class LocalModel(BoringModel):

        def __init__(self, arg1, arg2, arg3):
            if False:
                i = 10
                return i + 15
            super().__init__()
            self.save_hyperparameters(ignore=ignore)
    model = LocalModel(arg1=14, arg2=90, arg3=50)
    assert model.hparams.arg1 == 14
    for arg in ignore:
        assert arg not in model.hparams
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, overfit_batches=0.5)
    trainer.fit(model)
    raw_checkpoint_path = _raw_checkpoint_path(trainer)
    raw_checkpoint = torch.load(raw_checkpoint_path)
    assert LightningModule.CHECKPOINT_HYPER_PARAMS_KEY in raw_checkpoint
    assert raw_checkpoint[LightningModule.CHECKPOINT_HYPER_PARAMS_KEY]['arg1'] == 14
    model = LocalModel.load_from_checkpoint(raw_checkpoint_path, arg2=123, arg3=100)
    assert model.hparams.arg1 == 14
    for arg in ignore:
        assert arg not in model.hparams

class IgnoreAllParametersModel(BoringModel):

    def __init__(self, arg1, arg2, arg3):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.save_hyperparameters(ignore=('arg1', 'arg2', 'arg3'))

class NoParametersModel(BoringModel):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.save_hyperparameters()

@pytest.mark.parametrize('model', [IgnoreAllParametersModel(arg1=14, arg2=90, arg3=50), NoParametersModel()])
def test_save_no_parameters(model):
    if False:
        while True:
            i = 10
    'Test that calling save_hyperparameters works if no parameters need saving.'
    assert model.hparams == {}
    assert model._hparams_initial == {}

class HparamsKwargsContainerModel(BoringModel):

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        super().__init__()
        self.save_hyperparameters(kwargs)

class HparamsNamespaceContainerModel(BoringModel):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.save_hyperparameters(config)

def test_empty_hparams_container(tmpdir):
    if False:
        print('Hello World!')
    'Test that save_hyperparameters() is a no-op when saving an empty hparams container.'
    model = HparamsKwargsContainerModel()
    assert not model.hparams
    model = HparamsNamespaceContainerModel(Namespace())
    assert not model.hparams

def test_hparams_name_from_container(tmpdir):
    if False:
        while True:
            i = 10
    'Test that save_hyperparameters(container) captures the name of the argument correctly.'
    model = HparamsKwargsContainerModel(a=1, b=2)
    assert model._hparams_name is None
    model = HparamsNamespaceContainerModel(Namespace(a=1, b=2))
    assert model._hparams_name == 'config'

@dataclass
class DataClassModel(BoringModel):
    mandatory: int
    optional: str = 'optional'
    ignore_me: bool = False

    def __post_init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.save_hyperparameters(ignore=('ignore_me',))

def test_dataclass_lightning_module(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Test that save_hyperparameters() works with a LightningModule as a dataclass.'
    model = DataClassModel(33, optional='cocofruit')
    assert model.hparams == {'mandatory': 33, 'optional': 'cocofruit'}

class NoHparamsModel(BoringModel):
    """Tests a model without hparams."""

class DataModuleWithoutHparams(LightningDataModule):

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        if False:
            for i in range(10):
                print('nop')
        return DataLoader(RandomDataset(32, 64), batch_size=32)

class DataModuleWithHparams(LightningDataModule):

    def __init__(self, hparams):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.save_hyperparameters(hparams)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        if False:
            for i in range(10):
                print('nop')
        return DataLoader(RandomDataset(32, 64), batch_size=32)

def _get_mock_logger(tmpdir):
    if False:
        i = 10
        return i + 15
    mock_logger = mock.MagicMock(name='logger')
    mock_logger.name = 'mock_logger'
    mock_logger.save_dir = tmpdir
    mock_logger.version = '0'
    del mock_logger.__iter__
    return mock_logger

@pytest.mark.parametrize('model', [SaveHparamsModel({'arg1': 5, 'arg2': 'abc'}), NoHparamsModel()])
@pytest.mark.parametrize('data', [DataModuleWithHparams({'data_dir': 'foo'}), DataModuleWithoutHparams()])
def test_adding_datamodule_hparams(tmpdir, model, data):
    if False:
        return 10
    'Test that hparams from datamodule and model are logged.'
    org_model_hparams = copy.deepcopy(model.hparams_initial)
    org_data_hparams = copy.deepcopy(data.hparams_initial)
    mock_logger = _get_mock_logger(tmpdir)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, logger=mock_logger)
    trainer.fit(model, datamodule=data)
    assert org_model_hparams == model.hparams
    assert org_data_hparams == data.hparams
    merged_hparams = copy.deepcopy(org_model_hparams)
    merged_hparams.update(org_data_hparams)
    if merged_hparams:
        mock_logger.log_hyperparams.assert_called_with(merged_hparams)
    else:
        mock_logger.log_hyperparams.assert_not_called()

def test_no_datamodule_for_hparams(tmpdir):
    if False:
        return 10
    'Test that hparams model are logged if no datamodule is used.'
    model = SaveHparamsModel({'arg1': 5, 'arg2': 'abc'})
    org_model_hparams = copy.deepcopy(model.hparams_initial)
    data = DataModuleWithoutHparams()
    data.setup('fit')
    mock_logger = _get_mock_logger(tmpdir)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, logger=mock_logger)
    trainer.fit(model, datamodule=data)
    mock_logger.log_hyperparams.assert_called_with(org_model_hparams)

def test_colliding_hparams(tmpdir):
    if False:
        print('Hello World!')
    model = SaveHparamsModel({'data_dir': 'abc', 'arg2': 'abc'})
    data = DataModuleWithHparams({'data_dir': 'foo'})
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, logger=CSVLogger(tmpdir))
    with pytest.raises(RuntimeError, match='Error while merging hparams:'):
        trainer.fit(model, datamodule=data)

def test_nn_modules_warning_when_saved_as_hparams():
    if False:
        while True:
            i = 10

    class TorchModule(torch.nn.Module):

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__()
            self.l1 = torch.nn.Linear(4, 5)

    class CustomBoringModelWarn(BoringModel):

        def __init__(self, encoder, decoder, other_hparam=7):
            if False:
                i = 10
                return i + 15
            super().__init__()
            self.save_hyperparameters()
    with pytest.warns(UserWarning, match='is an instance of `nn.Module` and is already saved'):
        model = CustomBoringModelWarn(encoder=TorchModule(), decoder=TorchModule())
    assert list(model.hparams) == ['encoder', 'decoder', 'other_hparam']

    class CustomBoringModelNoWarn(BoringModel):

        def __init__(self, encoder, decoder, other_hparam=7):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__()
            self.save_hyperparameters('other_hparam')
    with no_warning_call(UserWarning, match='is an instance of `nn.Module` and is already saved'):
        model = CustomBoringModelNoWarn(encoder=TorchModule(), decoder=TorchModule())
    assert list(model.hparams) == ['other_hparam']