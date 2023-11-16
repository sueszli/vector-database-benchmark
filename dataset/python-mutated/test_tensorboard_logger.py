import math
import os
from unittest.mock import ANY, call, MagicMock, patch
import pytest
import torch
from ignite.contrib.handlers.tensorboard_logger import global_step_from_engine, GradsHistHandler, GradsScalarHandler, OptimizerParamsHandler, OutputHandler, TensorboardLogger, WeightsHistHandler, WeightsScalarHandler
from ignite.engine import Engine, Events, State

def test_optimizer_params_handler_wrong_setup():
    if False:
        i = 10
        return i + 15
    with pytest.raises(TypeError):
        OptimizerParamsHandler(optimizer=None)
    optimizer = MagicMock(spec=torch.optim.Optimizer)
    handler = OptimizerParamsHandler(optimizer=optimizer)
    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match='Handler OptimizerParamsHandler works only with TensorboardLogger'):
        handler(mock_engine, mock_logger, Events.ITERATION_STARTED)

def test_getattr_method():
    if False:
        print('Hello World!')
    mock_writer = MagicMock()
    logger = TensorboardLogger()
    logger.writer = mock_writer
    logger.add_scalar('loss', 0.5)
    mock_writer.add_scalar.assert_called_once_with('loss', 0.5)

def test_optimizer_params():
    if False:
        while True:
            i = 10
    optimizer = torch.optim.SGD([torch.tensor(0.0)], lr=0.01)
    wrapper = OptimizerParamsHandler(optimizer=optimizer, param_name='lr')
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.iteration = 123
    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.writer.add_scalar.assert_called_once_with('lr/group_0', 0.01, 123)
    wrapper = OptimizerParamsHandler(optimizer, param_name='lr', tag='generator')
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()
    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.writer.add_scalar.assert_called_once_with('generator/lr/group_0', 0.01, 123)

def test_output_handler_with_wrong_logger_type():
    if False:
        return 10
    wrapper = OutputHandler('tag', output_transform=lambda x: x)
    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match="Handler 'OutputHandler' works only with TensorboardLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

def test_output_handler_output_transform():
    if False:
        print('Hello World!')
    wrapper = OutputHandler('tag', output_transform=lambda x: x)
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.output = 12345
    mock_engine.state.iteration = 123
    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.writer.add_scalar.assert_called_once_with('tag/output', 12345, 123)
    wrapper = OutputHandler('another_tag', output_transform=lambda x: {'loss': x})
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()
    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.writer.add_scalar.assert_called_once_with('another_tag/loss', 12345, 123)

def test_output_handler_metric_names():
    if False:
        for i in range(10):
            print('nop')
    wrapper = OutputHandler('tag', metric_names=['a', 'b'])
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State(metrics={'a': 12.23, 'b': 23.45})
    mock_engine.state.iteration = 5
    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    assert mock_logger.writer.add_scalar.call_count == 2
    mock_logger.writer.add_scalar.assert_has_calls([call('tag/a', 12.23, 5), call('tag/b', 23.45, 5)], any_order=True)
    wrapper = OutputHandler('tag', metric_names=['a'])
    mock_engine = MagicMock()
    mock_engine.state = State(metrics={'a': torch.tensor([0.0, 1.0, 2.0, 3.0])})
    mock_engine.state.iteration = 5
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()
    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    assert mock_logger.writer.add_scalar.call_count == 4
    mock_logger.writer.add_scalar.assert_has_calls([call('tag/a/0', 0.0, 5), call('tag/a/1', 1.0, 5), call('tag/a/2', 2.0, 5), call('tag/a/3', 3.0, 5)], any_order=True)
    wrapper = OutputHandler('tag', metric_names=['a', 'c'])
    mock_engine = MagicMock()
    mock_engine.state = State(metrics={'a': 55.56, 'c': 'Some text'})
    mock_engine.state.iteration = 7
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()
    with pytest.warns(UserWarning):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    assert mock_logger.writer.add_scalar.call_count == 1
    mock_logger.writer.add_scalar.assert_has_calls([call('tag/a', 55.56, 7)], any_order=True)
    wrapper = OutputHandler('tag', metric_names='all')
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State(metrics={'a': 12.23, 'b': 23.45})
    mock_engine.state.iteration = 5
    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    assert mock_logger.writer.add_scalar.call_count == 2
    mock_logger.writer.add_scalar.assert_has_calls([call('tag/a', 12.23, 5), call('tag/b', 23.45, 5)], any_order=True)
    wrapper = OutputHandler('tag', metric_names='all')
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State(metrics={'a': torch.tensor(12.23), 'b': torch.tensor(23.45)})
    mock_engine.state.iteration = 5
    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    assert mock_logger.writer.add_scalar.call_count == 2
    mock_logger.writer.add_scalar.assert_has_calls([call('tag/a', torch.tensor(12.23).item(), 5), call('tag/b', torch.tensor(23.45).item(), 5)], any_order=True)

def test_output_handler_both():
    if False:
        print('Hello World!')
    wrapper = OutputHandler('tag', metric_names=['a', 'b'], output_transform=lambda x: {'loss': x})
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State(metrics={'a': 12.23, 'b': 23.45})
    mock_engine.state.epoch = 5
    mock_engine.state.output = 12345
    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    assert mock_logger.writer.add_scalar.call_count == 3
    mock_logger.writer.add_scalar.assert_has_calls([call('tag/a', 12.23, 5), call('tag/b', 23.45, 5), call('tag/loss', 12345, 5)], any_order=True)

def test_output_handler_with_wrong_global_step_transform_output():
    if False:
        for i in range(10):
            print('nop')

    def global_step_transform(*args, **kwargs):
        if False:
            print('Hello World!')
        return 'a'
    wrapper = OutputHandler('tag', output_transform=lambda x: {'loss': x}, global_step_transform=global_step_transform)
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5
    mock_engine.state.output = 12345
    with pytest.raises(TypeError, match='global_step must be int'):
        wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

def test_output_handler_with_global_step_from_engine():
    if False:
        for i in range(10):
            print('nop')
    mock_another_engine = MagicMock()
    mock_another_engine.state = State()
    mock_another_engine.state.epoch = 10
    mock_another_engine.state.output = 12.345
    wrapper = OutputHandler('tag', output_transform=lambda x: {'loss': x}, global_step_transform=global_step_from_engine(mock_another_engine))
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 1
    mock_engine.state.output = 0.123
    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    assert mock_logger.writer.add_scalar.call_count == 1
    mock_logger.writer.add_scalar.assert_has_calls([call('tag/loss', mock_engine.state.output, mock_another_engine.state.epoch)])
    mock_another_engine.state.epoch = 11
    mock_engine.state.output = 1.123
    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    assert mock_logger.writer.add_scalar.call_count == 2
    mock_logger.writer.add_scalar.assert_has_calls([call('tag/loss', mock_engine.state.output, mock_another_engine.state.epoch)])

def test_output_handler_with_global_step_transform():
    if False:
        while True:
            i = 10

    def global_step_transform(*args, **kwargs):
        if False:
            print('Hello World!')
        return 10
    wrapper = OutputHandler('tag', output_transform=lambda x: {'loss': x}, global_step_transform=global_step_transform)
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5
    mock_engine.state.output = 12345
    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    assert mock_logger.writer.add_scalar.call_count == 1
    mock_logger.writer.add_scalar.assert_has_calls([call('tag/loss', 12345, 10)])

def test_weights_scalar_handler_wrong_setup():
    if False:
        i = 10
        return i + 15
    model = MagicMock(spec=torch.nn.Module)
    wrapper = WeightsScalarHandler(model)
    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match="Handler 'WeightsScalarHandler' works only with TensorboardLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

def test_weights_scalar_handler(dummy_model_factory):
    if False:
        i = 10
        return i + 15
    model = dummy_model_factory(with_grads=True, with_frozen_layer=False)

    def _test(tag=None):
        if False:
            print('Hello World!')
        wrapper = WeightsScalarHandler(model, tag=tag)
        mock_logger = MagicMock(spec=TensorboardLogger)
        mock_logger.writer = MagicMock()
        mock_engine = MagicMock()
        mock_engine.state = State()
        mock_engine.state.epoch = 5
        wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
        tag_prefix = f'{tag}/' if tag else ''
        assert mock_logger.writer.add_scalar.call_count == 4
        mock_logger.writer.add_scalar.assert_has_calls([call(tag_prefix + 'weights_norm/fc1/weight', 0.0, 5), call(tag_prefix + 'weights_norm/fc1/bias', 0.0, 5), call(tag_prefix + 'weights_norm/fc2/weight', 12.0, 5), call(tag_prefix + 'weights_norm/fc2/bias', pytest.approx(math.sqrt(12.0)), 5)], any_order=True)
    _test()
    _test(tag='tag')

def test_weights_scalar_handler_whitelist(dummy_model_factory):
    if False:
        for i in range(10):
            print('nop')
    model = dummy_model_factory()
    wrapper = WeightsScalarHandler(model, whitelist=['fc2.weight'])
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5
    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    mock_logger.writer.add_scalar.assert_called_once_with('weights_norm/fc2/weight', 12.0, 5)
    mock_logger.writer.reset_mock()
    wrapper = WeightsScalarHandler(model, tag='model', whitelist=['fc1'])
    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    mock_logger.writer.add_scalar.assert_has_calls([call('model/weights_norm/fc1/weight', 0.0, 5), call('model/weights_norm/fc1/bias', 0.0, 5)], any_order=True)
    assert mock_logger.writer.add_scalar.call_count == 2
    mock_logger.writer.reset_mock()

    def weight_selector(n, _):
        if False:
            i = 10
            return i + 15
        return 'bias' in n
    wrapper = WeightsScalarHandler(model, tag='model', whitelist=weight_selector)
    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    mock_logger.writer.add_scalar.assert_has_calls([call('model/weights_norm/fc1/bias', 0.0, 5), call('model/weights_norm/fc2/bias', pytest.approx(math.sqrt(12.0)), 5)], any_order=True)
    assert mock_logger.writer.add_scalar.call_count == 2

def test_weights_hist_handler_wrong_setup():
    if False:
        while True:
            i = 10
    model = MagicMock(spec=torch.nn.Module)
    wrapper = WeightsHistHandler(model)
    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match="Handler 'WeightsHistHandler' works only with TensorboardLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

def test_weights_hist_handler(dummy_model_factory):
    if False:
        i = 10
        return i + 15
    model = dummy_model_factory(with_grads=True, with_frozen_layer=False)

    def _test(tag=None):
        if False:
            print('Hello World!')
        wrapper = WeightsHistHandler(model, tag=tag)
        mock_logger = MagicMock(spec=TensorboardLogger)
        mock_logger.writer = MagicMock()
        mock_engine = MagicMock()
        mock_engine.state = State()
        mock_engine.state.epoch = 5
        wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
        tag_prefix = f'{tag}/' if tag else ''
        assert mock_logger.writer.add_histogram.call_count == 4
        mock_logger.writer.add_histogram.assert_has_calls([call(tag=tag_prefix + 'weights/fc1/weight', values=ANY, global_step=5), call(tag=tag_prefix + 'weights/fc1/bias', values=ANY, global_step=5), call(tag=tag_prefix + 'weights/fc2/weight', values=ANY, global_step=5), call(tag=tag_prefix + 'weights/fc2/bias', values=ANY, global_step=5)], any_order=True)
    _test()
    _test(tag='tag')

def test_weights_hist_handler_whitelist(dummy_model_factory):
    if False:
        for i in range(10):
            print('nop')
    model = dummy_model_factory()
    wrapper = WeightsHistHandler(model, whitelist=['fc2.weight'])
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5
    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    mock_logger.writer.add_histogram.assert_called_once_with(tag='weights/fc2/weight', values=ANY, global_step=5)
    mock_logger.writer.reset_mock()
    wrapper = WeightsHistHandler(model, tag='model', whitelist=['fc1'])
    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    mock_logger.writer.add_histogram.assert_has_calls([call(tag='model/weights/fc1/weight', values=ANY, global_step=5), call(tag='model/weights/fc1/bias', values=ANY, global_step=5)], any_order=True)
    assert mock_logger.writer.add_histogram.call_count == 2
    mock_logger.writer.reset_mock()

    def weight_selector(n, _):
        if False:
            while True:
                i = 10
        return 'bias' in n
    wrapper = WeightsHistHandler(model, tag='model', whitelist=weight_selector)
    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    mock_logger.writer.add_histogram.assert_has_calls([call(tag='model/weights/fc1/bias', values=ANY, global_step=5), call(tag='model/weights/fc2/bias', values=ANY, global_step=5)], any_order=True)
    assert mock_logger.writer.add_histogram.call_count == 2

def test_grads_scalar_handler_wrong_setup():
    if False:
        return 10
    model = MagicMock(spec=torch.nn.Module)
    wrapper = GradsScalarHandler(model)
    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match="Handler 'GradsScalarHandler' works only with TensorboardLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

def test_grads_scalar_handler(dummy_model_factory, norm_mock):
    if False:
        i = 10
        return i + 15
    model = dummy_model_factory(with_grads=True, with_frozen_layer=False)

    def _test(tag=None):
        if False:
            i = 10
            return i + 15
        wrapper = GradsScalarHandler(model, reduction=norm_mock, tag=tag)
        mock_logger = MagicMock(spec=TensorboardLogger)
        mock_logger.writer = MagicMock()
        mock_engine = MagicMock()
        mock_engine.state = State()
        mock_engine.state.epoch = 5
        norm_mock.reset_mock()
        wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
        tag_prefix = f'{tag}/' if tag else ''
        mock_logger.writer.add_scalar.assert_has_calls([call(tag_prefix + 'grads_norm/fc1/weight', ANY, 5), call(tag_prefix + 'grads_norm/fc1/bias', ANY, 5), call(tag_prefix + 'grads_norm/fc2/weight', ANY, 5), call(tag_prefix + 'grads_norm/fc2/bias', ANY, 5)], any_order=True)
        assert mock_logger.writer.add_scalar.call_count == 4
        assert norm_mock.call_count == 4
    _test()
    _test(tag='tag')

def test_grads_scalar_handler_whitelist(dummy_model_factory, norm_mock):
    if False:
        return 10
    model = dummy_model_factory()
    wrapper = GradsScalarHandler(model, reduction=norm_mock, whitelist=['fc2.weight'])
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5
    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    mock_logger.writer.add_scalar.assert_called_once_with('grads_norm/fc2/weight', ANY, 5)
    mock_logger.writer.reset_mock()
    wrapper = GradsScalarHandler(model, tag='model', whitelist=['fc1'])
    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    mock_logger.writer.add_scalar.assert_has_calls([call('model/grads_norm/fc1/weight', ANY, 5), call('model/grads_norm/fc1/bias', ANY, 5)], any_order=True)
    assert mock_logger.writer.add_scalar.call_count == 2
    mock_logger.writer.reset_mock()

    def weight_selector(n, _):
        if False:
            while True:
                i = 10
        return 'bias' in n
    wrapper = GradsScalarHandler(model, tag='model', whitelist=weight_selector)
    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    mock_logger.writer.add_scalar.assert_has_calls([call('model/grads_norm/fc1/bias', ANY, 5), call('model/grads_norm/fc2/bias', ANY, 5)], any_order=True)
    assert mock_logger.writer.add_scalar.call_count == 2

def test_grads_hist_handler_wrong_setup():
    if False:
        for i in range(10):
            print('nop')
    model = MagicMock(spec=torch.nn.Module)
    wrapper = GradsHistHandler(model)
    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match="Handler 'GradsHistHandler' works only with TensorboardLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

def test_grads_hist_handler(dummy_model_factory):
    if False:
        return 10
    model = dummy_model_factory(with_grads=True, with_frozen_layer=False)

    def _test(tag=None):
        if False:
            for i in range(10):
                print('nop')
        wrapper = GradsHistHandler(model, tag=tag)
        mock_logger = MagicMock(spec=TensorboardLogger)
        mock_logger.writer = MagicMock()
        mock_engine = MagicMock()
        mock_engine.state = State()
        mock_engine.state.epoch = 5
        wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
        tag_prefix = f'{tag}/' if tag else ''
        assert mock_logger.writer.add_histogram.call_count == 4
        mock_logger.writer.add_histogram.assert_has_calls([call(tag=tag_prefix + 'grads/fc1/weight', values=ANY, global_step=5), call(tag=tag_prefix + 'grads/fc1/bias', values=ANY, global_step=5), call(tag=tag_prefix + 'grads/fc2/weight', values=ANY, global_step=5), call(tag=tag_prefix + 'grads/fc2/bias', values=ANY, global_step=5)], any_order=True)
    _test()
    _test(tag='tag')

def test_grads_hist_handler_whitelist(dummy_model_factory):
    if False:
        return 10
    model = dummy_model_factory()
    wrapper = GradsHistHandler(model, whitelist=['fc2.weight'])
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5
    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    mock_logger.writer.add_histogram.assert_called_once_with(tag='grads/fc2/weight', values=ANY, global_step=5)
    mock_logger.writer.reset_mock()
    wrapper = GradsHistHandler(model, tag='model', whitelist=['fc1'])
    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    mock_logger.writer.add_histogram.assert_has_calls([call(tag='model/grads/fc1/weight', values=ANY, global_step=5), call(tag='model/grads/fc1/bias', values=ANY, global_step=5)], any_order=True)
    assert mock_logger.writer.add_histogram.call_count == 2
    mock_logger.writer.reset_mock()

    def weight_selector(n, _):
        if False:
            return 10
        return 'bias' in n
    wrapper = GradsHistHandler(model, tag='model', whitelist=weight_selector)
    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    mock_logger.writer.add_histogram.assert_has_calls([call(tag='model/grads/fc1/bias', values=ANY, global_step=5), call(tag='model/grads/fc2/bias', values=ANY, global_step=5)], any_order=True)
    assert mock_logger.writer.add_histogram.call_count == 2

def test_integration(dirname):
    if False:
        return 10
    n_epochs = 5
    data = list(range(50))
    losses = torch.rand(n_epochs * len(data))
    losses_iter = iter(losses)

    def update_fn(engine, batch):
        if False:
            for i in range(10):
                print('nop')
        return next(losses_iter)
    trainer = Engine(update_fn)
    tb_logger = TensorboardLogger(log_dir=dirname)

    def dummy_handler(engine, logger, event_name):
        if False:
            for i in range(10):
                print('nop')
        global_step = engine.state.get_event_attrib_value(event_name)
        logger.writer.add_scalar('test_value', global_step, global_step)
    tb_logger.attach(trainer, log_handler=dummy_handler, event_name=Events.EPOCH_COMPLETED)
    trainer.run(data, max_epochs=n_epochs)
    tb_logger.close()
    written_files = os.listdir(dirname)
    written_files = [f for f in written_files if 'tfevents' in f]
    assert len(written_files) > 0

def test_integration_as_context_manager(dirname):
    if False:
        return 10
    n_epochs = 5
    data = list(range(50))
    losses = torch.rand(n_epochs * len(data))
    losses_iter = iter(losses)

    def update_fn(engine, batch):
        if False:
            return 10
        return next(losses_iter)
    with TensorboardLogger(log_dir=dirname) as tb_logger:
        trainer = Engine(update_fn)

        def dummy_handler(engine, logger, event_name):
            if False:
                print('Hello World!')
            global_step = engine.state.get_event_attrib_value(event_name)
            logger.writer.add_scalar('test_value', global_step, global_step)
        tb_logger.attach(trainer, log_handler=dummy_handler, event_name=Events.EPOCH_COMPLETED)
        trainer.run(data, max_epochs=n_epochs)
    written_files = os.listdir(dirname)
    written_files = [f for f in written_files if 'tfevents' in f]
    assert len(written_files) > 0

def test_no_tensorboardX_package(dirname):
    if False:
        print('Hello World!')
    from torch.utils.tensorboard import SummaryWriter
    with patch.dict('sys.modules', {'tensorboardX': None}):
        tb_logger = TensorboardLogger(log_dir=dirname)
        assert isinstance(tb_logger.writer, SummaryWriter), type(tb_logger.writer)
        tb_logger.close()

def test_no_torch_utils_tensorboard_package(dirname):
    if False:
        return 10
    from tensorboardX import SummaryWriter
    with patch.dict('sys.modules', {'torch.utils.tensorboard': None}):
        tb_logger = TensorboardLogger(log_dir=dirname)
        assert isinstance(tb_logger.writer, SummaryWriter), type(tb_logger.writer)
        tb_logger.close()

def test_no_tensorboardX_nor_torch_utils_tensorboard():
    if False:
        return 10
    with patch.dict('sys.modules', {'tensorboardX': None, 'torch.utils.tensorboard': None}):
        with pytest.raises(ModuleNotFoundError, match='This contrib module requires either tensorboardX or torch'):
            TensorboardLogger(log_dir=None)