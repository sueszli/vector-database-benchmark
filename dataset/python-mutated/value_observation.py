from chainer.training import extension

def observe_value(observation_key, target_func):
    if False:
        i = 10
        return i + 15
    'Returns a trainer extension to continuously record a value.\n\n    Args:\n        observation_key (str): Key of observation to record.\n        target_func (function): Function that returns the value to record.\n            It must take one argument: :class:~chainer.training.Trainer object.\n    Returns:\n        The extension function.\n\n    This extension is triggered each epoch by default.\n    To change this, use the ``trigger`` argument with the\n    :meth:`Trainer.extend() <chainer.training.Trainer.extend>` method.\n\n    '

    @extension.make_extension(trigger=(1, 'epoch'), priority=extension.PRIORITY_WRITER)
    def _observe_value(trainer):
        if False:
            i = 10
            return i + 15
        trainer.observation[observation_key] = target_func(trainer)
    return _observe_value

def observe_lr(optimizer_name='main', observation_key='lr'):
    if False:
        i = 10
        return i + 15
    'Returns a trainer extension to record the learning rate.\n\n    Args:\n        optimizer_name (str): Name of optimizer whose learning rate is\n            recorded.\n        observation_key (str): Key of observation to record.\n\n    Returns:\n        The extension function.\n\n    This extension is triggered each epoch by default.\n    To change this, use the ``trigger`` argument with the\n    :meth:`Trainer.extend() <chainer.training.Trainer.extend>` method.\n\n    '
    return observe_value(observation_key, lambda trainer: trainer.updater.get_optimizer(optimizer_name).lr)