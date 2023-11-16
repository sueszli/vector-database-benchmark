from chainer import reporter
from chainer.training import util

class BestValueTrigger(object):
    """Trigger invoked when specific value becomes best.

    Args:
        key (str): Key of value.
        compare (callable): Compare function which takes current best value and
            new value and returns whether new value is better than current
            best.
        trigger: Trigger that decides the comparison interval between current
            best value and new value. This must be a tuple in the form of
            ``<int>, 'epoch'`` or ``<int>, 'iteration'`` which is passed to
            :class:`~chainer.training.triggers.IntervalTrigger`.

    """

    def __init__(self, key, compare, trigger=(1, 'epoch')):
        if False:
            i = 10
            return i + 15
        self._key = key
        self._best_value = None
        self._interval_trigger = util.get_trigger(trigger)
        self._init_summary()
        self._compare = compare

    def __call__(self, trainer):
        if False:
            while True:
                i = 10
        'Decides whether the extension should be called on this iteration.\n\n        Args:\n            trainer (~chainer.training.Trainer): Trainer object that this\n                trigger is associated with. The ``observation`` of this trainer\n                is used to determine if the trigger should fire.\n\n        Returns:\n            bool: ``True`` if the corresponding extension should be invoked in\n            this iteration.\n\n        '
        observation = trainer.observation
        summary = self._summary
        key = self._key
        if key in observation:
            summary.add({key: observation[key]})
        if not self._interval_trigger(trainer):
            return False
        stats = summary.compute_mean()
        value = float(stats[key])
        self._init_summary()
        if self._best_value is None or self._compare(self._best_value, value):
            self._best_value = value
            return True
        return False

    def _init_summary(self):
        if False:
            print('Hello World!')
        self._summary = reporter.DictSummary()

    def serialize(self, serializer):
        if False:
            return 10
        self._interval_trigger.serialize(serializer['interval_trigger'])
        self._summary.serialize(serializer['summary'])
        self._best_value = serializer('best_value', self._best_value)

class MaxValueTrigger(BestValueTrigger):
    """Trigger invoked when specific value becomes maximum.

    For example you can use this trigger to take snapshot on the epoch the
    validation accuracy is maximum.

    Args:
        key (str): Key of value. The trigger fires when the value associated
            with this key becomes maximum.
        trigger: Trigger that decides the comparison interval between current
            best value and new value. This must be a tuple in the form of
            ``<int>, 'epoch'`` or ``<int>, 'iteration'`` which is passed to
            :class:`~chainer.training.triggers.IntervalTrigger`.

    """

    def __init__(self, key, trigger=(1, 'epoch')):
        if False:
            i = 10
            return i + 15
        super(MaxValueTrigger, self).__init__(key, lambda max_value, new_value: new_value > max_value, trigger)

class MinValueTrigger(BestValueTrigger):
    """Trigger invoked when specific value becomes minimum.

    For example you can use this trigger to take snapshot on the epoch the
    validation loss is minimum.

    Args:
        key (str): Key of value. The trigger fires when the value associated
            with this key becomes minimum.
        trigger: Trigger that decides the comparison interval between current
            best value and new value. This must be a tuple in the form of
            ``<int>, 'epoch'`` or ``<int>, 'iteration'`` which is passed to
            :class:`~chainer.training.triggers.IntervalTrigger`.

    """

    def __init__(self, key, trigger=(1, 'epoch')):
        if False:
            print('Hello World!')
        super(MinValueTrigger, self).__init__(key, lambda min_value, new_value: new_value < min_value, trigger)