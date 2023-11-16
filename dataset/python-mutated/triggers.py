from abc import abstractmethod
from bigdl.dllib.utils.common import JavaValue

class ZooTrigger(JavaValue):

    def jvm_class_constructor(self):
        if False:
            for i in range(10):
                print('nop')
        name = 'createZoo' + self.__class__.__name__
        print('creating: ' + name)
        return name

class EveryEpoch(ZooTrigger):
    """
    A trigger specifies a timespot or several timespots during training,
    and a corresponding action will be taken when the timespot(s) is reached.
    EveryEpoch is a trigger that triggers an action when each epoch finishs.
    Could be used as trigger in setvalidation and setcheckpoint in Optimizer,
    and also in TrainSummary.set_summary_trigger.

    >>> everyEpoch = EveryEpoch()
    creating: createZooEveryEpoch
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        '\n        Create a EveryEpoch trigger.\n        '
        ZooTrigger.__init__(self, None, 'float')

class SeveralIteration(ZooTrigger):
    """
    A trigger specifies a timespot or several timespots during training,
    and a corresponding action will be taken when the timespot(s) is reached.
    SeveralIteration is a trigger that triggers an action every "n"
    iterations.
    Could be used as trigger in setvalidation and setcheckpoint in Optimizer,
    and also in TrainSummary.set_summary_trigger.

    >>> serveralIteration = SeveralIteration(2)
    creating: createZooSeveralIteration
    """

    def __init__(self, interval):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a SeveralIteration trigger.\n\n\n        :param interval: interval is the "n" where an action is triggeredevery "n" iterations.\n\n        '
        ZooTrigger.__init__(self, None, 'float', interval)

class MaxEpoch(ZooTrigger):
    """
    A trigger specifies a timespot or several timespots during training,
    and a corresponding action will be taken when the timespot(s) is reached.
    MaxEpoch is a trigger that triggers an action when training reaches
    the number of epochs specified by "max_epoch".
    Usually used as end_trigger when creating an Optimizer.


    >>> maxEpoch = MaxEpoch(2)
    creating: createZooMaxEpoch
    """

    def __init__(self, max):
        if False:
            i = 10
            return i + 15
        '\n        Create a MaxEpoch trigger.\n\n        :param max_epoch: max_epoch\n        '
        ZooTrigger.__init__(self, None, 'float', max)

class MaxIteration(ZooTrigger):
    """
    A trigger specifies a timespot or several timespots during training,
    and a corresponding action will be taken when the timespot(s) is reached.
    MaxIteration is a trigger that triggers an action when training reaches
    the number of iterations specified by "max".
    Usually used as end_trigger when creating an Optimizer.


    >>> maxIteration = MaxIteration(2)
    creating: createZooMaxIteration
    """

    def __init__(self, max):
        if False:
            i = 10
            return i + 15
        '\n        Create a MaxIteration trigger.\n\n\n        :param max: max\n        '
        ZooTrigger.__init__(self, None, 'float', max)

class MaxScore(ZooTrigger):
    """
    A trigger that triggers an action when validation score larger than "max" score.


    >>> maxScore = MaxScore(0.7)
    creating: createZooMaxScore
    """

    def __init__(self, max):
        if False:
            return 10
        '\n        Create a MaxScore trigger.\n\n\n        :param max: max score\n        '
        ZooTrigger.__init__(self, None, 'float', max)

class MinLoss(ZooTrigger):
    """
     A trigger that triggers an action when training loss less than "min" loss.


    >>> minLoss = MinLoss(0.1)
    creating: createZooMinLoss
    """

    def __init__(self, min):
        if False:
            while True:
                i = 10
        '\n        Create a MinLoss trigger.\n\n\n        :param min: min loss\n        '
        ZooTrigger.__init__(self, None, 'float', min)

class TriggerAnd(ZooTrigger):
    """
    A trigger contains other triggers and triggers when all of them trigger (logical AND)


    >>> a = TriggerAnd(MinLoss(0.1), MaxEpoch(2))
    creating: createZooMinLoss
    creating: createZooMaxEpoch
    creating: createZooTriggerAnd
    """

    def __init__(self, first, *other):
        if False:
            print('Hello World!')
        '\n        Create a And trigger.\n\n\n        :param first: first ZooTrigger\n        :param other: other ZooTrigger\n        '
        ZooTrigger.__init__(self, None, 'float', first, list(other))

class TriggerOr(ZooTrigger):
    """
    A trigger contains other triggers and triggers when any of them trigger (logical OR)


    >>> o = TriggerOr(MinLoss(0.1), MaxEpoch(2))
    creating: createZooMinLoss
    creating: createZooMaxEpoch
    creating: createZooTriggerOr
    """

    def __init__(self, first, *other):
        if False:
            print('Hello World!')
        '\n        Create a Or trigger.\n\n\n        :param first: first ZooTrigger\n        :param other: other ZooTrigger\n        '
        ZooTrigger.__init__(self, None, 'float', first, list(other))