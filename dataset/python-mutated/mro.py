"""
MRO stands for method resolution order and it's used by class definitions
to determine which method will be run by a class instance. This module
shows how the MRO is useful for the classic diamond problem where classes
B and C depend on class A, and class D depends on classes B and C.
"""

class BasePlayer:
    """Base player."""

    def ping(self):
        if False:
            i = 10
            return i + 15
        return 'ping'

    def pong(self):
        if False:
            print('Hello World!')
        return 'pong'

class PongPlayer(BasePlayer):
    """Pong player."""

    def pong(self):
        if False:
            for i in range(10):
                print('nop')
        return 'PONG'

class NeutralPlayer(BasePlayer):
    """Neutral player."""

class ConfusedPlayer(PongPlayer, NeutralPlayer):
    """Confused player.

    This is what we call the diamond problem, where `BasePlayer` child classes
    are the same as `ConfusedPlayer` parent classes. Python has the MRO to
    determine which `ping` and `pong` methods are called via the `super()`
    call followed by the respective method.

    The `super()` call is usually used without any parameters, which
    means that we start the MRO process from the current class upwards.

    For more on the subject, please consult this link:

    https://www.python.org/download/releases/2.3/mro/
    """

    def ping(self):
        if False:
            while True:
                i = 10
        'Override `ping` method.'
        return 'pINg'

    def ping_pong(self):
        if False:
            while True:
                i = 10
        'Run `ping` and `pong` in different ways.'
        return [self.ping(), super().ping(), self.pong(), super().pong()]

class IndecisivePlayer(NeutralPlayer, PongPlayer):
    """Indecisive player.

    Notice that this class was created successfully without any conflicts
    even though the MRO of `ConfusedPlayer` is different.

    Notice that one of the `super()` calls uses additional parameters to
    start the MRO process from another class. This is generally discouraged
    as this bypasses the default method resolution process.
    """

    def pong(self):
        if False:
            return 10
        'Override `pong` method.'
        return 'pONg'

    def ping_pong(self):
        if False:
            for i in range(10):
                print('nop')
        'Run `ping` and `pong` in different ways.'
        return [self.ping(), super().ping(), self.pong(), super(PongPlayer, self).pong()]

def main():
    if False:
        for i in range(10):
            print('nop')
    assert ConfusedPlayer.mro() == [ConfusedPlayer, PongPlayer, NeutralPlayer, BasePlayer, object]
    assert IndecisivePlayer.mro() == [IndecisivePlayer, NeutralPlayer, PongPlayer, BasePlayer, object]
    assert ConfusedPlayer().ping_pong() == ['pINg', 'ping', 'PONG', 'PONG']
    assert IndecisivePlayer().ping_pong() == ['ping', 'ping', 'pONg', 'pong']
    class_creation_failed = False
    try:
        type('MissingPlayer', (ConfusedPlayer, IndecisivePlayer), {})
    except TypeError:
        class_creation_failed = True
    assert class_creation_failed is True
if __name__ == '__main__':
    main()