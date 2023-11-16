from golem.network.concent import soft_switch
from golem.tools.testwithdatabase import TestWithDatabase

class ConcentSwitchTestMixin:

    @property
    def _default(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def _turn(self, on: bool):
        if False:
            return 10
        raise NotImplementedError()

    def _is_on(self):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def test_default_value(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self._is_on(), self._default)

    def test_turn_on(self):
        if False:
            print('Hello World!')
        self._turn(False)
        self._turn(True)
        self.assertTrue(self._is_on())

    def test_turn_off(self):
        if False:
            for i in range(10):
                print('nop')
        self._turn(True)
        self._turn(False)
        self.assertFalse(self._is_on())

class TestConcentSoftSwitch(ConcentSwitchTestMixin, TestWithDatabase):

    @property
    def _default(self):
        if False:
            i = 10
            return i + 15
        return False

    def _turn(self, on: bool):
        if False:
            for i in range(10):
                print('nop')
        return soft_switch.concent_turn(on)

    def _is_on(self):
        if False:
            return 10
        return soft_switch.concent_is_on()

class TestConcentRequiredAsProvider(ConcentSwitchTestMixin, TestWithDatabase):

    @property
    def _default(self):
        if False:
            for i in range(10):
                print('nop')
        return True

    def _turn(self, on: bool):
        if False:
            while True:
                i = 10
        return soft_switch.required_as_provider_turn(on)

    def _is_on(self):
        if False:
            while True:
                i = 10
        return soft_switch.is_required_as_provider()