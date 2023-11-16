from .base import NodeTestBase, disable_key_reuse

class ConcentNodeTest(NodeTestBase):

    def test_force_report(self):
        if False:
            i = 10
            return i + 15
        self._run_test('concent.force_report')

    def test_force_download(self):
        if False:
            while True:
                i = 10
        self._run_test('concent.force_download')

    def test_force_accept(self):
        if False:
            while True:
                i = 10
        self._run_test('concent.force_accept')

    def test_additional_verification(self):
        if False:
            for i in range(10):
                print('nop')
        self._run_test('concent.additional_verification')

    @disable_key_reuse
    def test_force_payment(self):
        if False:
            print('Hello World!')
        self._run_test('concent.force_payment')