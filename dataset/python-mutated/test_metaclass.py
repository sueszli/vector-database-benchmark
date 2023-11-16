import testcommon

class TestFrompackages(testcommon.SampleParamsHolder):
    """
    This class is used for testing that inheriting from base class that
    uses `frompackages` import mechanism, doesnt brake the functionality
    of the base class.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        super(TestFrompackages, self).__init__()

def test_run(main=False):
    if False:
        while True:
            i = 10
    '\n    Instantiate the TestFrompackages and see that no exception is raised\n    Bug Discussion:\n    https://community.backtrader.com/topic/2661/frompackages-directive-functionality-seems-to-be-broken-when-using-inheritance\n    '
    test = TestFrompackages()
if __name__ == '__main__':
    test_run(main=True)