class UpgradeTester:
    """This migration has no upgrade because it is only the enabling of
    pragmas which do not affect database contents.
    """

    def __init__(self, config):
        if False:
            print('Hello World!')
        pass

    def load_data(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def check_upgrade(self):
        if False:
            return 10
        pass

class DowngradeTester:
    """This migration has no downgrade because it is only the enabling of
    pragmas, so we don't need to test the downgrade.
    """

    def __init__(self, config):
        if False:
            return 10
        pass

    def load_data(self):
        if False:
            i = 10
            return i + 15
        pass

    def check_downgrade(self):
        if False:
            for i in range(10):
                print('nop')
        pass