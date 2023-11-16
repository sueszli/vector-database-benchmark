from coalib.bears.LocalBear import LocalBear

class RaiseTestBear(LocalBear):
    """
    Just raises an exception (default ``RuntimeError``) when run.
    """

    @staticmethod
    def create_arguments(filename, file, config_file):
        if False:
            for i in range(10):
                print('nop')
        return ()

    def run(self, filename, file, cls: str='RuntimeError', msg: str="That's all the RaiseTestBear can do."):
        if False:
            i = 10
            return i + 15
        '\n        Just raise ``cls``.\n        '
        cls = eval(cls)
        raise cls(msg)

class RaiseTestExecuteBear(LocalBear):
    """
    Just raises an exception (default ``RuntimeError``) in execute.
    """

    @staticmethod
    def create_arguments(filename, file, config_file):
        if False:
            return 10
        return ()

    def execute(self, filename, file, debug=False, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Just raise ``cls``.\n        '
        cls = eval(str(self.section['cls']))
        raise cls(self.section['msg'])