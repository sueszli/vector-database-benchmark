import codecs
import six
from rqalpha.interface import AbstractStrategyLoader
from rqalpha.utils.strategy_loader_help import compile_strategy

class FileStrategyLoader(AbstractStrategyLoader):

    def __init__(self, strategy_file_path):
        if False:
            return 10
        self._strategy_file_path = strategy_file_path

    def load(self, scope):
        if False:
            print('Hello World!')
        with codecs.open(self._strategy_file_path, encoding='utf-8') as f:
            source_code = f.read()
        return compile_strategy(source_code, self._strategy_file_path, scope)

class SourceCodeStrategyLoader(AbstractStrategyLoader):

    def __init__(self, code):
        if False:
            i = 10
            return i + 15
        self._code = code

    def load(self, scope):
        if False:
            i = 10
            return i + 15
        return compile_strategy(self._code, 'strategy.py', scope)

class UserFuncStrategyLoader(AbstractStrategyLoader):

    def __init__(self, user_funcs):
        if False:
            for i in range(10):
                print('nop')
        self._user_funcs = user_funcs

    def load(self, scope):
        if False:
            print('Hello World!')
        for user_func in six.itervalues(self._user_funcs):
            user_func.__globals__.update(scope)
        return self._user_funcs