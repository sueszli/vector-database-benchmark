from scapy.contrib.automotive import log_automotive
from scapy.contrib.automotive.scanner.graph import _Edge
from scapy.contrib.automotive.ecu import EcuState, EcuResponse, Ecu
from scapy.contrib.automotive.scanner.test_case import AutomotiveTestCaseABC, TestCaseGenerator, StateGenerator, _SocketUnion
from typing import Any, List, Optional, Dict, Callable, cast, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from scapy.contrib.automotive.scanner.test_case import _TransitionTuple
    from scapy.contrib.automotive.scanner.configuration import AutomotiveTestCaseExecutorConfiguration
_TestCaseConnectorCallable = Callable[[AutomotiveTestCaseABC, AutomotiveTestCaseABC], Dict[str, Any]]

class StagedAutomotiveTestCase(AutomotiveTestCaseABC, TestCaseGenerator, StateGenerator):
    """ Helper object to build a pipeline of TestCases. This allows to combine
    TestCases and to execute them after each other. Custom connector functions
    can be used to exchange and manipulate the configuration of a subsequent
    TestCase.

    :param test_cases: A list of objects following the AutomotiveTestCaseABC
        interface
    :param connectors: A list of connector functions. A connector function
        takes two TestCase objects and returns a dictionary which is provided
        to the second TestCase as kwargs of the execute function.


    Example:
        >>> class MyTestCase2(AutomotiveTestCaseABC):
        >>>     pass
        >>>
        >>> class MyTestCase1(AutomotiveTestCaseABC):
        >>>     pass
        >>>
        >>> def connector(testcase1, testcase2):
        >>>     scan_range = len(testcase1.results)
        >>>     return {"verbose": True, "scan_range": scan_range}
        >>>
        >>> tc1 = MyTestCase1()
        >>> tc2 = MyTestCase2()
        >>> pipeline = StagedAutomotiveTestCase([tc1, tc2], [None, connector])
    """
    __delay_stages = 5

    def __init__(self, test_cases, connectors=None):
        if False:
            return 10
        super(StagedAutomotiveTestCase, self).__init__()
        self.__test_cases = test_cases
        self.__connectors = connectors
        self.__stage_index = 0
        self.__completion_delay = 0
        self.__current_kwargs = None

    def __getitem__(self, item):
        if False:
            print('Hello World!')
        return self.__test_cases[item]

    def __len__(self):
        if False:
            return 10
        return len(self.__test_cases)

    def __reduce__(self):
        if False:
            while True:
                i = 10
        (f, t, d) = super(StagedAutomotiveTestCase, self).__reduce__()
        try:
            del d['_StagedAutomotiveTestCase__connectors']
        except KeyError:
            pass
        return (f, t, d)

    @property
    def test_cases(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__test_cases

    @property
    def current_test_case(self):
        if False:
            return 10
        return self[self.__stage_index]

    @property
    def current_connector(self):
        if False:
            return 10
        if not self.__connectors:
            return None
        else:
            return self.__connectors[self.__stage_index]

    @property
    def previous_test_case(self):
        if False:
            print('Hello World!')
        return self.__test_cases[self.__stage_index - 1] if self.__stage_index > 0 else None

    def get_generated_test_case(self):
        if False:
            while True:
                i = 10
        try:
            test_case = cast(TestCaseGenerator, self.current_test_case)
            return test_case.get_generated_test_case()
        except AttributeError:
            return None

    def get_new_edge(self, socket, config):
        if False:
            while True:
                i = 10
        try:
            test_case = cast(StateGenerator, self.current_test_case)
            return test_case.get_new_edge(socket, config)
        except AttributeError:
            return None

    def get_transition_function(self, socket, edge):
        if False:
            for i in range(10):
                print('nop')
        try:
            test_case = cast(StateGenerator, self.current_test_case)
            return test_case.get_transition_function(socket, edge)
        except AttributeError:
            return None

    def has_completed(self, state):
        if False:
            for i in range(10):
                print('nop')
        if not (self.current_test_case.has_completed(state) and self.current_test_case.completed):
            self.__completion_delay = 0
            return False
        if self.__completion_delay < StagedAutomotiveTestCase.__delay_stages:
            self.__completion_delay += 1
            return False
        elif self.__stage_index == len(self.__test_cases) - 1:
            return True
        else:
            log_automotive.info('Staged AutomotiveTestCase %s completed', self.current_test_case.__class__.__name__)
            self.__stage_index += 1
            self.__completion_delay = 0
        return False

    def pre_execute(self, socket, state, global_configuration):
        if False:
            print('Hello World!')
        test_case_cls = self.current_test_case.__class__
        try:
            self.__current_kwargs = global_configuration[test_case_cls.__name__]
        except KeyError:
            self.__current_kwargs = dict()
            global_configuration[test_case_cls.__name__] = self.__current_kwargs
        if callable(self.current_connector) and self.__stage_index > 0:
            if self.previous_test_case:
                con = self.current_connector
                con_kwargs = con(self.previous_test_case, self.current_test_case)
                if self.__current_kwargs is not None and con_kwargs is not None:
                    self.__current_kwargs.update(con_kwargs)
        log_automotive.debug('Stage AutomotiveTestCase %s kwargs: %s', self.current_test_case.__class__.__name__, self.__current_kwargs)
        self.current_test_case.pre_execute(socket, state, global_configuration)

    def execute(self, socket, state, **kwargs):
        if False:
            i = 10
            return i + 15
        kwargs.update(self.__current_kwargs or dict())
        self.current_test_case.execute(socket, state, **kwargs)

    def post_execute(self, socket, state, global_configuration):
        if False:
            i = 10
            return i + 15
        self.current_test_case.post_execute(socket, state, global_configuration)

    @staticmethod
    def _show_headline(headline, sep='='):
        if False:
            return 10
        s = '\n\n' + sep * (len(headline) + 10) + '\n'
        s += ' ' * 5 + headline + '\n'
        s += sep * (len(headline) + 10) + '\n'
        return s + '\n'

    def show(self, dump=False, filtered=True, verbose=False):
        if False:
            i = 10
            return i + 15
        s = self._show_headline('AutomotiveTestCase Pipeline', '=')
        for (idx, t) in enumerate(self.__test_cases):
            s += self._show_headline('AutomotiveTestCase Stage %d' % idx, '-')
            s += t.show(True, filtered, verbose) or ''
        if dump:
            return s + '\n'
        else:
            print(s)
            return None

    @property
    def completed(self):
        if False:
            print('Hello World!')
        return all((e.completed for e in self.__test_cases)) and self.__completion_delay >= StagedAutomotiveTestCase.__delay_stages

    @property
    def supported_responses(self):
        if False:
            i = 10
            return i + 15
        supported_responses = list()
        for tc in self.test_cases:
            supported_responses += tc.supported_responses
        supported_responses.sort(key=Ecu.sort_key_func)
        return supported_responses

    def runtime_estimation(self):
        if False:
            i = 10
            return i + 15
        if hasattr(self.current_test_case, 'runtime_estimation'):
            cur_est = self.current_test_case.runtime_estimation()
            if cur_est:
                return (len(self.test_cases), self.__stage_index, float(self.__stage_index) / len(self.test_cases) + cur_est[2] / len(self.test_cases))
        return (len(self.test_cases), self.__stage_index, float(self.__stage_index) / len(self.test_cases))