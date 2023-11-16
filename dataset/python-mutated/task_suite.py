import sys
import logging
from typing import Type, Optional, Dict, Any, Callable, List, Iterable, Union, TextIO, Tuple
import numpy as np
from checklist.test_suite import TestSuite
from checklist.editor import Editor
from checklist.test_types import MFT, INV, DIR
from checklist.perturb import Perturb
from allennlp.common.registrable import Registrable
from allennlp.common.file_utils import cached_path
from allennlp.predictors.predictor import Predictor
from allennlp.confidence_checks.task_checklists import utils
logger = logging.getLogger(__name__)

class TaskSuite(Registrable):
    """
    Base class for various task test suites.

    This is a wrapper class around the CheckList toolkit introduced
    in the paper
    [Beyond Accuracy: Behavioral Testing of NLP models with CheckList (Ribeiro et al)]
    (https://api.semanticscholar.org/CorpusID:218551201).

    !!! Note
        To use the checklist integration you should install ``allennlp`` with the
        "checklist" extra (e.g. ``conda install allennlp-checklist``,
        ``pip install allennlp[checklist]`` or just install `checklist` after the fact).

    Task suites are intended to be used as a form of behavioral testing
    for NLP models to check for robustness across several general linguistic
    capabilities; eg. Vocabulary, SRL, Negation, etc.

    An example of the entire checklist process can be found at:
    [https://github.com/marcotcr/checklist/blob/master/notebooks/tutorials/]
    (https://github.com/marcotcr/checklist/blob/master/notebooks/tutorials/).

    A task suite should contain tests that check general capabilities, including
    but not limited to:

    * Vocabulary + POS : Important words/word types for the task
    * Taxonomy : Synonyms/antonyms, etc.
    * Robustness : To typos, irrelevant changes, etc.
    * NER : Appropriately understanding named entities.
    * Temporal : Understanding the order of events.
    * Negation
    * Coreference
    * Semantic Role Labeling : Understanding roles such as agents and objects.
    * Logic : Ability to handle symmetry, consistency, and conjunctions.
    * Fairness


    # Parameters

    suite: `checklist.test_suite.TestSuite`, optional (default = `None`)
        Pass in an existing test suite.

    add_default_tests: `bool` (default = `False`)
        Whether to add default checklist tests for the task.

    data: `List[Any]`, optional (default = `None`)
        If the data is provided, and `add_default_tests` is `True`,
        tests that perturb the data are also added.

        For instance, if the task is sentiment analysis, and the a
        list of sentences is passed, it will add tests that check
        a model's robustness to typos, etc.
    """
    _capabilities: List[str] = ['Vocabulary', 'Taxonomy', 'Robustness', 'NER', 'Fairness', 'Temporal', 'Negation', 'Coref', 'SRL', 'Logic']

    def __init__(self, suite: Optional[TestSuite]=None, add_default_tests: bool=True, data: Optional[List[Any]]=None, num_test_cases: int=100, **kwargs):
        if False:
            print('Hello World!')
        self.suite = suite or TestSuite()
        if add_default_tests:
            self._default_tests(data, num_test_cases)

    def _prediction_and_confidence_scores(self, predictor: Predictor) -> Callable:
        if False:
            i = 10
            return i + 15
        '\n        This makes certain assumptions about the task predictor\n        input and output expectations. This should return a function\n        that takes the data as input, passes it to the predictor,\n        and returns predictions and confidences.\n        '
        return NotImplementedError

    def describe(self):
        if False:
            i = 10
            return i + 15
        '\n        Gives a description of the test suite. This is intended as a utility for\n        examining the test suite.\n        '
        self._summary(overview_only=True)

    def summary(self, capabilities: Optional[List[str]]=None, file: TextIO=sys.stdout, **kwargs):
        if False:
            return 10
        '\n        Prints a summary of the test results.\n\n        # Parameters\n\n        capabilities : `List[str]`, optional (default = `None`)\n            If not None, will only show tests with these capabilities.\n        **kwargs : `type`\n            Will be passed as arguments to each test.summary()\n        '
        old_stdout = sys.stdout
        try:
            sys.stdout = file
            self._summary(capabilities=capabilities, **kwargs)
        finally:
            sys.stdout = old_stdout

    def _summary(self, overview_only: bool=False, capabilities: Optional[List[str]]=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Internal function for description and summary.\n        '

        def cap_order(x):
            if False:
                print('Hello World!')
            return self._capabilities.index(x) if x in self._capabilities else 100
        capabilities = capabilities or sorted(set([x['capability'] for x in self.suite.info.values()]), key=cap_order)
        print('\n\nThis suite contains {} tests across {} capabilities.'.format(len(self.suite.tests), len(capabilities)))
        print()
        for capability in capabilities:
            tests = [name for (name, test) in self.suite.info.items() if test['capability'] == capability]
            num_tests = len(tests)
            if num_tests > 0:
                print(f'\nCapability: "{capability}" ({num_tests} tests)\n')
                for test in tests:
                    description = self.suite.info[test]['description']
                    num_test_cases = len(self.suite.tests[test].data)
                    about_test = f'* Name: {test} ({num_test_cases} test cases)'
                    if description:
                        about_test += f'\n{description}'
                    print(about_test)
                    if not overview_only:
                        if 'format_example_fn' not in kwargs:
                            kwargs['format_example_fn'] = self.suite.info[test].get('format_example_fn', self._format_failing_examples)
                        if 'print_fn' not in kwargs:
                            kwargs['print_fn'] = self.suite.info[test].get('print_fn', self.suite.print_fn)
                        print()
                        self.suite.tests[test].summary(**kwargs)
                        print()

    def _format_failing_examples(self, inputs: Tuple[Any], pred: Any, conf: Union[np.array, np.ndarray], *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Formatting function for printing failed test examples.\n        '
        if conf.shape[0] <= 4:
            confs = ' '.join(['%.1f' % c for c in conf])
            ret = '%s %s' % (confs, str(inputs))
        else:
            conf = conf[pred]
            ret = '%s (%.1f) %s' % (pred, conf, str(inputs))
        return ret

    def run(self, predictor: Predictor, capabilities: Optional[List[str]]=None, max_examples: Optional[int]=None):
        if False:
            i = 10
            return i + 15
        '\n        Runs the predictor on the test suite data.\n\n        # Parameters\n\n        predictor : `Predictor`\n            The predictor object.\n        capabilities : `List[str]`, optional (default = `None`)\n            If not None, will only run tests with these capabilities.\n        max_examples : `int`, optional (default = `None`)\n            Maximum number of examples to run. If None, all examples will be run.\n        '
        preds_and_confs_fn = self._prediction_and_confidence_scores(predictor)
        if preds_and_confs_fn is NotImplementedError:
            raise NotImplementedError('The `_prediction_and_confidence_scores` function needs to be implemented for the class `{}`'.format(self.__class__))
        if not capabilities:
            self.suite.run(preds_and_confs_fn, overwrite=True, n=max_examples)
        else:
            for (_, test) in self.suite.tests.items():
                if test.capability in capabilities:
                    test.run(preds_and_confs_fn, verbose=True, overwrite=True, n=max_examples)

    @classmethod
    def constructor(cls, name: Optional[str]=None, suite_file: Optional[str]=None, extra_args: Optional[Dict[str, Any]]=None) -> 'TaskSuite':
        if False:
            i = 10
            return i + 15
        suite_class: Type[TaskSuite] = TaskSuite.by_name(name) if name is not None else cls
        if extra_args is None:
            extra_args = {}
        if suite_file is not None:
            return suite_class(TestSuite.from_file(cached_path(suite_file)), **extra_args)
        return suite_class(**extra_args)

    def save_suite(self, suite_file: str):
        if False:
            return 10
        '\n        Saves the suite to a file.\n        '
        self.suite.save(suite_file)

    def _default_tests(self, data: Optional[Iterable], num_test_cases: int=100):
        if False:
            while True:
                i = 10
        '\n        Derived TaskSuite classes can add any task-specific tests here.\n        '
        if data:
            self._punctuation_test(data, num_test_cases)
            self._typo_test(data, num_test_cases)
            self._contraction_test(data, num_test_cases)

    @classmethod
    def contractions(cls) -> Callable:
        if False:
            print('Hello World!')
        '\n        This returns a function which adds/removes contractions in relevant\n        `str` inputs of a task\'s inputs. For instance, "isn\'t" will be\n        changed to "is not", and "will not" will be changed to "won\'t".\n\n        Expected arguments for this function: `(example, **args, **kwargs)`\n        where the `example` is an instance of some task. It can be of any\n        type.\n\n        For example, for a sentiment analysis task, it will be a\n        a `str` (the sentence for which we want to predict the sentiment).\n        For a textual entailment task, it can be a tuple or a Dict, etc.\n\n        Expected output of this function is a list of instances for the task,\n        of the same type as `example`.\n        '
        return Perturb.contractions

    @classmethod
    def typos(cls) -> Callable:
        if False:
            print('Hello World!')
        "\n        This returns a function which adds simple typos in relevant\n        `str` inputs of a task's inputs.\n\n        Expected arguments for this function: `(example, **args, **kwargs)`\n        where the `example` is an instance of some task. It can be of any\n        type.\n\n        For example, for a sentiment analysis task, it will be a\n        a `str` (the sentence for which we want to predict the sentiment).\n        For a textual entailment task, it can be a tuple or a Dict, etc.\n\n        Expected output of this function is a list of instances for the task,\n        of the same type as `example`.\n        "
        return Perturb.add_typos

    @classmethod
    def punctuation(cls) -> Callable:
        if False:
            while True:
                i = 10
        '\n        This returns a function which adds/removes punctuations in relevant\n        `str` inputs of a task\'s inputs. For instance, "isn\'t" will be\n        changed to "is not", and "will not" will be changed to "won\'t".\n\n        Expected arguments for this function: `(example, **args, **kwargs)`\n        where the `example` is an instance of some task. It can be of any\n        type.\n\n        For example, for a sentiment analysis task, it will be a\n        a `str` (the sentence for which we want to predict the sentiment).\n        For a textual entailment task, it can be a tuple or a Dict, etc.\n\n        Expected output of this function is a list of instances for the task,\n        of the same type as `example`.\n        '
        return utils.toggle_punctuation

    def _punctuation_test(self, data: Iterable, num_test_cases: int):
        if False:
            for i in range(10):
                print('nop')
        '\n        Checks if the model is invariant to presence/absence of punctuation.\n        '
        template = Perturb.perturb(data, self.punctuation(), nsamples=num_test_cases)
        test = INV(template.data, name='Punctuation', description="Strip punctuation and / or add '.'", capability='Robustness')
        self.add_test(test)

    def _typo_test(self, data: Iterable, num_test_cases: int):
        if False:
            while True:
                i = 10
        '\n        Checks if the model is robust enough to be invariant to simple typos.\n        '
        template = Perturb.perturb(data, self.typos(), nsamples=num_test_cases, typos=1)
        test = INV(template.data, name='Typos', capability='Robustness', description='Add one typo to input by swapping two adjacent characters')
        self.add_test(test)
        template = Perturb.perturb(data, self.typos(), nsamples=num_test_cases, typos=2)
        test = INV(template.data, name='2 Typos', capability='Robustness', description='Add two typos to input by swapping two adjacent characters twice')
        self.add_test(test)

    def _contraction_test(self, data: Iterable, num_test_cases: int):
        if False:
            i = 10
            return i + 15
        "\n        Checks if the model is invariant to contractions and expansions\n        (eg. What is <-> What's).\n        "
        template = Perturb.perturb(data, self.contractions(), nsamples=num_test_cases)
        test = INV(template.data, name='Contractions', capability='Robustness', description="Contract or expand contractions, e.g. What is <-> What's")
        self.add_test(test)

    def _setup_editor(self):
        if False:
            while True:
                i = 10
        '\n        Sets up a `checklist.editor.Editor` object, to be used for adding\n        default tests to the suite.\n        '
        if not hasattr(self, 'editor'):
            self.editor = Editor()
            utils.add_common_lexicons(self.editor)

    def add_test(self, test: Union[MFT, INV, DIR]):
        if False:
            print('Hello World!')
        '\n        Adds a fully specified checklist test to the suite.\n        The tests can be of the following types:\n\n        * MFT: A minimum functionality test. It checks if the predicted output\n               matches the expected output.\n               For example, for a sentiment analysis task, a simple MFT can check\n               if the model always predicts a positive sentiment for very\n               positive words.\n               The test\'s data contains the input and the expected output.\n\n        * INV: An invariance test. It checks if the predicted output is invariant\n               to some change in the input.\n               For example, for a sentiment analysis task, an INV test can check\n               if the prediction stays consistent if simple typos are added.\n               The test\'s data contains the pairs (input, modified input).\n\n        * DIR: A directional expectation test. It checks if the predicted output\n               changes in some specific way in response to the change in input.\n               For example, for a sentiment analysis task, a DIR test can check if\n               adding a reducer (eg. "good" -> "somewhat good") causes the\n               prediction\'s positive confidence score to decrease (or at least not\n               increase).\n               The test\'s data contains single inputs or pairs (input, modified input).\n\n        Please refer to [the paper](https://api.semanticscholar.org/CorpusID:218551201)\n        for more details and examples.\n\n        Note: `test` needs to be fully specified; with name, capability and description.\n        '
        if test.data:
            self.suite.add(test)
        else:
            logger.warning("'{}' was not added, as it contains no examples.".format(test.name))