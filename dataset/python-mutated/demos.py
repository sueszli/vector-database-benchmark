"""
Interactive demos for the h2o-py library.

:copyright: (c) 2016 H2O.ai
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
from h2o.utils.compatibility import *
import linecache
import os
import sys
import h2o
from h2o.utils.typechecks import assert_is_type

def gbm(interactive=True, echo=True, testing=False):
    if False:
        while True:
            i = 10
    'GBM model demo.'

    def demo_body(go):
        if False:
            print('Hello World!')
        "\n        Demo of H2O's Gradient Boosting estimator.\n\n        This demo uploads a dataset to h2o, parses it, and shows a description.\n        Then it divides the dataset into training and test sets, builds a GLM\n        from the training set, and makes predictions for the test set.\n        Finally, default performance metrics are displayed.\n        "
        go()
        h2o.init()
        go()
        prostate = h2o.load_dataset('prostate')
        go()
        prostate.describe()
        go()
        (train, test) = prostate.split_frame(ratios=[0.7])
        go()
        train['CAPSULE'] = train['CAPSULE'].asfactor()
        test['CAPSULE'] = test['CAPSULE'].asfactor()
        go()
        from h2o.estimators import H2OGradientBoostingEstimator
        prostate_gbm = H2OGradientBoostingEstimator(distribution='bernoulli', ntrees=10, max_depth=8, min_rows=10, learn_rate=0.2)
        prostate_gbm.train(x=['AGE', 'RACE', 'PSA', 'VOL', 'GLEASON'], y='CAPSULE', training_frame=train)
        go()
        prostate_gbm.show()
        go()
        predictions = prostate_gbm.predict(test)
        predictions.show()
        go()
        from h2o.tree import H2OTree, H2ONode
        tree = H2OTree(prostate_gbm, 0, '0')
        len(tree)
        tree.left_children
        tree.right_children
        tree.root_node.show()
        go()
        performance = prostate_gbm.model_performance(test)
        performance.show()
    _run_demo(demo_body, interactive, echo, testing)

def deeplearning(interactive=True, echo=True, testing=False):
    if False:
        for i in range(10):
            print('nop')
    'Deep Learning model demo.'

    def demo_body(go):
        if False:
            for i in range(10):
                print('nop')
        "\n        Demo of H2O's Deep Learning model.\n\n        This demo uploads a dataset to h2o, parses it, and shows a description.\n        Then it divides the dataset into training and test sets, builds a GLM\n        from the training set, and makes predictions for the test set.\n        Finally, default performance metrics are displayed.\n        "
        go()
        h2o.init()
        go()
        prostate = h2o.load_dataset('prostate')
        go()
        prostate.describe()
        go()
        (train, test) = prostate.split_frame(ratios=[0.7])
        go()
        train['CAPSULE'] = train['CAPSULE'].asfactor()
        test['CAPSULE'] = test['CAPSULE'].asfactor()
        go()
        from h2o.estimators import H2ODeepLearningEstimator
        prostate_dl = H2ODeepLearningEstimator(activation='Tanh', hidden=[10, 10, 10], epochs=10000)
        prostate_dl.train(x=list(set(prostate.col_names) - {'ID', 'CAPSULE'}), y='CAPSULE', training_frame=train)
        go()
        prostate_dl.show()
        go()
        predictions = prostate_dl.predict(test)
        predictions.show()
        go()
        performance = prostate_dl.model_performance(test)
        performance.show()
    _run_demo(demo_body, interactive, echo, testing)

def glm(interactive=True, echo=True, testing=False):
    if False:
        for i in range(10):
            print('nop')
    'GLM model demo.'

    def demo_body(go):
        if False:
            return 10
        "\n        Demo of H2O's Generalized Linear Estimator.\n\n        This demo uploads a dataset to h2o, parses it, and shows a description.\n        Then it divides the dataset into training and test sets, builds a GLM\n        from the training set, and makes predictions for the test set.\n        Finally, default performance metrics are displayed.\n        "
        go()
        h2o.init()
        go()
        prostate = h2o.load_dataset('prostate')
        go()
        prostate.describe()
        go()
        (train, test) = prostate.split_frame(ratios=[0.7])
        go()
        train['CAPSULE'] = train['CAPSULE'].asfactor()
        test['CAPSULE'] = test['CAPSULE'].asfactor()
        go()
        from h2o.estimators import H2OGeneralizedLinearEstimator
        prostate_glm = H2OGeneralizedLinearEstimator(family='binomial', alpha=[0.5])
        prostate_glm.train(x=['AGE', 'RACE', 'PSA', 'VOL', 'GLEASON'], y='CAPSULE', training_frame=train)
        go()
        prostate_glm.show()
        go()
        predictions = prostate_glm.predict(test)
        predictions.show()
        go()
        performance = prostate_glm.model_performance(test)
        performance.show()
    _run_demo(demo_body, interactive, echo, testing)

def _run_demo(body_fn, interactive, echo, testing):
    if False:
        i = 10
        return i + 15
    "\n    Execute the demo, echoing commands and pausing for user input.\n\n    :param body_fn: function that contains the sequence of demo's commands.\n    :param interactive: If True, the user will be prompted to continue the demonstration after every segment.\n    :param echo: If True, the python commands that are executed will be displayed.\n    :param testing: Used for pyunit testing. h2o.init() will not be called if set to True.\n    :type body_fn: function\n    "

    class StopExecution(Exception):
        """Helper class for cancelling the demo."""
    assert_is_type(body_fn, type(_run_demo))
    if body_fn.__doc__:
        desc_lines = body_fn.__doc__.split('\n')
        while desc_lines[0].strip() == '':
            desc_lines = desc_lines[1:]
        while desc_lines[-1].strip() == '':
            desc_lines = desc_lines[:-1]
        strip_spaces = min((len(line) - len(line.lstrip(' ')) for line in desc_lines[1:] if line.strip() != ''))
        maxlen = max((len(line) for line in desc_lines))
        print('-' * maxlen)
        for line in desc_lines:
            print(line[strip_spaces:].rstrip())
        print('-' * maxlen)

    def controller():
        if False:
            print('Hello World!')
        'Print to console the next block of commands, and wait for keypress.'
        try:
            raise RuntimeError('Catch me!')
        except RuntimeError:
            print()
            if echo:
                tb = sys.exc_info()[2]
                fr = tb.tb_frame.f_back
                filename = fr.f_code.co_filename
                linecache.checkcache(filename)
                line = linecache.getline(filename, fr.f_lineno, fr.f_globals).rstrip()
                indent_len = len(line) - len(line.lstrip(' '))
                assert line[indent_len:] == 'go()'
                i = fr.f_lineno
                output_lines = []
                n_blank_lines = 0
                while True:
                    i += 1
                    line = linecache.getline(filename, i, fr.f_globals).rstrip()
                    if line[:indent_len].strip() != '':
                        break
                    line = line[indent_len:]
                    if line == 'go()':
                        break
                    prompt = '... ' if line.startswith(' ') else '>>> '
                    output_lines.append(prompt + line)
                    if line.strip() == '':
                        n_blank_lines += 1
                        if n_blank_lines > 5:
                            break
                    else:
                        n_blank_lines = 0
                for line in output_lines[:-n_blank_lines]:
                    print(line)
            if interactive:
                print('\n(press any key)', end='')
                key = _wait_for_keypress()
                print('\r                     \r', end='')
                if key.lower() == 'q':
                    raise StopExecution()
    _h2o_init = h2o.init
    if testing:
        h2o.init = lambda *args, **kwargs: None
    try:
        body_fn(controller)
        print('\n---- End of Demo ----')
    except (StopExecution, KeyboardInterrupt):
        print('\n---- Demo aborted ----')
    if testing:
        h2o.init = _h2o_init
    print()

def _wait_for_keypress():
    if False:
        while True:
            i = 10
    '\n    Wait for a key press on the console and return it.\n\n    Borrowed from http://stackoverflow.com/questions/983354/how-do-i-make-python-to-wait-for-a-pressed-key\n    '
    result = None
    if os.name == 'nt':
        import msvcrt
        result = msvcrt.getch()
    else:
        import termios
        fd = sys.stdin.fileno()
        oldterm = termios.tcgetattr(fd)
        newattr = termios.tcgetattr(fd)
        newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSANOW, newattr)
        try:
            result = sys.stdin.read(1)
        except IOError:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
    return result