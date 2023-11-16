DESCRIPTION = '\nRuns all the examples for testing purposes and reports successes and failures\nto stderr.  An example is marked successful if the running thread does not\nthrow an exception, for threaded examples, such as plotting, one needs to\ncheck the stderr messages as well.\n'
EPILOG = '\nExample Usage:\n   When no examples fail:\n     $ ./all.py > out\n     SUCCESSFUL:\n       - beginner.basic\n       [...]\n     NO FAILED EXAMPLES\n     $\n\n   When examples fail:\n     $ ./all.py -w > out\n     Traceback (most recent call last):\n       File "./all.py", line 111, in run_examples\n     [...]\n     SUCCESSFUL:\n       - beginner.basic\n       [...]\n     FAILED:\n       - intermediate.mplot2D\n       [...]\n     $\n\n   Obviously, we want to achieve the first result.\n'
import optparse
import os
import sys
import traceback
this_file = os.path.abspath(__file__)
sympy_dir = os.path.join(os.path.dirname(this_file), '..')
sympy_dir = os.path.normpath(sympy_dir)
sys.path.insert(0, sympy_dir)
import sympy
TERMINAL_EXAMPLES = ['beginner.basic', 'beginner.differentiation', 'beginner.expansion', 'beginner.functions', 'beginner.limits_examples', 'beginner.precision', 'beginner.print_pretty', 'beginner.series', 'beginner.substitution', 'intermediate.coupled_cluster', 'intermediate.differential_equations', 'intermediate.infinite_1d_box', 'intermediate.partial_differential_eqs', 'intermediate.trees', 'intermediate.vandermonde', 'advanced.curvilinear_coordinates', 'advanced.dense_coding_example', 'advanced.fem', 'advanced.gibbs_phenomenon', 'advanced.grover_example', 'advanced.hydrogen', 'advanced.pidigits', 'advanced.qft', 'advanced.relativity']
WINDOWED_EXAMPLES = ['beginner.plotting_nice_plot', 'intermediate.mplot2d', 'intermediate.mplot3d', 'intermediate.print_gtk', 'advanced.autowrap_integrators', 'advanced.autowrap_ufuncify', 'advanced.pyglet_plotting']
EXAMPLE_DIR = os.path.dirname(__file__)

def load_example_module(example):
    if False:
        i = 10
        return i + 15
    'Loads modules based upon the given package name'
    from importlib import import_module
    exmod = os.path.split(EXAMPLE_DIR)[1]
    modname = exmod + '.' + example
    return import_module(modname)

def run_examples(*, windowed=False, quiet=False, summary=True):
    if False:
        for i in range(10):
            print('nop')
    'Run all examples in the list of modules.\n\n    Returns a boolean value indicating whether all the examples were\n    successful.\n    '
    successes = []
    failures = []
    examples = TERMINAL_EXAMPLES
    if windowed:
        examples += WINDOWED_EXAMPLES
    if quiet:
        from sympy.testing.runtests import PyTestReporter
        reporter = PyTestReporter()
        reporter.write('Testing Examples\n')
        reporter.write('-' * reporter.terminal_width)
    else:
        reporter = None
    for example in examples:
        if run_example(example, reporter=reporter):
            successes.append(example)
        else:
            failures.append(example)
    if summary:
        show_summary(successes, failures, reporter=reporter)
    return len(failures) == 0

def run_example(example, *, reporter=None):
    if False:
        for i in range(10):
            print('nop')
    'Run a specific example.\n\n    Returns a boolean value indicating whether the example was successful.\n    '
    if reporter:
        reporter.write(example)
    else:
        print('=' * 79)
        print('Running: ', example)
    try:
        mod = load_example_module(example)
        if reporter:
            suppress_output(mod.main)
            reporter.write('[PASS]', 'Green', align='right')
        else:
            mod.main()
        return True
    except KeyboardInterrupt as e:
        raise e
    except:
        if reporter:
            reporter.write('[FAIL]', 'Red', align='right')
        traceback.print_exc()
        return False

class DummyFile:

    def write(self, x):
        if False:
            print('Hello World!')
        pass

def suppress_output(fn):
    if False:
        print('Hello World!')
    'Suppresses the output of fn on sys.stdout.'
    save_stdout = sys.stdout
    try:
        sys.stdout = DummyFile()
        fn()
    finally:
        sys.stdout = save_stdout

def show_summary(successes, failures, *, reporter=None):
    if False:
        i = 10
        return i + 15
    'Shows a summary detailing which examples were successful and which failed.'
    if reporter:
        reporter.write('-' * reporter.terminal_width)
        if failures:
            reporter.write('FAILED:\n', 'Red')
            for example in failures:
                reporter.write('  %s\n' % example)
        else:
            reporter.write('ALL EXAMPLES PASSED\n', 'Green')
    else:
        if successes:
            print('SUCCESSFUL: ', file=sys.stderr)
            for example in successes:
                print('  -', example, file=sys.stderr)
        else:
            print('NO SUCCESSFUL EXAMPLES', file=sys.stderr)
        if failures:
            print('FAILED: ', file=sys.stderr)
            for example in failures:
                print('  -', example, file=sys.stderr)
        else:
            print('NO FAILED EXAMPLES', file=sys.stderr)

def main(*args, **kws):
    if False:
        print('Hello World!')
    'Main script runner'
    parser = optparse.OptionParser()
    parser.add_option('-w', '--windowed', action='store_true', dest='windowed', help='also run examples requiring windowed environment')
    parser.add_option('-q', '--quiet', action='store_true', dest='quiet', help="runs examples in 'quiet mode' suppressing example output and               showing simple status messages.")
    parser.add_option('--no-summary', action='store_true', dest='no_summary', help='hides the summary at the end of testing the examples')
    (options, _) = parser.parse_args()
    return 0 if run_examples(windowed=options.windowed, quiet=options.quiet, summary=not options.no_summary) else 1
if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))