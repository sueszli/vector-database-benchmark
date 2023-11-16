import operator
import random
import time
from inspect import signature
from itertools import chain, count, islice, repeat
from types import SimpleNamespace
from about_time import about_time
from .utils import fix_signature
from ..utils import terminal
from ..utils.cells import fix_cells, is_wide, join_cells, strip_marks, to_cells
from ..utils.colors import BLUE, BLUE_BOLD, CYAN, DIM, GREEN, ORANGE, ORANGE_BOLD, RED, YELLOW_BOLD

def spinner_controller(*, natural, skip_compiler=False):
    if False:
        return 10

    def inner_controller(spinner_inner_factory, op_params=None, extra_commands=None):
        if False:
            for i in range(10):
                print('nop')

        def spinner_compiler_dispatcher_factory(actual_length=None):
            if False:
                print('Hello World!')
            'Compile this spinner factory into an actual spinner runner.\n            The previous parameters were the styling parameters, which defined a style.\n            These are called operational parameters, which `alive_progress` binds dynamically\n            as needed. Do not call this manually.\n\n            Args:\n                actual_length (int): the actual length to compile the frames renditions\n\n            Returns:\n                a spinner runner\n\n            '
            if skip_compiler:
                return spinner_inner_factory(actual_length, **op_params)
            with about_time() as t_compile:
                gen = spinner_inner_factory(actual_length, **op_params)
                spec = spinner_compiler(gen, natural, extra_commands.get(True, ()))
            return spinner_runner_factory(spec, t_compile, extra_commands.get(False, ()))

        def compile_and_check(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            'Compile this spinner factory at its natural length, and...'
            spinner_compiler_dispatcher_factory().check(*args, **kwargs)

        def set_operational(**params):
            if False:
                while True:
                    i = 10
            signature(spinner_inner_factory).bind(1, **params)
            return inner_controller(spinner_inner_factory, params, extra_commands)

        def schedule_command(command):
            if False:
                for i in range(10):
                    print('nop')

            def inner_schedule(*args, **kwargs):
                if False:
                    while True:
                        i = 10
                signature(command).bind(1, *args, **kwargs)
                (extra, cmd_type) = (dict(extra_commands), EXTRA_COMMANDS[command])
                extra[cmd_type] = extra.get(cmd_type, ()) + ((command, args, kwargs),)
                return inner_controller(spinner_inner_factory, op_params, extra)
            return fix_signature(inner_schedule, command, 1)
        spinner_compiler_dispatcher_factory.__dict__.update(check=fix_signature(compile_and_check, check, 1), op=set_operational, **{c.__name__: schedule_command(c) for c in EXTRA_COMMANDS})
        (op_params, extra_commands) = (op_params or {}, extra_commands or {})
        spinner_compiler_dispatcher_factory.natural = natural
        return spinner_compiler_dispatcher_factory
    return inner_controller
'\nThe commands here are made available in the compiler controller, thus in all spinners.\n\nThey work lazily: when called they only schedule themselves to be run when the spinner\ngets compiled, i.e., when it receives the operational parameters like `actual_length`.\n\nThey can take place inside the compiler or inside the runner.\nCompiler commands can change the data at will, before the animation specs are computed.\nRunner commands can only change presentation order.\n'

def extra_command(is_compiler):
    if False:
        while True:
            i = 10

    def inner_command(command):
        if False:
            for i in range(10):
                print('nop')
        EXTRA_COMMANDS[command] = is_compiler
        return command
    return inner_command
EXTRA_COMMANDS = {}
(compiler_command, runner_command) = (extra_command(True), extra_command(False))

@compiler_command
def replace(spec, old, new):
    if False:
        while True:
            i = 10
    'Replace a portion of the frames by another with the same length.\n\n    Args:\n        old (str): the old string to be replaced\n        new (str): the new string\n\n    '
    spec.data = tuple((tuple((to_cells(join_cells(frame).replace(old, new)) for frame in cycle)) for cycle in spec.data))

@compiler_command
def pause(spec, edges=None, center=None, other=None):
    if False:
        for i in range(10):
            print('nop')
    "Make the animation appear to pause at the edges or at the middle, or make it slower as\n    a whole, or both.\n\n    Use without arguments to get their defaults, which gives a small pause at the edges,\n    very nice for bouncing text with `hide=False`. Please note that the defaults only apply\n    if none of the params are set.\n\n    In the future, I'd like to make this a `pace` command, which would receive a sequence\n    of ints of any length, and apply it bouncing across the cycle. For example to smoothly\n    decelerate it could be (6, 3, 2, 1), which would become (6, 3, 2, 1, 1, ..., 1, 2, 3, 6).\n\n    Args:\n        edges (Optional[int]): how many times the first and last frames of a cycle repeats\n            default is 8.\n        center (Optional[int]): how many times the middle frame of a cycle repeats\n            default is 1.\n        other (Optional[int]): how many times all the other frames of a cycle repeats\n            default is 1.\n\n    "
    (edges, center, other) = (max(1, x or 1) for x in (edges, center, other))
    if all((x == 1 for x in (edges, center, other))):
        (edges, center, other) = (8, 1, 1)

    def repeats_func(length):
        if False:
            while True:
                i = 10
        return {0: edges, length - 1: edges, round(length / 2): center}
    spec.data = tuple((tuple(chain.from_iterable((repeat(frame, repeats.get(i) or other) for (i, frame) in enumerate(cycle)))) for (cycle, repeats) in ((cycle, repeats_func(len(cycle))) for cycle in spec.data)))

@compiler_command
def reshape(spec, num_frames):
    if False:
        i = 10
        return i + 15
    'Reshape frame data into another grouping. It can be used to simplify content\n    description, or for artistic effects.\n\n    Args:\n        num_frames (int): the number of consecutive frames to group\n\n    '
    flatten = chain.from_iterable((cycle for cycle in spec.data))
    spec.data = tuple(iter(lambda : tuple(islice(flatten, num_frames)), ()))

@compiler_command
def bounce(spec):
    if False:
        return 10
    'Make the animation bounce its cycles.'
    spec.data = tuple(chain(spec.data, spec.data[-2:0:-1]))

@compiler_command
def transpose(spec):
    if False:
        return 10
    'Transpose the frame content matrix, exchanging columns for rows. It can be used\n    to simplify content description, or for artistic effects.'
    spec.data = tuple((tuple(cycle) for cycle in zip(*spec.data)))

@runner_command
def sequential(spec):
    if False:
        i = 10
        return i + 15
    'Configure the runner to play the compiled cycles in sequential order.'

    def cycle_data(data):
        if False:
            print('Hello World!')
        while True:
            yield from data
    cycle_data.name = 'sequential'
    spec.__dict__.update(strategy=cycle_data, cycles=len(spec.data))

@runner_command
def randomize(spec, cycles=None):
    if False:
        print('Hello World!')
    'Configure the runner to play the compiled cycles in random order.\n\n    Args:\n        cycles (Optional[int]): number of cycles to play randomized\n\n    '

    def cycle_data(data):
        if False:
            i = 10
            return i + 15
        while True:
            yield random.choice(data)
    cycle_data.name = 'randomized'
    spec.__dict__.update(strategy=cycle_data, cycles=max(0, cycles or 0) or spec.cycles)

def apply_extra_commands(spec, extra_commands):
    if False:
        while True:
            i = 10
    for (command, args, kwargs) in extra_commands:
        command(spec, *args, **kwargs)

def spinner_compiler(gen, natural, extra_commands):
    if False:
        while True:
            i = 10
    'Optimized spinner compiler, which compiles ahead of time all frames of all cycles\n    of a spinner.\n\n    Args:\n        gen (Generator): the generator expressions that defines the cycles and their frames\n        natural (int): the natural length of the spinner\n        extra_commands (tuple[tuple[cmd, list[Any], dict[Any]]]): requested extra commands\n\n    Returns:\n        the spec of a compiled animation\n\n    '
    spec = SimpleNamespace(data=tuple((tuple((fix_cells(frame) for frame in cycle)) for cycle in gen)), natural=natural)
    apply_extra_commands(spec, extra_commands)
    frames = tuple((len(cycle) for cycle in spec.data))
    spec.__dict__.update(cycles=len(spec.data), length=len(spec.data[0][0]), frames=frames, total_frames=sum(frames))
    assert max((len(frame) for cycle in spec.data for frame in cycle)) == min((len(frame) for cycle in spec.data for frame in cycle)), render_data(spec, True) or 'Different cell lengths detected in frame data.'
    return spec

def spinner_runner_factory(spec, t_compile, extra_commands):
    if False:
        print('Hello World!')
    'Optimized spinner runner, which receives the spec of an animation, and controls\n    the flow of cycles and frames already compiled to a certain screen length and with\n    wide chars fixed, thus avoiding any overhead in runtime within complex spinners,\n    while allowing their factories to be garbage collected.\n\n    Args:\n        spec (SimpleNamespace): the spec of an animation\n        t_compile (about_time.Handler): the compile time information\n        extra_commands (tuple[tuple[cmd, list[Any], dict[Any]]]): requested extra commands\n\n    Returns:\n        a spinner runner\n\n    '

    def spinner_runner():
        if False:
            return 10
        'Wow, you are really deep! This is the runner of a compiled spinner.\n        Every time you call this function, a different generator will kick in,\n        which yields the frames of the current animation cycle. Enjoy!'
        yield from next(cycle_gen)

    def runner_check(*args, **kwargs):
        if False:
            print('Hello World!')
        return check(spec, *args, **kwargs)
    spinner_runner.__dict__.update(spec.__dict__, check=fix_signature(runner_check, check, 1))
    spec.__dict__.update(t_compile=t_compile, runner=spinner_runner)
    sequential(spec)
    apply_extra_commands(spec, extra_commands)
    cycle_gen = spec.strategy(spec.data)
    return spinner_runner

def check(spec, verbosity=0):
    if False:
        return 10
    'Check the specs, contents, codepoints, and even the animation of this compiled spinner.\n    \n    Args:\n        verbosity (int): change the verbosity level\n                             0 for specs only (default)\n                               /                 \\\n                              /           3 to include animation\n                             /                      \\\n            1 to unfold frame data   --------   4 to unfold frame data\n                            |                        |\n            2 to reveal codepoints   --------   5 to reveal codepoints\n\n    '
    verbosity = max(0, min(5, verbosity or 0))
    if verbosity in (1, 2, 4, 5):
        render_data(spec, verbosity in (2, 5))
    spec_data(spec)
    duration = spec.t_compile.duration_human
    print(f'\nSpinner frames compiled in: {GREEN(duration)}')
    print(f'(call {HELP_MSG[verbosity]})')
    if verbosity in (3, 4, 5):
        animate(spec)

def __check(p):
    if False:
        while True:
            i = 10
    return f"{BLUE(f'.{check.__name__}(')}{BLUE_BOLD(p)}{BLUE(')')}"
SECTION = ORANGE_BOLD
HELP_MSG = {0: f'{__check(1)} to unfold frame data, or {__check(3)} to include animation', 1: f'{__check(2)} to reveal codepoints, or {__check(4)} to include animation, or {__check(0)} to fold up frame data', 2: f'{__check(5)} to include animation, or {__check(1)} to hide codepoints', 3: f'{__check(4)} to unfold frame data, or {__check(0)} to omit animation', 4: f'{__check(5)} to reveal codepoints, or {__check(1)} to omit animation, or {__check(3)} to fold up frame data', 5: f'{__check(2)} to omit animation, or {__check(4)} to hide codepoints'}

def spec_data(spec):
    if False:
        return 10

    def info(field):
        if False:
            print('Hello World!')
        return f"{YELLOW_BOLD(field.split('.')[0])}: {operator.attrgetter(field)(spec)}"
    print(f"\n{SECTION('Specs')}")
    print(info('length'), f"({info('natural')})")
    print(info('cycles'), f"({info('strategy.name')})")
    print('\n'.join((info(field) for field in ('frames', 'total_frames'))))

def format_codepoints(frame):
    if False:
        print('Hello World!')
    codes = '|'.join(((ORANGE if is_wide(g) else BLUE)(' '.join((hex(ord(c)).replace('0x', '') for c in g))) for g in frame))
    return f' -> {RED(sum((len(fragment) for fragment in frame)))}:[{codes}]'

def render_data(spec, show_codepoints):
    if False:
        for i in range(10):
            print('nop')
    print(f"\n{SECTION('Frame data')}", end='')
    whole_index = count(1)
    (lf, wf) = (f'>{1 + len(str(max(spec.frames)))}', f'<{len(str(spec.total_frames))}')
    codepoints = format_codepoints if show_codepoints else lambda _: ''
    for (i, cycle) in enumerate(spec.data, 1):
        frames = map(lambda fragment: tuple(strip_marks(fragment)), cycle)
        print(f'\ncycle {i}\n' + '\n'.join((DIM(li, lf) + f" |{''.join(frame)}| {DIM(wi, wf)}" + codepoints(frame) for (li, frame, wi) in zip(count(1), frames, whole_index))))

def animate(spec):
    if False:
        while True:
            i = 10
    print(f"\n{SECTION('Animation')}")
    (cf, lf, tf) = (f'>{len(str(x))}' for x in (spec.cycles, max(spec.frames), spec.total_frames))
    from itertools import cycle
    (cycles, frames) = (cycle(range(1, spec.cycles + 1)), cycle(range(1, spec.total_frames + 1)))
    term = terminal.get_term()
    term.hide_cursor()
    try:
        while True:
            c = next(cycles)
            for (i, f) in enumerate(spec.runner(), 1):
                n = next(frames)
                print(f'\r{CYAN(c, cf)}:{CYAN(i, lf)} -->{join_cells(f)}<-- {CYAN(n, tf)} ')
                print(DIM('(press CTRL+C to stop)'), end='')
                term.clear_end_line()
                time.sleep(1 / 15)
                term.cursor_up_1()
    except KeyboardInterrupt:
        pass
    finally:
        term.show_cursor()