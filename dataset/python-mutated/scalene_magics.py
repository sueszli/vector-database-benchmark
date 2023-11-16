import contextlib
import sys
import textwrap
from typing import Any
with contextlib.suppress(Exception):
    from IPython.core.magic import Magics, line_cell_magic, line_magic, magics_class
    from scalene import scalene_profiler
    from scalene.scalene_arguments import ScaleneArguments
    from scalene.scalene_parseargs import ScaleneParseArgs

    @magics_class
    class ScaleneMagics(Magics):
        """IPython (Jupyter) support for magics for Scalene (%scrun and %%scalene)."""

        def run_code(self, args: ScaleneArguments, code: str) -> None:
            if False:
                for i in range(10):
                    print('nop')
            import IPython
            filename = f'_ipython-input-{len(IPython.get_ipython().history_manager.input_hist_raw) - 1}-profile'
            with open(filename, 'w+') as tmpfile:
                tmpfile.write(code)
            args.memory = False
            scalene_profiler.Scalene.set_initialized()
            scalene_profiler.Scalene.run_profiler(args, [filename], is_jupyter=True)

        @line_cell_magic
        def scalene(self, line: str, cell: str='') -> None:
            if False:
                for i in range(10):
                    print('nop')
            '%%scalene magic: see https://github.com/plasma-umass/scalene for usage info.'
            print('SCALENE MAGIC')
            if line:
                sys.argv = ['scalene', '--ipython', *line.split()]
                (args, _left) = ScaleneParseArgs.parse_args()
                print(f'args={args!r}, _left={_left!r}')
            else:
                args = ScaleneArguments()
                print(f'args={args!r}')
            if args and cell:
                self.run_code(args, '\n' + cell)
                print(f'cell={cell!r}')

        @line_magic
        def scrun(self, line: str='') -> None:
            if False:
                return 10
            '%scrun magic: see https://github.com/plasma-umass/scalene for usage info.'
            print('SCRUN MAGIC')
            if line:
                sys.argv = ['scalene', '--ipython', *line.split()]
                (args, left) = ScaleneParseArgs.parse_args()
                if args:
                    self.run_code(args, ' '.join(left))

    def load_ipython_extension(ip: Any) -> None:
        if False:
            while True:
                i = 10
        print('LOADING')
        ip.register_magics(ScaleneMagics)
        with contextlib.suppress(Exception):
            with open('scalene-usage.txt', 'r') as usage:
                usage_str = usage.read()
            ScaleneMagics.scrun.__doc__ = usage_str
            ScaleneMagics.scalene.__doc__ = usage_str
        print('\n'.join(textwrap.wrap('Scalene extension successfully loaded. Note: Scalene currently only supports CPU+GPU profiling inside Jupyter notebooks. For full Scalene profiling, use the command line version.')))
        if sys.platform == 'darwin':
            print()
            print('\n'.join(textwrap.wrap('NOTE: in Jupyter notebook on MacOS, Scalene cannot profile child processes. Do not run to try Scalene with multiprocessing in Jupyter Notebook.')))