import os
import sys
import click
from spin.cmds import meson
from spin import util

@click.command()
@click.argument('sphinx_target', default='html')
@click.option('--clean', is_flag=True, default=False, help='Clean previously built docs before building')
@click.option('--build/--no-build', 'first_build', default=True, help='Build project before generating docs')
@click.option('--plot/--no-plot', 'sphinx_gallery_plot', default=True, help='Sphinx gallery: enable/disable plots')
@click.option('--jobs', '-j', default='auto', help='Number of parallel build jobs')
@click.option('--install-deps/--no-install-deps', default=False, help='Install dependencies before building')
@click.pass_context
def docs(ctx, sphinx_target, clean, first_build, jobs, sphinx_gallery_plot, install_deps):
    if False:
        for i in range(10):
            print('nop')
    '📖 Build documentation\n\n    By default, SPHINXOPTS="-W", raising errors on warnings.\n    To build without raising on warnings:\n\n      SPHINXOPTS="" spin docs\n\n    The command is roughly equivalent to `cd doc && make SPHINX_TARGET`.\n    To get a list of viable `SPHINX_TARGET`:\n\n      spin docs help\n\n    '
    if install_deps:
        util.run(['pip', 'install', '-q', '-r', 'requirements/docs.txt'])
    for extra_param in ('install_deps',):
        del ctx.params[extra_param]
    ctx.forward(meson.docs)

@click.command()
@click.argument('asv_args', nargs=-1)
def asv(asv_args):
    if False:
        print('Hello World!')
    '🏃 Run `asv` to collect benchmarks\n\n    ASV_ARGS are passed through directly to asv, e.g.:\n\n    spin asv -- dev -b TransformSuite\n\n    Please see CONTRIBUTING.txt\n    '
    site_path = meson._get_site_packages()
    if site_path is None:
        print('No built scikit-image found; run `spin build` first.')
        sys.exit(1)
    os.environ['PYTHONPATH'] = f"{site_path}{os.sep}:{os.environ.get('PYTHONPATH', '')}"
    util.run(['asv'] + list(asv_args))

@click.command()
def sdist():
    if False:
        for i in range(10):
            print('nop')
    '📦 Build a source distribution in `dist/`.'
    util.run(['python', '-m', 'build', '.', '--sdist'])

@click.command(context_settings={'ignore_unknown_options': True})
@click.argument('ipython_args', metavar='', nargs=-1)
@click.pass_context
def ipython(ctx, ipython_args):
    if False:
        for i in range(10):
            print('nop')
    '💻 Launch IPython shell with PYTHONPATH set\n\n    OPTIONS are passed through directly to IPython, e.g.:\n\n    spin ipython -i myscript.py\n    '
    env = os.environ
    env['PYTHONWARNINGS'] = env.get('PYTHONWARNINGS', 'all')
    preimport = "import skimage as ski; print(f'\\nPreimported scikit-image {ski.__version__} as ski')"
    ctx.params['ipython_args'] = (f'--TerminalIPythonApp.exec_lines={preimport}',) + ipython_args
    ctx.forward(meson.ipython)