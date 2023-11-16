import sys
from setuptools.command.egg_info import egg_info as _egg_info

class egg_info(_egg_info):

    def run(self):
        if False:
            i = 10
            return i + 15
        if 'sdist' in sys.argv:
            import warnings
            import textwrap
            msg = textwrap.dedent('\n                `build_src` is being run, this may lead to missing\n                files in your sdist!  You want to use distutils.sdist\n                instead of the setuptools version:\n\n                    from distutils.command.sdist import sdist\n                    cmdclass={\'sdist\': sdist}"\n\n                See numpy\'s setup.py or gh-7131 for details.')
            warnings.warn(msg, UserWarning, stacklevel=2)
        self.run_command('build_src')
        _egg_info.run(self)