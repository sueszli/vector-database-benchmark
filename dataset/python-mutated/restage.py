import llnl.util.tty as tty
import spack.cmd
import spack.cmd.common.arguments as arguments
import spack.repo
description = 'revert checked out package source code'
section = 'build'
level = 'long'

def setup_parser(subparser):
    if False:
        return 10
    arguments.add_common_arguments(subparser, ['specs'])

def restage(parser, args):
    if False:
        for i in range(10):
            print('nop')
    if not args.specs:
        tty.die('spack restage requires at least one package spec.')
    specs = spack.cmd.parse_specs(args.specs, concretize=True)
    for spec in specs:
        spec.package.do_restage()