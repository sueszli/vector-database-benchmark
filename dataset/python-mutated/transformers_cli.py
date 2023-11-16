from argparse import ArgumentParser
from .add_new_model import AddNewModelCommand
from .add_new_model_like import AddNewModelLikeCommand
from .convert import ConvertCommand
from .download import DownloadCommand
from .env import EnvironmentCommand
from .lfs import LfsCommands
from .pt_to_tf import PTtoTFCommand
from .run import RunCommand
from .serving import ServeCommand
from .user import UserCommands

def main():
    if False:
        print('Hello World!')
    parser = ArgumentParser('Transformers CLI tool', usage='transformers-cli <command> [<args>]')
    commands_parser = parser.add_subparsers(help='transformers-cli command helpers')
    ConvertCommand.register_subcommand(commands_parser)
    DownloadCommand.register_subcommand(commands_parser)
    EnvironmentCommand.register_subcommand(commands_parser)
    RunCommand.register_subcommand(commands_parser)
    ServeCommand.register_subcommand(commands_parser)
    UserCommands.register_subcommand(commands_parser)
    AddNewModelCommand.register_subcommand(commands_parser)
    AddNewModelLikeCommand.register_subcommand(commands_parser)
    LfsCommands.register_subcommand(commands_parser)
    PTtoTFCommand.register_subcommand(commands_parser)
    args = parser.parse_args()
    if not hasattr(args, 'func'):
        parser.print_help()
        exit(1)
    service = args.func(args)
    service.run()
if __name__ == '__main__':
    main()