import pytest
from thefuck.rules.aws_cli import match, get_new_command
from thefuck.types import Command
no_suggestions = 'usage: aws [options] <command> <subcommand> [<subcommand> ...] [parameters]\nTo see help text, you can run:\n\n  aws help\n  aws <command> help\n  aws <command> <subcommand> help\naws: error: argument command: Invalid choice, valid choices are:\n\ndynamodb                                 | dynamodbstreams\nec2                                      | ecr\n'
misspelled_command = "usage: aws [options] <command> <subcommand> [<subcommand> ...] [parameters]\nTo see help text, you can run:\n\n  aws help\n  aws <command> help\n  aws <command> <subcommand> help\naws: error: argument command: Invalid choice, valid choices are:\n\ndynamodb                                 | dynamodbstreams\nec2                                      | ecr\n\n\nInvalid choice: 'dynamdb', maybe you meant:\n\n  * dynamodb\n"
misspelled_subcommand = "usage: aws [options] <command> <subcommand> [<subcommand> ...] [parameters]\nTo see help text, you can run:\n\n  aws help\n  aws <command> help\n  aws <command> <subcommand> help\naws: error: argument operation: Invalid choice, valid choices are:\n\nquery                                    | scan\nupdate-item                              | update-table\n\n\nInvalid choice: 'scn', maybe you meant:\n\n  * scan\n"
misspelled_subcommand_with_multiple_options = "usage: aws [options] <command> <subcommand> [<subcommand> ...] [parameters]\nTo see help text, you can run:\n\n  aws help\n  aws <command> help\n  aws <command> <subcommand> help\naws: error: argument operation: Invalid choice, valid choices are:\n\ndescribe-table                           | get-item\nlist-tables                              | put-item\n\n\nInvalid choice: 't-item', maybe you meant:\n\n  * put-item\n  * get-item\n"

@pytest.mark.parametrize('command', [Command('aws dynamdb scan', misspelled_command), Command('aws dynamodb scn', misspelled_subcommand), Command('aws dynamodb t-item', misspelled_subcommand_with_multiple_options)])
def test_match(command):
    if False:
        print('Hello World!')
    assert match(command)

def test_not_match():
    if False:
        for i in range(10):
            print('nop')
    assert not match(Command('aws dynamodb invalid', no_suggestions))

@pytest.mark.parametrize('command, result', [(Command('aws dynamdb scan', misspelled_command), ['aws dynamodb scan']), (Command('aws dynamodb scn', misspelled_subcommand), ['aws dynamodb scan']), (Command('aws dynamodb t-item', misspelled_subcommand_with_multiple_options), ['aws dynamodb put-item', 'aws dynamodb get-item'])])
def test_get_new_command(command, result):
    if False:
        while True:
            i = 10
    assert get_new_command(command) == result