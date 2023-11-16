import pytest
from thefuck.rules.hostscli import no_website, get_new_command, match
from thefuck.types import Command
no_website_long = '\n{}:\n\nNo Domain list found for website: a_website_that_does_not_exist\n\nPlease raise a Issue here: https://github.com/dhilipsiva/hostscli/issues/new\nif you think we should add domains for this website.\n\ntype `hostscli websites` to see a list of websites that you can block/unblock\n'.format(no_website)

@pytest.mark.parametrize('command', [Command('hostscli block a_website_that_does_not_exist', no_website_long)])
def test_match(command):
    if False:
        print('Hello World!')
    assert match(command)

@pytest.mark.parametrize('command, result', [(Command('hostscli block a_website_that_does_not_exist', no_website_long), ['hostscli websites'])])
def test_get_new_command(command, result):
    if False:
        print('Hello World!')
    assert get_new_command(command) == result