from tests.support.help_reformatting import reformat_help_message, split_paragraphs

class TestParseHelp:

    def test_format_help_message(self):
        if False:
            return 10
        assert reformat_help_message(self.help_message) == "usage: trash-list [-h] [--print-completion {bash,zsh,tcsh}] [--version] [--volumes] [--trash-dirs] [--trash-dir TRASH_DIRS] [--all-users]\n\nList trashed files\n\noptions:\n  -h, --help            show this help message and exit\n  --print-completion {bash,zsh,tcsh}\n                        print shell completion script\n  --version             show program's version number and exit\n  --volumes             list volumes\n  --trash-dirs          list trash dirs\n  --trash-dir TRASH_DIRS\n                        specify the trash directory to use\n  --all-users           list trashcans of all the users\n\nReport bugs to https://github.com/andreafrancia/trash-cli/issues\n"

    def test_first(self):
        if False:
            print('Hello World!')
        assert split_paragraphs(self.help_message)[0] == 'usage: trash-list [-h] [--print-completion {bash,zsh,tcsh}] [--version]\n                  [--volumes] [--trash-dirs] [--trash-dir TRASH_DIRS]\n                  [--all-users]\n'

    def test_second(self):
        if False:
            print('Hello World!')
        assert split_paragraphs(self.help_message)[1] == 'List trashed files\n'

    def test_third(self):
        if False:
            for i in range(10):
                print('nop')
        assert split_paragraphs(self.help_message)[2] == "options:\n  -h, --help            show this help message and exit\n  --print-completion {bash,zsh,tcsh}\n                        print shell completion script\n  --version             show program's version number and exit\n  --volumes             list volumes\n  --trash-dirs          list trash dirs\n  --trash-dir TRASH_DIRS\n                        specify the trash directory to use\n  --all-users           list trashcans of all the users\n"

    def test_fourth(self):
        if False:
            return 10
        assert split_paragraphs(self.help_message)[3] == 'Report bugs to https://github.com/andreafrancia/trash-cli/issues\n'

    def test_only_four(self):
        if False:
            return 10
        assert len(split_paragraphs(self.help_message)) == 4
    help_message = "usage: trash-list [-h] [--print-completion {bash,zsh,tcsh}] [--version]\n                  [--volumes] [--trash-dirs] [--trash-dir TRASH_DIRS]\n                  [--all-users]\n\nList trashed files\n\noptions:\n  -h, --help            show this help message and exit\n  --print-completion {bash,zsh,tcsh}\n                        print shell completion script\n  --version             show program's version number and exit\n  --volumes             list volumes\n  --trash-dirs          list trash dirs\n  --trash-dir TRASH_DIRS\n                        specify the trash directory to use\n  --all-users           list trashcans of all the users\n\nReport bugs to https://github.com/andreafrancia/trash-cli/issues\n"