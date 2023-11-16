from tests.support.help_reformatting import normalize_spaces

class TestNormalizeSpaces:

    def test(self):
        if False:
            while True:
                i = 10
        text = 'usage: trash-list [-h] [--print-completion {bash,zsh,tcsh}] [--version]\n                  [--volumes] [--trash-dirs] [--trash-dir TRASH_DIRS]\n                  [--all-users]'
        assert normalize_spaces(text) == 'usage: trash-list [-h] [--print-completion {bash,zsh,tcsh}] [--version] [--volumes] [--trash-dirs] [--trash-dir TRASH_DIRS] [--all-users]'