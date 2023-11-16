from typing import Iterable, NamedTuple
from trashcli.empty.user import User
from trashcli.trash_dirs_scanner import TrashDir
UserIntention = NamedTuple('UserIntention', [('ok_to_empty', bool), ('trash_dirs', Iterable[TrashDir])])

class Guard:

    def __init__(self, user):
        if False:
            while True:
                i = 10
        self.user = user

    def ask_the_user(self, parsed_interactive, trash_dirs):
        if False:
            return 10
        if parsed_interactive:
            return self._interactive(trash_dirs)
        else:
            return self.non_interactive(trash_dirs)

    def _interactive(self, trash_dirs):
        if False:
            for i in range(10):
                print('nop')
        trash_dirs_list = list(trash_dirs)
        ok_to_empty = self.user.do_you_wanna_empty_trash_dirs(trash_dirs_list)
        list_result = trash_dirs_list if ok_to_empty else []
        return UserIntention(ok_to_empty=ok_to_empty, trash_dirs=list_result)

    def non_interactive(self, trash_dirs):
        if False:
            return 10
        trash_dirs_list = trash_dirs
        return UserIntention(ok_to_empty=True, trash_dirs=trash_dirs_list)