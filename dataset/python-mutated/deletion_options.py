import os
from hscommon.gui.base import GUIObject
from hscommon.trans import tr

class DeletionOptionsView:
    """Expected interface for :class:`DeletionOptions`'s view.

    *Not actually used in the code. For documentation purposes only.*

    Our view presents the user with an appropriate way (probably a mix of checkboxes and radio
    buttons) to set the different flags in :class:`DeletionOptions`. Note that
    :attr:`DeletionOptions.use_hardlinks` is only relevant if :attr:`DeletionOptions.link_deleted`
    is true. This is why we toggle the "enabled" state of that flag.

    We expect the view to set :attr:`DeletionOptions.link_deleted` immediately as the user changes
    its value because it will toggle :meth:`set_hardlink_option_enabled`

    Other than the flags, there's also a prompt message which has a dynamic content, defined by
    :meth:`update_msg`.
    """

    def update_msg(self, msg: str):
        if False:
            return 10
        "Update the dialog's prompt with ``str``."

    def show(self):
        if False:
            while True:
                i = 10
        'Show the dialog in a modal fashion.\n\n        Returns whether the dialog was "accepted" (the user pressed OK).\n        '

    def set_hardlink_option_enabled(self, is_enabled: bool):
        if False:
            for i in range(10):
                print('nop')
        'Enable or disable the widget controlling :attr:`DeletionOptions.use_hardlinks`.'

class DeletionOptions(GUIObject):
    """Present the user with deletion options before proceeding.

    When the user activates "Send to trash", we present him with a couple of options that changes
    the behavior of that deletion operation.
    """

    def __init__(self):
        if False:
            return 10
        GUIObject.__init__(self)
        self.use_hardlinks = False
        self.direct = False

    def show(self, mark_count):
        if False:
            return 10
        'Prompt the user with a modal dialog offering our deletion options.\n\n        :param int mark_count: Number of dupes marked for deletion.\n        :rtype: bool\n        :returns: Whether the user accepted the dialog (we cancel deletion if false).\n        '
        self._link_deleted = False
        self.view.set_hardlink_option_enabled(False)
        self.use_hardlinks = False
        self.direct = False
        msg = tr('You are sending {} file(s) to the Trash.').format(mark_count)
        self.view.update_msg(msg)
        return self.view.show()

    def supports_links(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns whether our platform supports symlinks.'
        try:
            os.symlink()
        except NotImplementedError:
            return False
        except OSError:
            return False
        except TypeError:
            return True

    @property
    def link_deleted(self):
        if False:
            while True:
                i = 10
        'Replace deleted dupes with symlinks (or hardlinks) to the dupe group reference.\n\n        *bool*. *get/set*\n\n        Whether the link is a symlink or hardlink is decided by :attr:`use_hardlinks`.\n        '
        return self._link_deleted

    @link_deleted.setter
    def link_deleted(self, value):
        if False:
            while True:
                i = 10
        self._link_deleted = value
        hardlinks_enabled = value and self.supports_links()
        self.view.set_hardlink_option_enabled(hardlinks_enabled)