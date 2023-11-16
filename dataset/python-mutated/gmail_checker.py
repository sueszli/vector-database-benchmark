import imaplib
import re
from libqtile.log_utils import logger
from libqtile.widget import base

class GmailChecker(base.ThreadPoolText):
    """
    A simple gmail checker. If 'status_only_unseen' is True - set 'fmt' for one
    argument, ex. 'unseen: {0}'
    """
    defaults = [('update_interval', 30, 'Update time in seconds.'), ('username', None, 'username'), ('password', None, 'password'), ('email_path', 'INBOX', 'email_path'), ('display_fmt', 'inbox[{0}],unseen[{1}]', 'Display format'), ('status_only_unseen', False, 'Only show unseen messages')]

    def __init__(self, **config):
        if False:
            for i in range(10):
                print('nop')
        base.ThreadPoolText.__init__(self, '', **config)
        self.add_defaults(GmailChecker.defaults)

    def poll(self):
        if False:
            i = 10
            return i + 15
        self.gmail = imaplib.IMAP4_SSL('imap.gmail.com')
        self.gmail.login(self.username, self.password)
        (answer, raw_data) = self.gmail.status(self.email_path, '(MESSAGES UNSEEN)')
        if answer == 'OK':
            dec = raw_data[0].decode()
            messages = int(re.search('MESSAGES\\s+(\\d+)', dec).group(1))
            unseen = int(re.search('UNSEEN\\s+(\\d+)', dec).group(1))
            if self.status_only_unseen:
                return self.display_fmt.format(unseen)
            else:
                return self.display_fmt.format(messages, unseen)
        else:
            logger.exception('GmailChecker UNKNOWN error, answer: %s, raw_data: %s', answer, raw_data)
            return 'UNKNOWN ERROR'