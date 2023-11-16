import imaplib

class ImapEmail:

    def imap_open(self, imap_folder, email_sender, email_password, imap_server) -> imaplib.IMAP4_SSL:
        if False:
            while True:
                i = 10
        '\n        Function to open an IMAP connection to the email server.\n\n        Args:\n            imap_folder (str): The folder to open.\n            email_sender (str): The email address of the sender.\n            email_password (str): The password of the sender.\n\n        Returns:\n            imaplib.IMAP4_SSL: The IMAP connection.\n        '
        conn = imaplib.IMAP4_SSL(imap_server)
        conn.login(email_sender, email_password)
        conn.select(imap_folder)
        return conn

    def adjust_imap_folder(self, imap_folder, email_sender) -> str:
        if False:
            print('Hello World!')
        '\n        Function to adjust the IMAP folder based on the email address of the sender.\n\n        Args:\n            imap_folder (str): The folder to open.\n            email_sender (str): The email address of the sender.\n\n        Returns:\n            str: The adjusted IMAP folder.\n        '
        if '@gmail' in email_sender.lower():
            if 'sent' in imap_folder.lower():
                return '"[Gmail]/Sent Mail"'
            if 'draft' in imap_folder.lower():
                return '"[Gmail]/Drafts"'
        return imap_folder