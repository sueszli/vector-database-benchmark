import os
import re
from email.header import decode_header
from bs4 import BeautifulSoup

class ReadEmail:

    def clean_email_body(self, email_body):
        if False:
            i = 10
            return i + 15
        '\n        Function to clean the email body.\n\n        Args:\n            email_body (str): The email body to be cleaned.\n\n        Returns:\n            str: The cleaned email body.\n        '
        if email_body is None:
            email_body = ''
        email_body = BeautifulSoup(email_body, 'html.parser')
        email_body = email_body.get_text()
        email_body = ''.join(email_body.splitlines())
        email_body = ' '.join(email_body.split())
        email_body = email_body.encode('ascii', 'ignore')
        email_body = email_body.decode('utf-8', 'ignore')
        email_body = re.sub('http\\S+', '', email_body)
        return email_body

    def clean(self, text):
        if False:
            return 10
        '\n        Function to clean the text.\n\n        Args:\n            text (str): The text to be cleaned.\n\n        Returns:\n            str: The cleaned text.\n        '
        return ''.join((c if c.isalnum() else '_' for c in text))

    def obtain_header(self, msg):
        if False:
            i = 10
            return i + 15
        '\n        Function to obtain the header of the email.\n\n        Args:\n            msg (email.message.Message): The email message.\n\n        Returns:\n            str: The From field of the email.\n        '
        if msg['Subject'] is not None:
            (Subject, encoding) = decode_header(msg['Subject'])[0]
        else:
            Subject = ''
            encoding = ''
        if isinstance(Subject, bytes):
            try:
                if encoding is not None:
                    Subject = Subject.decode(encoding)
                else:
                    Subject = ''
            except [LookupError] as err:
                pass
        From = msg['From']
        To = msg['To']
        Date = msg['Date']
        return (From, To, Date, Subject)

    def download_attachment(self, part, subject):
        if False:
            for i in range(10):
                print('nop')
        '\n        Function to download the attachment from the email.\n\n        Args:\n            part (email.message.Message): The email message.\n            subject (str): The subject of the email.\n\n        Returns:\n            None\n        '
        filename = part.get_filename()
        if filename:
            folder_name = self.clean(subject)
            if not os.path.isdir(folder_name):
                os.mkdir(folder_name)
                filepath = os.path.join(folder_name, filename)
                open(filepath, 'wb').write(part.get_payload(decode=True))