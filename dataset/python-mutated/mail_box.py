import imaplib
import email
from datetime import datetime, timedelta
import html
from bs4 import BeautifulSoup
import re
import openai
import smtplib
from email.mime.text import MIMEText

class Mailbox:
    gmail_address = ''
    gmail_password = ''
    imap_server = 'imap.gmail.com'
    imap_port = 993
    to_addresses = ['']
    max_emails = 3

    def get_all_work_summary(self):
        if False:
            for i in range(10):
                print('nop')
        print('Getting mail...')
        try:
            mailbox = imaplib.IMAP4_SSL(self.imap_server, self.imap_port)
            mailbox.login(self.gmail_address, self.gmail_password)
            mailbox.select('INBOX')
            today = datetime.now().strftime('%d-%b-%Y')
            search_criteria = f'(SINCE "{today}")'
            (status, email_ids) = mailbox.search(None, search_criteria)
            if status == 'OK':
                email_ids = email_ids[0].split()
                print(f'Number of emails received today: {len(email_ids)}')
                max_emails = min(len(email_ids), self.max_emails)
                all_email_content = ''
                for i in range(max_emails):
                    email_id = email_ids[i]
                    email_content = self.get_email_content(mailbox, email_id)
                    if email_content:
                        all_email_content += f'{i + 1}、{email_content}\n'
            mailbox.logout()
            return all_email_content
        except Exception as e:
            print('Failed to get email:', str(e))

    def get_email_content(self, mailbox, email_id):
        if False:
            print('Hello World!')
        (status, email_data) = mailbox.fetch(email_id, '(RFC822)')
        if status == 'OK':
            raw_email = email_data[0][1]
            msg = email.message_from_bytes(raw_email)
            sender = msg['From']
            sender = re.findall('<(.*?)>', sender)
            sender = sender[0] if sender else ''
            if sender.lower().endswith('.com') and (not msg['In-Reply-To']):
                email_content = ''
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        if content_type == 'text/plain':
                            email_content = part.get_payload(decode=True).decode('utf-8')
                            break
                        elif content_type == 'text/html':
                            email_content = part.get_payload(decode=True).decode('utf-8')
                            email_content = html.unescape(email_content)
                            break
                else:
                    email_content = msg.get_payload(decode=True).decode('utf-8')
                if 'html' in email_content.lower():
                    soup = BeautifulSoup(email_content, 'html.parser')
                    email_content = soup.get_text()
                email_content = re.sub('\\s+', '', email_content)
                email_content = re.sub('=\\?.*?\\?=', '', email_content)
                email_content = re.sub('---.*', '', email_content)
                return f'{sender}Send an email with the content{email_content}'
        return ''

    def get_summary_by_ai(self, email_content: str, prompt: str) -> str:
        if False:
            while True:
                i = 10
        print('Asking AI to summarize email content...')
        response = openai.chat.completions.create(model='gpt-3.5-turbo-0613', messages=[{'role': 'system', 'content': prompt}, {'role': 'user', 'content': email_content}])
        summary = response.choices[0].message.content.strip()
        return summary

    def send_mail(self, summary, theme='Email summary summary'):
        if False:
            print('Hello World!')
        from_address = self.gmail_address
        to_addresses = self.to_addresses
        yesterday = (datetime.now() - timedelta(days=0)).strftime('%Y-%m-%d')
        subject = yesterday + theme
        body = summary
        try:
            smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
            smtp_server.ehlo()
            smtp_server.starttls()
            smtp_server.login(self.gmail_address, self.gmail_password)
            for to_address in to_addresses:
                message = MIMEText(body, 'plain', 'utf-8')
                message['Subject'] = subject
                message['From'] = from_address
                message['To'] = to_address
                smtp_server.sendmail(from_address, to_address, message.as_string())
                print('Email sent successfully to:', to_address)
            smtp_server.quit()
            print('All emails have been sent successfully!')
            return True
        except Exception as e:
            print('Email sending failed:', str(e))