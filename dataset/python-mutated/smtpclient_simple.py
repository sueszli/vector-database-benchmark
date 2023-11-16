"""
Demonstrate sending mail via SMTP.
"""
import sys
from email.mime.text import MIMEText
from twisted.internet import reactor
from twisted.mail.smtp import sendmail
from twisted.python import log

def send(message, subject, sender, recipients, host):
    if False:
        i = 10
        return i + 15
    '\n    Send email to one or more addresses.\n    '
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)
    dfr = sendmail(host, sender, recipients, msg.as_string())

    def success(r):
        if False:
            for i in range(10):
                print('nop')
        reactor.stop()

    def error(e):
        if False:
            print('Hello World!')
        print(e)
        reactor.stop()
    dfr.addCallback(success)
    dfr.addErrback(error)
    reactor.run()
if __name__ == '__main__':
    msg = 'This is the message body'
    subject = 'This is the message subject'
    host = 'smtp.example.com'
    sender = 'sender@example.com'
    recipients = ['recipient@example.com']
    log.startLogging(sys.stdout)
    send(msg, subject, sender, recipients, host)