"""
Purpose

Shows how to send email by using an Amazon Pinpoint SMTP server.
"""
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from os import environ
import logging
import smtplib
logger = logging.getLogger(__name__)

def send_smtp_message(smtp_server, smtp_username, smtp_password, sender, to_address, cc_address, subject, html_message, text_message):
    if False:
        return 10
    '\n    Sends an email by using an Amazon Pinpoint SMTP server.\n\n    :param smtp_server: An smtplib SMTP session.\n    :param smtp_username: The username to use to connect to the SMTP server.\n    :param smtp_password: The password to use to connect to the SMTP server.\n    :param sender: The "From" address. This address must be verified.\n    :param to_address: The "To" address. If your account is still in the sandbox,\n                       this address must be verified.\n    :param cc_address: The "CC" address. If your account is still in the sandbox,\n                       this address must be verified.\n    :param subject: The subject line of the email.\n    :param html_message: The HTML body of the email.\n    :param text_message: The email body for recipients with non-HTML email clients.\n    '
    msg = MIMEMultipart('alternative')
    msg['From'] = sender
    msg['To'] = to_address
    msg['Cc'] = cc_address
    msg['Subject'] = subject
    msg.attach(MIMEText(html_message, 'html'))
    msg.attach(MIMEText(text_message, 'plain'))
    smtp_server.ehlo()
    smtp_server.starttls()
    smtp_server.ehlo()
    smtp_server.login(smtp_username, smtp_password)
    smtp_server.sendmail(sender, to_address, msg.as_string())

def main():
    if False:
        for i in range(10):
            print('nop')
    host = 'email-smtp.us-west-2.amazonaws.com'
    port = 587
    sender = 'sender@example.com'
    to_address = 'recipient@example.com'
    cc_address = 'cc_recipient@example.com'
    subject = 'Amazon Pinpoint Test (Python smtplib)'
    text_message = 'Amazon Pinpoint Test\r\nThis email was sent through the Amazon Pinpoint SMTP interface using the Python smtplib package.'
    html_message = "<html>\n    <head></head>\n    <body>\n      <h1>Amazon Pinpoint SMTP Email Test</h1>\n      <p>This email was sent with Amazon Pinpoint using the\n        <a href='https://www.python.org/'>Python</a>\n        <a href='https://docs.python.org/3/library/smtplib.html'>\n        smtplib</a> library.</p>\n    </body>\n    </html>"
    smtp_username = 'AKIAIOSFODNN7EXAMPLE'
    smtp_password = environ['SMTP_PASSWORD']
    print('Sending email through SMTP server.')
    try:
        with smtplib.SMTP(host, port) as smtp_server:
            send_smtp_message(smtp_server, smtp_username, smtp_password, sender, to_address, cc_address, subject, html_message, text_message)
    except Exception:
        logger.exception("Couldn't send message.")
        raise
    else:
        print('Email sent!')
if __name__ == '__main__':
    main()