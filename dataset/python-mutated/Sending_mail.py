import csv
from email.message import EmailMessage
import smtplib

def get_credentials(filepath):
    if False:
        return 10
    with open('credentials.txt', 'r') as f:
        email_address = f.readline()
        email_pass = f.readline()
    return (email_address, email_pass)

def login(email_address, email_pass, s):
    if False:
        while True:
            i = 10
    s.ehlo()
    s.starttls()
    s.ehlo()
    s.login(email_address, email_pass)
    print('login')

def send_mail():
    if False:
        print('Hello World!')
    s = smtplib.SMTP('smtp.gmail.com', 587)
    (email_address, email_pass) = get_credentials('./credentials.txt')
    login(email_address, email_pass, s)
    subject = 'Welcome to Python'
    body = "Python is an interpreted, high-level,\n    general-purpose programming language.\n\n    Created by Guido van Rossum and first released in 1991,\n    Python's design philosophy emphasizes code readability\n\n    with its notable use of significant whitespace"
    message = EmailMessage()
    message.set_content(body)
    message['Subject'] = subject
    with open('emails.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for email in spamreader:
            s.send_message(email_address, email[0], message)
            print('Send To ' + email[0])
    s.quit()
    print('sent')
if __name__ == '__main__':
    send_mail()