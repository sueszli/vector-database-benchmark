import PySimpleGUI as sg
'\n    Learn how to send emails from PySimpleGUI using the smtplib and email modules\n    \n    The GUI portion is simple\n\n    Based on a send-email script originally written by by Israel Dryer\n    (Thank you Israel for figuring out the hard part of the stmp and email module calls!)\n    \n    Copyright 2019, 2022 PySimpleGUI\n\n'
smtp_host_default = ''
import smtplib as smtp
from email.message import EmailMessage

def send_an_email(from_address, to_address, subject, message_text, user, password, smtp_host, smtp_port):
    if False:
        i = 10
        return i + 15
    server = smtp.SMTP(host=smtp_host, port=smtp_port)
    server.starttls()
    try:
        server.login(user=user, password=password)
    except Exception as e:
        sg.popup_error('Error authenticaing your email credentials', e, image=sg.EMOJI_BASE64_WEARY)
        server.close()
        return
    msg = EmailMessage()
    msg['From'] = from_address
    msg['To'] = to_address
    msg['Subject'] = subject
    msg.set_payload(message_text)
    try:
        server.send_message(msg)
    except Exception as e:
        sg.popup_error('Error sending your email', e, image=sg.EMOJI_BASE64_WEARY)
        server.close()
        return
    server.close()
    sg.popup('Email sent successfully!', image=sg.EMOJI_BASE64_HAPPY_JOY)
'\n    important notes about using gmail\n\n    - Gmail has locked things down pretty good with what it considers less secure apps. That \n        would include access your Gmail account from the smtplib library in Python. However, there \n        is a work around. You can enable access from "Less Secure Apps" by going to your Gmail \n        account and enabling that feature. However, you should do this at your own peril, and after \n        carefully reading the warnings: https://support.google.com/accounts/answer/6010255.\n\n    smtplib | https://docs.python.org/3/library/smtplib.html?#module-smtplib   \n    email.message | https://docs.python.org/3/library/email.message.html?#module-email.message   \n    email examples in Python | https://docs.python.org/3.7/library/email.examples.html  \n'

def main():
    if False:
        return 10
    smtp_server_dict = {'gmail.com': 'smtp.gmail.com', 'hotmail.com': 'smtp.office365.com', 'live.com': 'smtp.office365.com', 'yahoo.com': 'smtp.mail.yahoo.com'}
    sg.theme('Dark Blue 3')
    layout = [[sg.Text('Send an Email', font='Default 18')], [sg.T('From:', size=(8, 1)), sg.Input(key='-EMAIL FROM-', size=(35, 1))], [sg.T('To:', size=(8, 1)), sg.Input(key='-EMAIL TO-', size=(35, 1))], [sg.T('Subject:', size=(8, 1)), sg.Input(key='-EMAIL SUBJECT-', size=(35, 1))], [sg.T('Mail login information', font='Default 18')], [sg.T('User:', size=(8, 1)), sg.Input(key='-USER-', size=(35, 1), enable_events=True)], [sg.T('Password:', size=(8, 1)), sg.Input(password_char='*', key='-PASSWORD-', size=(35, 1))], [sg.T('SMTP Server Info', font='_ 14')], [sg.T('SMTP Hostname'), sg.Input(smtp_host_default, s=20, key='-SMTP HOST-'), sg.T('SMTP Port'), sg.In(587, s=4, key='-SMTP PORT-')], [sg.Multiline('Type your message here', size=(60, 10), key='-EMAIL TEXT-')], [sg.Button('Send'), sg.Button('Exit')]]
    window = sg.Window('Send An Email', layout)
    while True:
        (event, values) = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        if event == 'Send':
            if values['-SMTP HOST-']:
                sg.popup_quick_message('Sending your message... this will take a moment...', background_color='red')
                send_an_email(from_address=values['-EMAIL FROM-'], to_address=values['-EMAIL TO-'], subject=values['-EMAIL SUBJECT-'], message_text=values['-EMAIL TEXT-'], user=values['-USER-'], password=values['-PASSWORD-'], smtp_host=values['-SMTP HOST-'], smtp_port=values['-SMTP PORT-'])
            else:
                sg.popup_error('Missing SMTP Hostname... you have to supply a hostname (gmail, hotmail, live, yahoo are autofilled)')
        elif event == '-USER-':
            for service in smtp_server_dict.keys():
                if service in values[event].lower():
                    window['-SMTP HOST-'].update(smtp_server_dict[service])
                    break
    window.close()
if __name__ == '__main__':
    main()