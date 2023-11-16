from django.core.mail import EmailMessage

def send_graph_email(subject, sender, recipients, attachments=None, body=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    :param str sender: sender's email address\n    :param list recipients: list of recipient emails\n    :param list attachments: list of triples of the form:\n        (filename, content, mimetype). See the django docs\n        https://docs.djangoproject.com/en/1.3/topics/email/#django.core.mail.EmailMessage\n    "
    attachments = attachments or []
    msg = EmailMessage(subject=subject, from_email=sender, to=recipients, body=body, attachments=attachments)
    msg.send()