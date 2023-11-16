from __future__ import annotations
import codecs
import os
import smtplib
import socket
import logging
import mimetypes
from time import time
from typing import Any, Iterable, Optional, Tuple, Union, IO
from email.message import EmailMessage
from email import utils
from ckan.common import _, config
import ckan
import ckan.model as model
import ckan.lib.helpers as h
from ckan.lib.base import render
log = logging.getLogger(__name__)
AttachmentWithType = Union[Tuple[str, IO[str], str], Tuple[str, IO[bytes], str]]
AttachmentWithoutType = Union[Tuple[str, IO[str]], Tuple[str, IO[bytes]]]
Attachment = Union[AttachmentWithType, AttachmentWithoutType]

class MailerException(Exception):
    pass

def _mail_recipient(recipient_name: str, recipient_email: str, sender_name: str, sender_url: str, subject: Any, body: Any, body_html: Optional[Any]=None, headers: Optional[dict[str, Any]]=None, attachments: Optional[Iterable[Attachment]]=None) -> None:
    if False:
        i = 10
        return i + 15
    if not headers:
        headers = {}
    if not attachments:
        attachments = []
    mail_from = config.get('smtp.mail_from')
    reply_to = config.get('smtp.reply_to')
    msg = EmailMessage()
    msg.set_content(body, cte='base64')
    if body_html:
        msg.add_alternative(body_html, subtype='html', cte='base64')
    for (k, v) in headers.items():
        if k in msg.keys():
            msg.replace_header(k, v)
        else:
            msg.add_header(k, v)
    msg['Subject'] = subject
    msg['From'] = utils.formataddr((sender_name, mail_from))
    msg['To'] = utils.formataddr((recipient_name, recipient_email))
    msg['Date'] = utils.formatdate(time())
    if not config.get('ckan.hide_version'):
        msg['X-Mailer'] = 'CKAN %s' % ckan.__version__
    if reply_to and reply_to != '' and (not msg['Reply-to']):
        msg['Reply-to'] = reply_to
    for attachment in attachments:
        if len(attachment) == 3:
            (name, _file, media_type) = attachment
        else:
            (name, _file) = attachment
            media_type = None
        if not media_type:
            (media_type, _encoding) = mimetypes.guess_type(name)
        if media_type:
            (main_type, sub_type) = media_type.split('/')
        else:
            main_type = sub_type = None
        msg.add_attachment(_file.read(), filename=name, maintype=main_type, subtype=sub_type)
    smtp_server = config.get('smtp.server')
    smtp_starttls = config.get('smtp.starttls')
    smtp_user = config.get('smtp.user')
    smtp_password = config.get('smtp.password')
    try:
        smtp_connection = smtplib.SMTP(smtp_server)
    except (socket.error, smtplib.SMTPConnectError) as e:
        log.exception(e)
        raise MailerException('SMTP server could not be connected to: "%s" %s' % (smtp_server, e))
    try:
        smtp_connection.ehlo()
        if smtp_starttls:
            if smtp_connection.has_extn('STARTTLS'):
                smtp_connection.starttls()
                smtp_connection.ehlo()
            else:
                raise MailerException('SMTP server does not support STARTTLS')
        if smtp_user:
            assert smtp_password, 'If smtp.user is configured then smtp.password must be configured as well.'
            smtp_connection.login(smtp_user, smtp_password)
        smtp_connection.sendmail(mail_from, [recipient_email], msg.as_string())
        log.info('Sent email to {0}'.format(recipient_email))
    except smtplib.SMTPException as e:
        msg = '%r' % e
        log.exception(msg)
        raise MailerException(msg)
    finally:
        smtp_connection.quit()

def mail_recipient(recipient_name: str, recipient_email: str, subject: str, body: str, body_html: Optional[str]=None, headers: Optional[dict[str, Any]]=None, attachments: Optional[Iterable[Attachment]]=None) -> None:
    if False:
        i = 10
        return i + 15
    "Sends an email to a an email address.\n\n    .. note:: You need to set up the :ref:`email-settings` to able to send\n        emails.\n\n    :param recipient_name: the name of the recipient\n    :type recipient: string\n    :param recipient_email: the email address of the recipient\n    :type recipient: string\n\n    :param subject: the email subject\n    :type subject: string\n    :param body: the email body, in plain text\n    :type body: string\n    :param body_html: the email body, in html format (optional)\n    :type body_html: string\n    :headers: extra headers to add to email, in the form\n        {'Header name': 'Header value'}\n    :type: dict\n    :attachments: a list of tuples containing file attachments to add to the\n        email. Tuples should contain the file name and a file-like object\n        pointing to the file contents::\n\n            [\n                ('some_report.csv', file_object),\n            ]\n\n        Optionally, you can add a third element to the tuple containing the\n        media type. If not provided, it will be guessed using\n        the ``mimetypes`` module::\n\n            [\n                ('some_report.csv', file_object, 'text/csv'),\n            ]\n    :type: list\n    "
    site_title = config.get('ckan.site_title')
    site_url = config.get('ckan.site_url')
    return _mail_recipient(recipient_name, recipient_email, site_title, site_url, subject, body, body_html=body_html, headers=headers, attachments=attachments)

def mail_user(recipient: model.User, subject: str, body: str, body_html: Optional[str]=None, headers: Optional[dict[str, Any]]=None, attachments: Optional[Iterable[Attachment]]=None) -> None:
    if False:
        while True:
            i = 10
    'Sends an email to a CKAN user.\n\n    You need to set up the :ref:`email-settings` to able to send emails.\n\n    :param recipient: a CKAN user object\n    :type recipient: a model.User object\n\n    For further parameters see\n    :py:func:`~ckan.lib.mailer.mail_recipient`.\n    '
    if recipient.email is None or not len(recipient.email):
        raise MailerException(_('No recipient email address available!'))
    mail_recipient(recipient.display_name, recipient.email, subject, body, body_html=body_html, headers=headers, attachments=attachments)

def get_reset_link_body(user: model.User) -> str:
    if False:
        for i in range(10):
            print('nop')
    extra_vars = {'reset_link': get_reset_link(user), 'site_title': config.get('ckan.site_title'), 'site_url': config.get('ckan.site_url'), 'user_name': user.name}
    return render('emails/reset_password.txt', extra_vars)

def get_invite_body(user: model.User, group_dict: Optional[dict[str, Any]]=None, role: Optional[str]=None) -> str:
    if False:
        print('Hello World!')
    extra_vars = {'reset_link': get_reset_link(user), 'site_title': config.get('ckan.site_title'), 'site_url': config.get('ckan.site_url'), 'user_name': user.name}
    if role:
        extra_vars['role_name'] = h.roles_translated().get(role, _(role))
    if group_dict:
        group_type = _('organization') if group_dict['is_organization'] else _('group')
        extra_vars['group_type'] = group_type
        extra_vars['group_title'] = group_dict.get('title')
    return render('emails/invite_user.txt', extra_vars)

def get_reset_link(user: model.User) -> str:
    if False:
        i = 10
        return i + 15
    return h.url_for('user.perform_reset', id=user.id, key=user.reset_key, qualified=True)

def send_reset_link(user: model.User) -> None:
    if False:
        while True:
            i = 10
    create_reset_key(user)
    body = get_reset_link_body(user)
    extra_vars = {'site_title': config.get('ckan.site_title')}
    subject = render('emails/reset_password_subject.txt', extra_vars)
    subject = subject.split('\n')[0]
    mail_user(user, subject, body)

def send_invite(user: model.User, group_dict: Optional[dict[str, Any]]=None, role: Optional[str]=None) -> None:
    if False:
        while True:
            i = 10
    create_reset_key(user)
    body = get_invite_body(user, group_dict, role)
    extra_vars = {'site_title': config.get('ckan.site_title')}
    subject = render('emails/invite_user_subject.txt', extra_vars)
    subject = subject.split('\n')[0]
    mail_user(user, subject, body)

def create_reset_key(user: model.User):
    if False:
        for i in range(10):
            print('nop')
    user.reset_key = make_key()
    model.repo.commit_and_remove()

def make_key():
    if False:
        for i in range(10):
            print('nop')
    return codecs.encode(os.urandom(16), 'hex').decode()

def verify_reset_link(user: model.User, key: Optional[str]) -> bool:
    if False:
        print('Hello World!')
    if not key:
        return False
    if not user.reset_key or len(user.reset_key) < 5:
        return False
    return key.strip() == user.reset_key