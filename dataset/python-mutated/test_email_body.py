import base64
import os
import frappe
from frappe import safe_decode
from frappe.email.doctype.email_queue.email_queue import QueueBuilder, SendMailContext
from frappe.email.email_body import get_email, get_header, inline_style_in_html, replace_filename_with_cid
from frappe.email.receive import Email
from frappe.tests.utils import FrappeTestCase

class TestEmailBody(FrappeTestCase):

    def setUp(self):
        if False:
            return 10
        email_html = '\n<div>\n\t<h3>Hey John Doe!</h3>\n\t<p>This is embedded image you asked for</p>\n\t<img embed="assets/frappe/images/frappe-favicon.svg" />\n</div>\n'
        email_text = '\nHey John Doe!\nThis is the text version of this email\n'
        img_path = os.path.abspath('assets/frappe/images/frappe-favicon.svg')
        with open(img_path, 'rb') as f:
            img_content = f.read()
            img_base64 = base64.b64encode(img_content).decode()
        self.img_base64 = fixed_column_width(img_base64, 76)
        self.email_string = get_email(recipients=['test@example.com'], sender='me@example.com', subject='Test Subject', content=email_html, text_content=email_text).as_string().replace('\r\n', '\n')

    def test_prepare_message_returns_already_encoded_string(self):
        if False:
            i = 10
            return i + 15
        uni_chr1 = chr(40960)
        uni_chr2 = chr(1972)
        QueueBuilder(recipients=['test@example.com'], sender='me@example.com', subject='Test Subject', message=f'<h1>{uni_chr1}abcd{uni_chr2}</h1>', text_content='whatever').process()
        queue_doc = frappe.get_last_doc('Email Queue')
        mail_ctx = SendMailContext(queue_doc=queue_doc)
        result = mail_ctx.build_message(recipient_email='test@test.com')
        self.assertTrue(b'<h1>=EA=80=80abcd=DE=B4</h1>' in result)

    def test_prepare_message_returns_cr_lf(self):
        if False:
            i = 10
            return i + 15
        QueueBuilder(recipients=['test@example.com'], sender='me@example.com', subject='Test Subject', message='<h1>\n this is a test of newlines\n' + '</h1>', text_content='whatever').process()
        queue_doc = frappe.get_last_doc('Email Queue')
        mail_ctx = SendMailContext(queue_doc=queue_doc)
        result = safe_decode(mail_ctx.build_message(recipient_email='test@test.com'))
        self.assertTrue(result.count('\n') == result.count('\r'))

    def test_image(self):
        if False:
            for i in range(10):
                print('nop')
        img_signature = '\nContent-Type: image/svg+xml\nMIME-Version: 1.0\nContent-Transfer-Encoding: base64\nContent-Disposition: inline; filename="frappe-favicon.svg"\n'
        self.assertTrue(img_signature in self.email_string)
        self.assertTrue(self.img_base64 in self.email_string)

    def test_text_content(self):
        if False:
            return 10
        text_content = '\nContent-Type: text/plain; charset="utf-8"\nMIME-Version: 1.0\nContent-Transfer-Encoding: quoted-printable\n\n\nHey John Doe!\nThis is the text version of this email\n'
        self.assertTrue(text_content in self.email_string)

    def test_email_content(self):
        if False:
            i = 10
            return i + 15
        html_head = '\nContent-Type: text/html; charset="utf-8"\nMIME-Version: 1.0\nContent-Transfer-Encoding: quoted-printable\n\n<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.=\nw3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">\n<html xmlns=3D"http://www.w3.org/1999/xhtml">\n'
        html = '<h3>Hey John Doe!</h3>'
        self.assertTrue(html_head in self.email_string)
        self.assertTrue(html in self.email_string)

    def test_replace_filename_with_cid(self):
        if False:
            while True:
                i = 10
        original_message = '\n\t\t\t<div>\n\t\t\t\t<img embed="assets/frappe/images/frappe-favicon.svg" alt="test" />\n\t\t\t\t<img embed="notexists.jpg" />\n\t\t\t</div>\n\t\t'
        (message, inline_images) = replace_filename_with_cid(original_message)
        processed_message = '\n\t\t\t<div>\n\t\t\t\t<img src="cid:{}" alt="test" />\n\t\t\t\t<img  />\n\t\t\t</div>\n\t\t'.format(inline_images[0].get('content_id'))
        self.assertEqual(message, processed_message)

    def test_inline_styling(self):
        if False:
            print('Hello World!')
        html = '\n<h3>Hi John</h3>\n<p>This is a test email</p>\n'
        transformed_html = '\n<h3>Hi John</h3>\n<p style="margin:1em 0 !important">This is a test email</p>\n'
        self.assertTrue(transformed_html in inline_style_in_html(html))

    def test_email_header(self):
        if False:
            return 10
        email_html = '\n<h3>Hey John Doe!</h3>\n<p>This is embedded image you asked for</p>\n'
        email_string = get_email(recipients=['test@example.com'], sender='me@example.com', subject='Test Subject\u2028, with line break, \nand Line feed \rand carriage return.', content=email_html, header=['Email Title', 'green']).as_string()
        self.assertTrue('<span class=3D"indicator indicator-green"></span>' in email_string)
        self.assertTrue('<span>Email Title</span>' in email_string)
        self.assertIn('Subject: Test Subject, with line break, and Line feed and carriage return.', email_string)

    def test_get_email_header(self):
        if False:
            return 10
        html = get_header(['This is test', 'orange'])
        self.assertTrue('<span class="indicator indicator-orange"></span>' in html)
        self.assertTrue('<span>This is test</span>' in html)
        html = get_header(['This is another test'])
        self.assertTrue('<span>This is another test</span>' in html)
        html = get_header('This is string')
        self.assertTrue('<span>This is string</span>' in html)

    def test_8bit_utf_8_decoding(self):
        if False:
            while True:
                i = 10
        text_content_bytes = b'\xed\x95\x9c\xea\xb8\x80\xe1\xa5\xa1\xe2\x95\xa5\xe0\xba\xaa\xe0\xa4\x8f'
        text_content = text_content_bytes.decode('utf-8')
        content_bytes = b'MIME-Version: 1.0\nContent-Type: text/plain; charset=utf-8\nContent-Disposition: inline\nContent-Transfer-Encoding: 8bit\nFrom: test1_@erpnext.com\nReply-To: test2_@erpnext.com\n' + text_content_bytes
        mail = Email(content_bytes)
        self.assertEqual(mail.text_content, text_content)

    def test_poorly_encoded_messages(self):
        if False:
            i = 10
            return i + 15
        mail = Email.decode_email('=?iso-2022-jp?B?VEFLQVlBTUEgS2FvcnUgWxskQnxiOzMbKEIgGyRCNzAbKEJd?=\n\t<user@example.com>')
        self.assertIn('user@example.com', mail)

def fixed_column_width(string, chunk_size):
    if False:
        i = 10
        return i + 15
    parts = [string[0 + i:chunk_size + i] for i in range(0, len(string), chunk_size)]
    return '\n'.join(parts)