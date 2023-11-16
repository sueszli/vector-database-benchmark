import cgi
import unittest
from odoo.tools import html_sanitize, append_content_to_html, plaintext2html, email_split
from . import test_mail_examples

class TestSanitizer(unittest.TestCase):
    """ Test the html sanitizer that filters html to remove unwanted attributes """

    def test_basic_sanitizer(self):
        if False:
            while True:
                i = 10
        cases = [('yop', '<p>yop</p>'), ('lala<p>yop</p>xxx', '<p>lala</p><p>yop</p>xxx'), ("Merci à l'intérêt pour notre produit.nous vous contacterons bientôt. Merci", u"<p>Merci à l'intérêt pour notre produit.nous vous contacterons bientôt. Merci</p>")]
        for (content, expected) in cases:
            html = html_sanitize(content)
            self.assertEqual(html, expected, 'html_sanitize is broken')

    def test_mako(self):
        if False:
            while True:
                i = 10
        cases = [('<p>Some text</p>\n<% set signup_url = object.get_signup_url() %>\n% if signup_url:\n<p>\n    You can access this document and pay online via our Customer Portal:\n</p>', '<p>Some text</p>\n<% set signup_url = object.get_signup_url() %>\n% if signup_url:\n<p>\n    You can access this document and pay online via our Customer Portal:\n</p>')]
        for (content, expected) in cases:
            html = html_sanitize(content, silent=False)
            self.assertEqual(html, expected, 'html_sanitize: broken mako management')

    def test_evil_malicious_code(self):
        if False:
            for i in range(10):
                print('nop')
        cases = ["<IMG SRC=javascript:alert('XSS')>", '<IMG SRC=&#106;&#97;&#118;&#97;&#115;&#99;&#114;&#105;&#112;&#116;&#58;&#97;&#108;&#101;&#114;&#116;&#40;&#39;&#88;&#83;&#83;&#39;&#41;>', '<IMG SRC=&#x6A&#x61&#x76&#x61&#x73&#x63&#x72&#x69&#x70&#x74&#x3A&#x61&#x6C&#x65&#x72&#x74&#x28&#x27&#x58&#x53&#x53&#x27&#x29>', '<IMG SRC="jav&#x0D;ascript:alert(\'XSS\');">', '<IMG SRC="jav&#x0A;ascript:alert(\'XSS\');">', '<IMG SRC="jav   ascript:alert(\'XSS\');">', '<IMG SRC="jav&#x09;ascript:alert(\'XSS\');">', '<IMG SRC=" &#14;  javascript:alert(\'XSS\');">', '<IMG SRC="javascript:alert(\'XSS\')"', '<IMG """><SCRIPT>alert("XSS")</SCRIPT>">', '<SCRIPT/XSS SRC="http://ha.ckers.org/xss.js"></SCRIPT>', '<SCRIPT/SRC="http://ha.ckers.org/xss.js"></SCRIPT>', '<<SCRIPT>alert("XSS");//<</SCRIPT>', '<SCRIPT SRC=http://ha.ckers.org/xss.js?< B >', '<INPUT TYPE="IMAGE" SRC="javascript:alert(\'XSS\');">', '<BODY BACKGROUND="javascript:alert(\'XSS\')">', '<IMG DYNSRC="javascript:alert(\'XSS\')">', '<IMG LOWSRC="javascript:alert(\'XSS\')">', '<TABLE BACKGROUND="javascript:alert(\'XSS\')">', '<TABLE><TD BACKGROUND="javascript:alert(\'XSS\')">', '<DIV STYLE="background-image: url(javascript:alert(\'XSS\'))">', '<DIV STYLE="background-image:\x075\x072\x06C\x028\'\x06a\x061\x076\x061\x073\x063\x072\x069\x070\x074\x03a\x061\x06c\x065\x072\x074\x028.1027\x058.1053\x053\x027\x029\'\x029">', '<DIV STYLE="background-image: url(&#1;javascript:alert(\'XSS\'))">', '<IMG SRC=\'vbscript:msgbox("XSS")\'>', "<BODY ONLOAD=alert('XSS')>", '<BR SIZE="&{alert(\'XSS\')}\\>', '<LINK REL="stylesheet" HREF="javascript:alert(\'XSS\');">', '<LINK REL="stylesheet" HREF="http://ha.ckers.org/xss.css">', "<STYLE>@import'http://ha.ckers.org/xss.css';</STYLE>", '<META HTTP-EQUIV="Link" Content="<http://ha.ckers.org/xss.css>; REL=stylesheet">', '<STYLE>BODY{-moz-binding:url("http://ha.ckers.org/xssmoz.xml#xss")}</STYLE>', '<IMG STYLE="xss:expr/*XSS*/ession(alert(\'XSS\'))">']
        for content in cases:
            html = html_sanitize(content)
            self.assertNotIn('javascript', html, 'html_sanitize did not remove a malicious javascript')
            self.assertTrue('ha.ckers.org' not in html or 'http://ha.ckers.org/xss.css' in html, 'html_sanitize did not remove a malicious code in %s (%s)' % (content, html))
        content = "<!--[if gte IE 4]><SCRIPT>alert('XSS');</SCRIPT><![endif]-->"
        self.assertEquals(html_sanitize(content, silent=False), '')

    def test_html(self):
        if False:
            return 10
        sanitized_html = html_sanitize(test_mail_examples.MISC_HTML_SOURCE)
        for tag in ['<div', '<b', '<i', '<u', '<strike', '<li', '<blockquote', '<a href']:
            self.assertIn(tag, sanitized_html, 'html_sanitize stripped too much of original html')
        for attr in ['javascript']:
            self.assertNotIn(attr, sanitized_html, 'html_sanitize did not remove enough unwanted attributes')

    def test_sanitize_escape_emails(self):
        if False:
            return 10
        emails = ['Charles <charles.bidule@truc.fr>', "Dupuis <'tr/-: ${dupuis#$'@truc.baz.fr>", 'Technical <service/technical+2@open.com>', 'Div nico <div-nico@open.com>']
        for email in emails:
            self.assertIn(cgi.escape(email), html_sanitize(email), 'html_sanitize stripped emails of original html')

    def test_sanitize_unescape_emails(self):
        if False:
            while True:
                i = 10
        not_emails = ['<blockquote cite="mid:CAEJSRZvWvud8c6Qp=wfNG6O1+wK3i_jb33qVrF7XyrgPNjnyUA@mail.gmail.com" type="cite">cat</blockquote>', '<img alt="@github-login" class="avatar" src="/web/image/pi" height="36" width="36">']
        for email in not_emails:
            sanitized = html_sanitize(email)
            left_part = email.split('>')[0]
            self.assertNotIn(cgi.escape(email), sanitized, 'html_sanitize stripped emails of original html')
            self.assertIn(left_part, sanitized)

    def test_style_parsing(self):
        if False:
            while True:
                i = 10
        test_data = [('<span style="position: fixed; top: 0px; left: 50px; width: 40%; height: 50%; background-color: red;">Coin coin </span>', ['background-color: red', 'Coin coin'], ['position', 'top', 'left']), ('<div style=\'before: "Email Address; coincoin cheval: lapin";  \n   font-size: 30px; max-width: 100%; after: "Not sure\n    \n          this; means: anything ?#ùµ"\n    ; some-property: 2px; top: 3\'>youplaboum</div>', ['font-size: 30px', 'youplaboum'], ['some-property', 'top', 'cheval']), ('<span style="width">Coincoin</span>', [], ['width'])]
        for (test, in_lst, out_lst) in test_data:
            new_html = html_sanitize(test, sanitize_attributes=False, sanitize_style=True, strip_style=False, strip_classes=False)
            for text in in_lst:
                self.assertIn(text, new_html)
            for text in out_lst:
                self.assertNotIn(text, new_html)
        new_html = html_sanitize(test_data[0][0], sanitize_attributes=False, strip_style=True, strip_classes=False)
        self.assertEqual(new_html, u'<span>Coin coin </span>')

    def test_style_class(self):
        if False:
            while True:
                i = 10
        html = html_sanitize(test_mail_examples.REMOVE_CLASS, sanitize_attributes=True, sanitize_style=True, strip_classes=True)
        for ext in test_mail_examples.REMOVE_CLASS_IN:
            self.assertIn(ext, html)
        for ext in test_mail_examples.REMOVE_CLASS_OUT:
            self.assertNotIn(ext, html)

    def test_style_class_only(self):
        if False:
            while True:
                i = 10
        html = html_sanitize(test_mail_examples.REMOVE_CLASS, sanitize_attributes=False, sanitize_style=True, strip_classes=True)
        for ext in test_mail_examples.REMOVE_CLASS_IN:
            self.assertIn(ext, html)
        for ext in test_mail_examples.REMOVE_CLASS_OUT:
            self.assertNotIn(ext, html)

    def test_edi_source(self):
        if False:
            i = 10
            return i + 15
        html = html_sanitize(test_mail_examples.EDI_LIKE_HTML_SOURCE)
        self.assertIn("font-family: 'Lucida Grande', Ubuntu, Arial, Verdana, sans-serif;", html, 'html_sanitize removed valid styling')
        self.assertIn('src="https://www.paypal.com/en_US/i/btn/btn_paynowCC_LG.gif"', html, 'html_sanitize removed valid img')
        self.assertNotIn('</body></html>', html, 'html_sanitize did not remove extra closing tags')

    def test_quote_blockquote(self):
        if False:
            while True:
                i = 10
        html = html_sanitize(test_mail_examples.QUOTE_BLOCKQUOTE)
        for ext in test_mail_examples.QUOTE_BLOCKQUOTE_IN:
            self.assertIn(ext, html)
        for ext in test_mail_examples.QUOTE_BLOCKQUOTE_OUT:
            self.assertIn('<span data-o-mail-quote="1">%s' % cgi.escape(ext.decode('utf-8')), html)

    def test_quote_thunderbird(self):
        if False:
            while True:
                i = 10
        html = html_sanitize(test_mail_examples.QUOTE_THUNDERBIRD_1)
        for ext in test_mail_examples.QUOTE_THUNDERBIRD_1_IN:
            self.assertIn(ext, html)
        for ext in test_mail_examples.QUOTE_THUNDERBIRD_1_OUT:
            self.assertIn('<span data-o-mail-quote="1">%s</span>' % cgi.escape(ext.decode('utf-8')), html)

    def test_quote_hotmail_html(self):
        if False:
            i = 10
            return i + 15
        html = html_sanitize(test_mail_examples.QUOTE_HOTMAIL_HTML)
        for ext in test_mail_examples.QUOTE_HOTMAIL_HTML_IN:
            self.assertIn(ext, html)
        for ext in test_mail_examples.QUOTE_HOTMAIL_HTML_OUT:
            self.assertIn(ext, html)
        html = html_sanitize(test_mail_examples.HOTMAIL_1)
        for ext in test_mail_examples.HOTMAIL_1_IN:
            self.assertIn(ext, html)
        for ext in test_mail_examples.HOTMAIL_1_OUT:
            self.assertIn(ext, html)

    def test_quote_thunderbird_html(self):
        if False:
            while True:
                i = 10
        html = html_sanitize(test_mail_examples.QUOTE_THUNDERBIRD_HTML)
        for ext in test_mail_examples.QUOTE_THUNDERBIRD_HTML_IN:
            self.assertIn(ext, html)
        for ext in test_mail_examples.QUOTE_THUNDERBIRD_HTML_OUT:
            self.assertIn(ext, html)

    def test_quote_basic_text(self):
        if False:
            print('Hello World!')
        test_data = [('This is Sparta!\n--\nAdministrator\n+9988776655', ['This is Sparta!'], ['\n--\nAdministrator\n+9988776655']), ('<p>This is Sparta!\n--\nAdministrator</p>', [], ['\n--\nAdministrator']), ('<p>This is Sparta!<br/>--<br>Administrator</p>', ['This is Sparta!'], []), ('This is Sparta!\n>Ah bon ?\nCertes\n> Chouette !\nClair', ['This is Sparta!', 'Certes', 'Clair'], ['\n>Ah bon ?', '\n> Chouette !'])]
        for (test, in_lst, out_lst) in test_data:
            new_html = html_sanitize(test)
            for text in in_lst:
                self.assertIn(text, new_html)
            for text in out_lst:
                self.assertIn('<span data-o-mail-quote="1">%s</span>' % cgi.escape(text), new_html)

    def test_quote_signature(self):
        if False:
            while True:
                i = 10
        test_data = [('<div>Hello<pre>--<br />Administrator</pre></div>', ['<pre data-o-mail-quote="1">--', '<br data-o-mail-quote="1">'])]
        for (test, in_lst) in test_data:
            new_html = html_sanitize(test)
            for text in in_lst:
                self.assertIn(text, new_html)

    def test_quote_gmail(self):
        if False:
            for i in range(10):
                print('nop')
        html = html_sanitize(test_mail_examples.GMAIL_1)
        for ext in test_mail_examples.GMAIL_1_IN:
            self.assertIn(ext, html)
        for ext in test_mail_examples.GMAIL_1_OUT:
            self.assertIn('<span data-o-mail-quote="1">%s</span>' % cgi.escape(ext), html)

    def test_quote_text(self):
        if False:
            print('Hello World!')
        html = html_sanitize(test_mail_examples.TEXT_1)
        for ext in test_mail_examples.TEXT_1_IN:
            self.assertIn(ext, html)
        for ext in test_mail_examples.TEXT_1_OUT:
            self.assertIn('<span data-o-mail-quote="1">%s</span>' % cgi.escape(ext), html)
        html = html_sanitize(test_mail_examples.TEXT_2)
        for ext in test_mail_examples.TEXT_2_IN:
            self.assertIn(ext, html)
        for ext in test_mail_examples.TEXT_2_OUT:
            self.assertIn('<span data-o-mail-quote="1">%s</span>' % cgi.escape(ext), html)

    def test_quote_bugs(self):
        if False:
            for i in range(10):
                print('nop')
        html = html_sanitize(test_mail_examples.BUG1)
        for ext in test_mail_examples.BUG_1_IN:
            self.assertIn(ext, html)
        for ext in test_mail_examples.BUG_1_OUT:
            self.assertIn('<span data-o-mail-quote="1">%s</span>' % cgi.escape(ext.decode('utf-8')), html)

    def test_misc(self):
        if False:
            for i in range(10):
                print('nop')
        html = html_sanitize('')
        self.assertEqual(html, '')
        html = html_sanitize(False)
        self.assertEqual(html, False)
        html = html_sanitize(u'<?xml version="1.0" encoding="iso-8859-1"?>\n<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"\n         "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">\n<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">\n <head>\n  <title>404 - Not Found</title>\n </head>\n <body>\n  <h1>404 - Not Found</h1>\n </body>\n</html>\n')
        self.assertNotIn('encoding', html)
        self.assertNotIn('<title>404 - Not Found</title>', html)
        self.assertIn('<h1>404 - Not Found</h1>', html)

class TestHtmlTools(unittest.TestCase):
    """ Test some of our generic utility functions about html """

    def test_plaintext2html(self):
        if False:
            while True:
                i = 10
        cases = [('First \nSecond \nThird\n \nParagraph\n\r--\nSignature paragraph', 'div', '<div><p>First <br/>Second <br/>Third</p><p>Paragraph</p><p>--<br/>Signature paragraph</p></div>'), ('First<p>It should be escaped</p>\nSignature', False, '<p>First&lt;p&gt;It should be escaped&lt;/p&gt;<br/>Signature</p>')]
        for (content, container_tag, expected) in cases:
            html = plaintext2html(content, container_tag)
            self.assertEqual(html, expected, 'plaintext2html is broken')

    def test_append_to_html(self):
        if False:
            return 10
        test_samples = [('<!DOCTYPE...><HTML encoding="blah">some <b>content</b></HtMl>', '--\nYours truly', True, True, False, '<!DOCTYPE...><html encoding="blah">some <b>content</b>\n<pre>--\nYours truly</pre>\n</html>'), ('<!DOCTYPE...><HTML encoding="blah">some <b>content</b></HtMl>', '--\nYours truly', True, False, False, '<!DOCTYPE...><html encoding="blah">some <b>content</b>\n<p>--<br/>Yours truly</p>\n</html>'), ('<html><body>some <b>content</b></body></html>', '<!DOCTYPE...>\n<html><body>\n<p>--</p>\n<p>Yours truly</p>\n</body>\n</html>', False, False, False, '<html><body>some <b>content</b>\n\n\n<p>--</p>\n<p>Yours truly</p>\n\n\n</body></html>')]
        for (html, content, plaintext_flag, preserve_flag, container_tag, expected) in test_samples:
            self.assertEqual(append_content_to_html(html, content, plaintext_flag, preserve_flag, container_tag), expected, 'append_content_to_html is broken')

class TestEmailTools(unittest.TestCase):
    """ Test some of our generic utility functions for emails """

    def test_email_split(self):
        if False:
            return 10
        cases = [('John <12345@gmail.com>', ['12345@gmail.com']), ('d@x; 1@2', ['d@x', '1@2']), ("'(ss)' <123@gmail.com>, 'foo' <foo@bar>", ['123@gmail.com', 'foo@bar']), ('"john@gmail.com"<johnny@gmail.com>', ['johnny@gmail.com']), ('"<jg>" <johnny@gmail.com>', ['johnny@gmail.com'])]
        for (text, expected) in cases:
            self.assertEqual(email_split(text), expected, 'email_split is broken')