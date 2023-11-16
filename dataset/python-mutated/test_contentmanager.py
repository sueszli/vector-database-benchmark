import unittest
from test.test_email import TestEmailBase, parameterize
import textwrap
from email import policy
from email.message import EmailMessage
from email.contentmanager import ContentManager, raw_data_manager

@parameterize
class TestContentManager(TestEmailBase):
    policy = policy.default
    message = EmailMessage
    get_key_params = {'full_type': (1, 'text/plain'), 'maintype_only': (2, 'text'), 'null_key': (3, '')}

    def get_key_as_get_content_key(self, order, key):
        if False:
            while True:
                i = 10

        def foo_getter(msg, foo=None):
            if False:
                while True:
                    i = 10
            bar = msg['X-Bar-Header']
            return (foo, bar)
        cm = ContentManager()
        cm.add_get_handler(key, foo_getter)
        m = self._make_message()
        m['Content-Type'] = 'text/plain'
        m['X-Bar-Header'] = 'foo'
        self.assertEqual(cm.get_content(m, foo='bar'), ('bar', 'foo'))

    def get_key_as_get_content_key_order(self, order, key):
        if False:
            return 10

        def bar_getter(msg):
            if False:
                return 10
            return msg['X-Bar-Header']

        def foo_getter(msg):
            if False:
                for i in range(10):
                    print('nop')
            return msg['X-Foo-Header']
        cm = ContentManager()
        cm.add_get_handler(key, foo_getter)
        for (precedence, key) in self.get_key_params.values():
            if precedence > order:
                cm.add_get_handler(key, bar_getter)
        m = self._make_message()
        m['Content-Type'] = 'text/plain'
        m['X-Bar-Header'] = 'bar'
        m['X-Foo-Header'] = 'foo'
        self.assertEqual(cm.get_content(m), 'foo')

    def test_get_content_raises_if_unknown_mimetype_and_no_default(self):
        if False:
            i = 10
            return i + 15
        cm = ContentManager()
        m = self._make_message()
        m['Content-Type'] = 'text/plain'
        with self.assertRaisesRegex(KeyError, 'text/plain'):
            cm.get_content(m)

    class BaseThing(str):
        pass
    baseobject_full_path = __name__ + '.' + 'TestContentManager.BaseThing'

    class Thing(BaseThing):
        pass
    testobject_full_path = __name__ + '.' + 'TestContentManager.Thing'
    set_key_params = {'type': (0, Thing), 'full_path': (1, testobject_full_path), 'qualname': (2, 'TestContentManager.Thing'), 'name': (3, 'Thing'), 'base_type': (4, BaseThing), 'base_full_path': (5, baseobject_full_path), 'base_qualname': (6, 'TestContentManager.BaseThing'), 'base_name': (7, 'BaseThing'), 'str_type': (8, str), 'str_full_path': (9, 'builtins.str'), 'str_name': (10, 'str'), 'null_key': (11, None)}

    def set_key_as_set_content_key(self, order, key):
        if False:
            while True:
                i = 10

        def foo_setter(msg, obj, foo=None):
            if False:
                while True:
                    i = 10
            msg['X-Foo-Header'] = foo
            msg.set_payload(obj)
        cm = ContentManager()
        cm.add_set_handler(key, foo_setter)
        m = self._make_message()
        msg_obj = self.Thing()
        cm.set_content(m, msg_obj, foo='bar')
        self.assertEqual(m['X-Foo-Header'], 'bar')
        self.assertEqual(m.get_payload(), msg_obj)

    def set_key_as_set_content_key_order(self, order, key):
        if False:
            return 10

        def foo_setter(msg, obj):
            if False:
                while True:
                    i = 10
            msg['X-FooBar-Header'] = 'foo'
            msg.set_payload(obj)

        def bar_setter(msg, obj):
            if False:
                i = 10
                return i + 15
            msg['X-FooBar-Header'] = 'bar'
        cm = ContentManager()
        cm.add_set_handler(key, foo_setter)
        for (precedence, key) in self.get_key_params.values():
            if precedence > order:
                cm.add_set_handler(key, bar_setter)
        m = self._make_message()
        msg_obj = self.Thing()
        cm.set_content(m, msg_obj)
        self.assertEqual(m['X-FooBar-Header'], 'foo')
        self.assertEqual(m.get_payload(), msg_obj)

    def test_set_content_raises_if_unknown_type_and_no_default(self):
        if False:
            while True:
                i = 10
        cm = ContentManager()
        m = self._make_message()
        msg_obj = self.Thing()
        with self.assertRaisesRegex(KeyError, self.testobject_full_path):
            cm.set_content(m, msg_obj)

    def test_set_content_raises_if_called_on_multipart(self):
        if False:
            i = 10
            return i + 15
        cm = ContentManager()
        m = self._make_message()
        m['Content-Type'] = 'multipart/foo'
        with self.assertRaises(TypeError):
            cm.set_content(m, 'test')

    def test_set_content_calls_clear_content(self):
        if False:
            while True:
                i = 10
        m = self._make_message()
        m['Content-Foo'] = 'bar'
        m['Content-Type'] = 'text/html'
        m['To'] = 'test'
        m.set_payload('abc')
        cm = ContentManager()
        cm.add_set_handler(str, lambda *args, **kw: None)
        m.set_content('xyz', content_manager=cm)
        self.assertIsNone(m['Content-Foo'])
        self.assertIsNone(m['Content-Type'])
        self.assertEqual(m['To'], 'test')
        self.assertIsNone(m.get_payload())

@parameterize
class TestRawDataManager(TestEmailBase):
    policy = policy.default.clone(max_line_length=60, content_manager=raw_data_manager)
    message = EmailMessage

    def test_get_text_plain(self):
        if False:
            for i in range(10):
                print('nop')
        m = self._str_msg(textwrap.dedent('            Content-Type: text/plain\n\n            Basic text.\n            '))
        self.assertEqual(raw_data_manager.get_content(m), 'Basic text.\n')

    def test_get_text_html(self):
        if False:
            while True:
                i = 10
        m = self._str_msg(textwrap.dedent('            Content-Type: text/html\n\n            <p>Basic text.</p>\n            '))
        self.assertEqual(raw_data_manager.get_content(m), '<p>Basic text.</p>\n')

    def test_get_text_plain_latin1(self):
        if False:
            return 10
        m = self._bytes_msg(textwrap.dedent('            Content-Type: text/plain; charset=latin1\n\n            Basìc tëxt.\n            ').encode('latin1'))
        self.assertEqual(raw_data_manager.get_content(m), 'Basìc tëxt.\n')

    def test_get_text_plain_latin1_quoted_printable(self):
        if False:
            return 10
        m = self._str_msg(textwrap.dedent('            Content-Type: text/plain; charset="latin-1"\n            Content-Transfer-Encoding: quoted-printable\n\n            Bas=ECc t=EBxt.\n            '))
        self.assertEqual(raw_data_manager.get_content(m), 'Basìc tëxt.\n')

    def test_get_text_plain_utf8_base64(self):
        if False:
            for i in range(10):
                print('nop')
        m = self._str_msg(textwrap.dedent('            Content-Type: text/plain; charset="utf8"\n            Content-Transfer-Encoding: base64\n\n            QmFzw6xjIHTDq3h0Lgo=\n            '))
        self.assertEqual(raw_data_manager.get_content(m), 'Basìc tëxt.\n')

    def test_get_text_plain_bad_utf8_quoted_printable(self):
        if False:
            print('Hello World!')
        m = self._str_msg(textwrap.dedent('            Content-Type: text/plain; charset="utf8"\n            Content-Transfer-Encoding: quoted-printable\n\n            Bas=c3=acc t=c3=abxt=fd.\n            '))
        self.assertEqual(raw_data_manager.get_content(m), 'Basìc tëxt�.\n')

    def test_get_text_plain_bad_utf8_quoted_printable_ignore_errors(self):
        if False:
            for i in range(10):
                print('nop')
        m = self._str_msg(textwrap.dedent('            Content-Type: text/plain; charset="utf8"\n            Content-Transfer-Encoding: quoted-printable\n\n            Bas=c3=acc t=c3=abxt=fd.\n            '))
        self.assertEqual(raw_data_manager.get_content(m, errors='ignore'), 'Basìc tëxt.\n')

    def test_get_text_plain_utf8_base64_recoverable_bad_CTE_data(self):
        if False:
            while True:
                i = 10
        m = self._str_msg(textwrap.dedent('            Content-Type: text/plain; charset="utf8"\n            Content-Transfer-Encoding: base64\n\n            QmFzw6xjIHTDq3h0Lgoÿ=\n            '))
        self.assertEqual(raw_data_manager.get_content(m, errors='ignore'), 'Basìc tëxt.\n')

    def test_get_text_invalid_keyword(self):
        if False:
            print('Hello World!')
        m = self._str_msg(textwrap.dedent('            Content-Type: text/plain\n\n            Basic text.\n            '))
        with self.assertRaises(TypeError):
            raw_data_manager.get_content(m, foo='ignore')

    def test_get_non_text(self):
        if False:
            return 10
        template = textwrap.dedent('            Content-Type: {}\n            Content-Transfer-Encoding: base64\n\n            Ym9ndXMgZGF0YQ==\n            ')
        for maintype in 'audio image video application'.split():
            with self.subTest(maintype=maintype):
                m = self._str_msg(template.format(maintype + '/foo'))
                self.assertEqual(raw_data_manager.get_content(m), b'bogus data')

    def test_get_non_text_invalid_keyword(self):
        if False:
            for i in range(10):
                print('nop')
        m = self._str_msg(textwrap.dedent('            Content-Type: image/jpg\n            Content-Transfer-Encoding: base64\n\n            Ym9ndXMgZGF0YQ==\n            '))
        with self.assertRaises(TypeError):
            raw_data_manager.get_content(m, errors='ignore')

    def test_get_raises_on_multipart(self):
        if False:
            return 10
        m = self._str_msg(textwrap.dedent('            Content-Type: multipart/mixed; boundary="==="\n\n            --===\n            --===--\n            '))
        with self.assertRaises(KeyError):
            raw_data_manager.get_content(m)

    def test_get_message_rfc822_and_external_body(self):
        if False:
            i = 10
            return i + 15
        template = textwrap.dedent('            Content-Type: message/{}\n\n            To: foo@example.com\n            From: bar@example.com\n            Subject: example\n\n            an example message\n            ')
        for subtype in 'rfc822 external-body'.split():
            with self.subTest(subtype=subtype):
                m = self._str_msg(template.format(subtype))
                sub_msg = raw_data_manager.get_content(m)
                self.assertIsInstance(sub_msg, self.message)
                self.assertEqual(raw_data_manager.get_content(sub_msg), 'an example message\n')
                self.assertEqual(sub_msg['to'], 'foo@example.com')
                self.assertEqual(sub_msg['from'].addresses[0].username, 'bar')

    def test_get_message_non_rfc822_or_external_body_yields_bytes(self):
        if False:
            for i in range(10):
                print('nop')
        m = self._str_msg(textwrap.dedent('            Content-Type: message/partial\n\n            To: foo@example.com\n            From: bar@example.com\n            Subject: example\n\n            The real body is in another message.\n            '))
        self.assertEqual(raw_data_manager.get_content(m)[:10], b'To: foo@ex')

    def test_set_text_plain(self):
        if False:
            while True:
                i = 10
        m = self._make_message()
        content = 'Simple message.\n'
        raw_data_manager.set_content(m, content)
        self.assertEqual(str(m), textwrap.dedent('            Content-Type: text/plain; charset="utf-8"\n            Content-Transfer-Encoding: 7bit\n\n            Simple message.\n            '))
        self.assertEqual(m.get_payload(decode=True).decode('utf-8'), content)
        self.assertEqual(m.get_content(), content)

    def test_set_text_plain_null(self):
        if False:
            i = 10
            return i + 15
        m = self._make_message()
        content = ''
        raw_data_manager.set_content(m, content)
        self.assertEqual(str(m), textwrap.dedent('            Content-Type: text/plain; charset="utf-8"\n            Content-Transfer-Encoding: 7bit\n\n\n            '))
        self.assertEqual(m.get_payload(decode=True).decode('utf-8'), '\n')
        self.assertEqual(m.get_content(), '\n')

    def test_set_text_html(self):
        if False:
            for i in range(10):
                print('nop')
        m = self._make_message()
        content = '<p>Simple message.</p>\n'
        raw_data_manager.set_content(m, content, subtype='html')
        self.assertEqual(str(m), textwrap.dedent('            Content-Type: text/html; charset="utf-8"\n            Content-Transfer-Encoding: 7bit\n\n            <p>Simple message.</p>\n            '))
        self.assertEqual(m.get_payload(decode=True).decode('utf-8'), content)
        self.assertEqual(m.get_content(), content)

    def test_set_text_charset_latin_1(self):
        if False:
            i = 10
            return i + 15
        m = self._make_message()
        content = 'Simple message.\n'
        raw_data_manager.set_content(m, content, charset='latin-1')
        self.assertEqual(str(m), textwrap.dedent('            Content-Type: text/plain; charset="iso-8859-1"\n            Content-Transfer-Encoding: 7bit\n\n            Simple message.\n            '))
        self.assertEqual(m.get_payload(decode=True).decode('utf-8'), content)
        self.assertEqual(m.get_content(), content)

    def test_set_text_plain_long_line_heuristics(self):
        if False:
            i = 10
            return i + 15
        m = self._make_message()
        content = 'Simple but long message that is over 78 characters long to force transfer encoding.\n'
        raw_data_manager.set_content(m, content)
        self.assertEqual(str(m), textwrap.dedent('            Content-Type: text/plain; charset="utf-8"\n            Content-Transfer-Encoding: quoted-printable\n\n            Simple but long message that is over 78 characters long to =\n            force transfer encoding.\n            '))
        self.assertEqual(m.get_payload(decode=True).decode('utf-8'), content)
        self.assertEqual(m.get_content(), content)

    def test_set_text_short_line_minimal_non_ascii_heuristics(self):
        if False:
            while True:
                i = 10
        m = self._make_message()
        content = "et là il est monté sur moi et il commence à m'éto.\n"
        raw_data_manager.set_content(m, content)
        self.assertEqual(bytes(m), textwrap.dedent('            Content-Type: text/plain; charset="utf-8"\n            Content-Transfer-Encoding: 8bit\n\n            et là il est monté sur moi et il commence à m\'éto.\n            ').encode('utf-8'))
        self.assertEqual(m.get_payload(decode=True).decode('utf-8'), content)
        self.assertEqual(m.get_content(), content)

    def test_set_text_long_line_minimal_non_ascii_heuristics(self):
        if False:
            return 10
        m = self._make_message()
        content = "j'ai un problème de python. il est sorti de son vivarium.  et là il est monté sur moi et il commence à m'éto.\n"
        raw_data_manager.set_content(m, content)
        self.assertEqual(bytes(m), textwrap.dedent('            Content-Type: text/plain; charset="utf-8"\n            Content-Transfer-Encoding: quoted-printable\n\n            j\'ai un probl=C3=A8me de python. il est sorti de son vivari=\n            um.  et l=C3=A0 il est mont=C3=A9 sur moi et il commence =\n            =C3=A0 m\'=C3=A9to.\n            ').encode('utf-8'))
        self.assertEqual(m.get_payload(decode=True).decode('utf-8'), content)
        self.assertEqual(m.get_content(), content)

    def test_set_text_11_lines_long_line_minimal_non_ascii_heuristics(self):
        if False:
            while True:
                i = 10
        m = self._make_message()
        content = '\n' * 10 + "j'ai un problème de python. il est sorti de son vivarium.  et là il est monté sur moi et il commence à m'éto.\n"
        raw_data_manager.set_content(m, content)
        self.assertEqual(bytes(m), textwrap.dedent('            Content-Type: text/plain; charset="utf-8"\n            Content-Transfer-Encoding: quoted-printable\n            ' + '\n' * 10 + "\n            j'ai un probl=C3=A8me de python. il est sorti de son vivari=\n            um.  et l=C3=A0 il est mont=C3=A9 sur moi et il commence =\n            =C3=A0 m'=C3=A9to.\n            ").encode('utf-8'))
        self.assertEqual(m.get_payload(decode=True).decode('utf-8'), content)
        self.assertEqual(m.get_content(), content)

    def test_set_text_maximal_non_ascii_heuristics(self):
        if False:
            while True:
                i = 10
        m = self._make_message()
        content = 'áàäéèęöő.\n'
        raw_data_manager.set_content(m, content)
        self.assertEqual(bytes(m), textwrap.dedent('            Content-Type: text/plain; charset="utf-8"\n            Content-Transfer-Encoding: 8bit\n\n            áàäéèęöő.\n            ').encode('utf-8'))
        self.assertEqual(m.get_payload(decode=True).decode('utf-8'), content)
        self.assertEqual(m.get_content(), content)

    def test_set_text_11_lines_maximal_non_ascii_heuristics(self):
        if False:
            for i in range(10):
                print('nop')
        m = self._make_message()
        content = '\n' * 10 + 'áàäéèęöő.\n'
        raw_data_manager.set_content(m, content)
        self.assertEqual(bytes(m), textwrap.dedent('            Content-Type: text/plain; charset="utf-8"\n            Content-Transfer-Encoding: 8bit\n            ' + '\n' * 10 + '\n            áàäéèęöő.\n            ').encode('utf-8'))
        self.assertEqual(m.get_payload(decode=True).decode('utf-8'), content)
        self.assertEqual(m.get_content(), content)

    def test_set_text_long_line_maximal_non_ascii_heuristics(self):
        if False:
            for i in range(10):
                print('nop')
        m = self._make_message()
        content = 'áàäéèęöőáàäéèęöőáàäéèęöőáàäéèęöőáàäéèęöőáàäéèęöőáàäéèęöőáàäéèęöőáàäéèęöőáàäéèęöőáàäéèęöőáàäéèęöő.\n'
        raw_data_manager.set_content(m, content)
        self.assertEqual(bytes(m), textwrap.dedent('            Content-Type: text/plain; charset="utf-8"\n            Content-Transfer-Encoding: base64\n\n            w6HDoMOkw6nDqMSZw7bFkcOhw6DDpMOpw6jEmcO2xZHDocOgw6TDqcOoxJnD\n            tsWRw6HDoMOkw6nDqMSZw7bFkcOhw6DDpMOpw6jEmcO2xZHDocOgw6TDqcOo\n            xJnDtsWRw6HDoMOkw6nDqMSZw7bFkcOhw6DDpMOpw6jEmcO2xZHDocOgw6TD\n            qcOoxJnDtsWRw6HDoMOkw6nDqMSZw7bFkcOhw6DDpMOpw6jEmcO2xZHDocOg\n            w6TDqcOoxJnDtsWRLgo=\n            ').encode('utf-8'))
        self.assertEqual(m.get_payload(decode=True).decode('utf-8'), content)
        self.assertEqual(m.get_content(), content)

    def test_set_text_11_lines_long_line_maximal_non_ascii_heuristics(self):
        if False:
            return 10
        m = self._make_message()
        content = '\n' * 10 + 'áàäéèęöőáàäéèęöőáàäéèęöőáàäéèęöőáàäéèęöőáàäéèęöőáàäéèęöőáàäéèęöőáàäéèęöőáàäéèęöőáàäéèęöőáàäéèęöő.\n'
        raw_data_manager.set_content(m, '\n' * 10 + 'áàäéèęöőáàäéèęöőáàäéèęöőáàäéèęöőáàäéèęöőáàäéèęöőáàäéèęöőáàäéèęöőáàäéèęöőáàäéèęöőáàäéèęöőáàäéèęöő.\n')
        self.assertEqual(bytes(m), textwrap.dedent('            Content-Type: text/plain; charset="utf-8"\n            Content-Transfer-Encoding: quoted-printable\n            ' + '\n' * 10 + '\n            =C3=A1=C3=A0=C3=A4=C3=A9=C3=A8=C4=99=C3=B6=C5=91=C3=A1=C3=\n            =A0=C3=A4=C3=A9=C3=A8=C4=99=C3=B6=C5=91=C3=A1=C3=A0=C3=A4=\n            =C3=A9=C3=A8=C4=99=C3=B6=C5=91=C3=A1=C3=A0=C3=A4=C3=A9=C3=\n            =A8=C4=99=C3=B6=C5=91=C3=A1=C3=A0=C3=A4=C3=A9=C3=A8=C4=99=\n            =C3=B6=C5=91=C3=A1=C3=A0=C3=A4=C3=A9=C3=A8=C4=99=C3=B6=C5=\n            =91=C3=A1=C3=A0=C3=A4=C3=A9=C3=A8=C4=99=C3=B6=C5=91=C3=A1=\n            =C3=A0=C3=A4=C3=A9=C3=A8=C4=99=C3=B6=C5=91=C3=A1=C3=A0=C3=\n            =A4=C3=A9=C3=A8=C4=99=C3=B6=C5=91=C3=A1=C3=A0=C3=A4=C3=A9=\n            =C3=A8=C4=99=C3=B6=C5=91=C3=A1=C3=A0=C3=A4=C3=A9=C3=A8=C4=\n            =99=C3=B6=C5=91=C3=A1=C3=A0=C3=A4=C3=A9=C3=A8=C4=99=C3=B6=\n            =C5=91.\n            ').encode('utf-8'))
        self.assertEqual(m.get_payload(decode=True).decode('utf-8'), content)
        self.assertEqual(m.get_content(), content)

    def test_set_text_non_ascii_with_cte_7bit_raises(self):
        if False:
            return 10
        m = self._make_message()
        with self.assertRaises(UnicodeError):
            raw_data_manager.set_content(m, 'áàäéèęöő.\n', cte='7bit')

    def test_set_text_non_ascii_with_charset_ascii_raises(self):
        if False:
            while True:
                i = 10
        m = self._make_message()
        with self.assertRaises(UnicodeError):
            raw_data_manager.set_content(m, 'áàäéèęöő.\n', charset='ascii')

    def test_set_text_non_ascii_with_cte_7bit_and_charset_ascii_raises(self):
        if False:
            i = 10
            return i + 15
        m = self._make_message()
        with self.assertRaises(UnicodeError):
            raw_data_manager.set_content(m, 'áàäéèęöő.\n', cte='7bit', charset='ascii')

    def test_set_message(self):
        if False:
            while True:
                i = 10
        m = self._make_message()
        m['Subject'] = 'Forwarded message'
        content = self._make_message()
        content['To'] = 'python@vivarium.org'
        content['From'] = 'police@monty.org'
        content['Subject'] = 'get back in your box'
        content.set_content('Or face the comfy chair.')
        raw_data_manager.set_content(m, content)
        self.assertEqual(str(m), textwrap.dedent('            Subject: Forwarded message\n            Content-Type: message/rfc822\n            Content-Transfer-Encoding: 8bit\n\n            To: python@vivarium.org\n            From: police@monty.org\n            Subject: get back in your box\n            Content-Type: text/plain; charset="utf-8"\n            Content-Transfer-Encoding: 7bit\n            MIME-Version: 1.0\n\n            Or face the comfy chair.\n            '))
        payload = m.get_payload(0)
        self.assertIsInstance(payload, self.message)
        self.assertEqual(str(payload), str(content))
        self.assertIsInstance(m.get_content(), self.message)
        self.assertEqual(str(m.get_content()), str(content))

    def test_set_message_with_non_ascii_and_coercion_to_7bit(self):
        if False:
            return 10
        m = self._make_message()
        m['Subject'] = 'Escape report'
        content = self._make_message()
        content['To'] = 'police@monty.org'
        content['From'] = 'victim@monty.org'
        content['Subject'] = 'Help'
        content.set_content("j'ai un problème de python. il est sorti de son vivarium.")
        raw_data_manager.set_content(m, content)
        self.assertEqual(bytes(m), textwrap.dedent('            Subject: Escape report\n            Content-Type: message/rfc822\n            Content-Transfer-Encoding: 8bit\n\n            To: police@monty.org\n            From: victim@monty.org\n            Subject: Help\n            Content-Type: text/plain; charset="utf-8"\n            Content-Transfer-Encoding: 8bit\n            MIME-Version: 1.0\n\n            j\'ai un problème de python. il est sorti de son vivarium.\n            ').encode('utf-8'))
        self.assertEqual(m.as_string(maxheaderlen=self.policy.max_line_length), textwrap.dedent('            Subject: Escape report\n            Content-Type: message/rfc822\n            Content-Transfer-Encoding: 8bit\n\n            To: police@monty.org\n            From: victim@monty.org\n            Subject: Help\n            Content-Type: text/plain; charset="utf-8"\n            Content-Transfer-Encoding: base64\n            MIME-Version: 1.0\n\n            aidhaSB1biBwcm9ibMOobWUgZGUgcHl0aG9uLiBpbCBlc3Qgc29ydGkgZGUgc29uIHZpdmFyaXVt\n            Lgo=\n            '))
        self.assertIsInstance(m.get_content(), self.message)
        self.assertEqual(str(m.get_content()), str(content))

    def test_set_message_invalid_cte_raises(self):
        if False:
            print('Hello World!')
        m = self._make_message()
        content = self._make_message()
        for cte in 'quoted-printable base64'.split():
            for subtype in 'rfc822 external-body'.split():
                with self.subTest(cte=cte, subtype=subtype):
                    with self.assertRaises(ValueError) as ar:
                        m.set_content(content, subtype, cte=cte)
                    exc = str(ar.exception)
                    self.assertIn(cte, exc)
                    self.assertIn(subtype, exc)
        subtype = 'external-body'
        for cte in '8bit binary'.split():
            with self.subTest(cte=cte, subtype=subtype):
                with self.assertRaises(ValueError) as ar:
                    m.set_content(content, subtype, cte=cte)
                exc = str(ar.exception)
                self.assertIn(cte, exc)
                self.assertIn(subtype, exc)

    def test_set_image_jpg(self):
        if False:
            while True:
                i = 10
        for content in (b'bogus content', bytearray(b'bogus content'), memoryview(b'bogus content')):
            with self.subTest(content=content):
                m = self._make_message()
                raw_data_manager.set_content(m, content, 'image', 'jpeg')
                self.assertEqual(str(m), textwrap.dedent('                    Content-Type: image/jpeg\n                    Content-Transfer-Encoding: base64\n\n                    Ym9ndXMgY29udGVudA==\n                    '))
                self.assertEqual(m.get_payload(decode=True), content)
                self.assertEqual(m.get_content(), content)

    def test_set_audio_aif_with_quoted_printable_cte(self):
        if False:
            while True:
                i = 10
        m = self._make_message()
        content = b'b\xffgus\tcon\nt\rent ' + b'z' * 100
        m.set_content(content, 'audio', 'aif', cte='quoted-printable')
        self.assertEqual(bytes(m), textwrap.dedent('            Content-Type: audio/aif\n            Content-Transfer-Encoding: quoted-printable\n            MIME-Version: 1.0\n\n            b=FFgus=09con=0At=0Dent=20zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz=\n            zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz').encode('latin-1'))
        self.assertEqual(m.get_payload(decode=True), content)
        self.assertEqual(m.get_content(), content)

    def test_set_video_mpeg_with_binary_cte(self):
        if False:
            for i in range(10):
                print('nop')
        m = self._make_message()
        content = b'b\xffgus\tcon\nt\rent ' + b'z' * 100
        m.set_content(content, 'video', 'mpeg', cte='binary')
        self.assertEqual(bytes(m), textwrap.dedent('            Content-Type: video/mpeg\n            Content-Transfer-Encoding: binary\n            MIME-Version: 1.0\n\n            ').encode('ascii') + b'b\xffgus\tcon\nt\nent zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz' + b'zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz')
        self.assertEqual(m.get_payload(decode=True), content)
        self.assertEqual(m.get_content(), content)

    def test_set_application_octet_stream_with_8bit_cte(self):
        if False:
            print('Hello World!')
        m = self._make_message()
        content = b'b\xffgus\tcon\nt\rent\n' + b'z' * 60 + b'\n'
        m.set_content(content, 'application', 'octet-stream', cte='8bit')
        self.assertEqual(bytes(m), textwrap.dedent('            Content-Type: application/octet-stream\n            Content-Transfer-Encoding: 8bit\n            MIME-Version: 1.0\n\n            ').encode('ascii') + b'b\xffgus\tcon\nt\nent\n' + b'zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz\n')
        self.assertEqual(m.get_payload(decode=True), content)
        self.assertEqual(m.get_content(), content)

    def test_set_headers_from_header_objects(self):
        if False:
            for i in range(10):
                print('nop')
        m = self._make_message()
        content = 'Simple message.\n'
        header_factory = self.policy.header_factory
        raw_data_manager.set_content(m, content, headers=(header_factory('To', 'foo@example.com'), header_factory('From', 'foo@example.com'), header_factory('Subject', "I'm talking to myself.")))
        self.assertEqual(str(m), textwrap.dedent('            Content-Type: text/plain; charset="utf-8"\n            To: foo@example.com\n            From: foo@example.com\n            Subject: I\'m talking to myself.\n            Content-Transfer-Encoding: 7bit\n\n            Simple message.\n            '))

    def test_set_headers_from_strings(self):
        if False:
            for i in range(10):
                print('nop')
        m = self._make_message()
        content = 'Simple message.\n'
        raw_data_manager.set_content(m, content, headers=('X-Foo-Header: foo', 'X-Bar-Header: bar'))
        self.assertEqual(str(m), textwrap.dedent('            Content-Type: text/plain; charset="utf-8"\n            X-Foo-Header: foo\n            X-Bar-Header: bar\n            Content-Transfer-Encoding: 7bit\n\n            Simple message.\n            '))

    def test_set_headers_with_invalid_duplicate_string_header_raises(self):
        if False:
            while True:
                i = 10
        m = self._make_message()
        content = 'Simple message.\n'
        with self.assertRaisesRegex(ValueError, 'Content-Type'):
            raw_data_manager.set_content(m, content, headers=('Content-Type: foo/bar',))

    def test_set_headers_with_invalid_duplicate_header_header_raises(self):
        if False:
            while True:
                i = 10
        m = self._make_message()
        content = 'Simple message.\n'
        header_factory = self.policy.header_factory
        with self.assertRaisesRegex(ValueError, 'Content-Type'):
            raw_data_manager.set_content(m, content, headers=(header_factory('Content-Type', ' foo/bar'),))

    def test_set_headers_with_defective_string_header_raises(self):
        if False:
            i = 10
            return i + 15
        m = self._make_message()
        content = 'Simple message.\n'
        with self.assertRaisesRegex(ValueError, 'a@fairly@@invalid@address'):
            raw_data_manager.set_content(m, content, headers=('To: a@fairly@@invalid@address',))
            print(m['To'].defects)

    def test_set_headers_with_defective_header_header_raises(self):
        if False:
            print('Hello World!')
        m = self._make_message()
        content = 'Simple message.\n'
        header_factory = self.policy.header_factory
        with self.assertRaisesRegex(ValueError, 'a@fairly@@invalid@address'):
            raw_data_manager.set_content(m, content, headers=(header_factory('To', 'a@fairly@@invalid@address'),))
            print(m['To'].defects)

    def test_set_disposition_inline(self):
        if False:
            while True:
                i = 10
        m = self._make_message()
        m.set_content('foo', disposition='inline')
        self.assertEqual(m['Content-Disposition'], 'inline')

    def test_set_disposition_attachment(self):
        if False:
            for i in range(10):
                print('nop')
        m = self._make_message()
        m.set_content('foo', disposition='attachment')
        self.assertEqual(m['Content-Disposition'], 'attachment')

    def test_set_disposition_foo(self):
        if False:
            while True:
                i = 10
        m = self._make_message()
        m.set_content('foo', disposition='foo')
        self.assertEqual(m['Content-Disposition'], 'foo')

    def test_set_filename(self):
        if False:
            print('Hello World!')
        m = self._make_message()
        m.set_content('foo', filename='bar.txt')
        self.assertEqual(m['Content-Disposition'], 'attachment; filename="bar.txt"')

    def test_set_filename_and_disposition_inline(self):
        if False:
            return 10
        m = self._make_message()
        m.set_content('foo', disposition='inline', filename='bar.txt')
        self.assertEqual(m['Content-Disposition'], 'inline; filename="bar.txt"')

    def test_set_non_ascii_filename(self):
        if False:
            while True:
                i = 10
        m = self._make_message()
        m.set_content('foo', filename='ábárî.txt')
        self.assertEqual(bytes(m), textwrap.dedent('            Content-Type: text/plain; charset="utf-8"\n            Content-Transfer-Encoding: 7bit\n            Content-Disposition: attachment;\n             filename*=utf-8\'\'%C3%A1b%C3%A1r%C3%AE.txt\n            MIME-Version: 1.0\n\n            foo\n            ').encode('ascii'))

    def test_set_content_bytes_cte_7bit(self):
        if False:
            print('Hello World!')
        m = self._make_message()
        m.set_content(b'ASCII-only message.\n', maintype='application', subtype='octet-stream', cte='7bit')
        self.assertEqual(str(m), textwrap.dedent('            Content-Type: application/octet-stream\n            Content-Transfer-Encoding: 7bit\n            MIME-Version: 1.0\n\n            ASCII-only message.\n            '))
    content_object_params = {'text_plain': ('content', ()), 'text_html': ('content', ('html',)), 'application_octet_stream': (b'content', ('application', 'octet_stream')), 'image_jpeg': (b'content', ('image', 'jpeg')), 'message_rfc822': (message(), ()), 'message_external_body': (message(), ('external-body',))}

    def content_object_as_header_receiver(self, obj, mimetype):
        if False:
            while True:
                i = 10
        m = self._make_message()
        m.set_content(obj, *mimetype, headers=('To: foo@example.com', 'From: bar@simple.net'))
        self.assertEqual(m['to'], 'foo@example.com')
        self.assertEqual(m['from'], 'bar@simple.net')

    def content_object_as_disposition_inline_receiver(self, obj, mimetype):
        if False:
            i = 10
            return i + 15
        m = self._make_message()
        m.set_content(obj, *mimetype, disposition='inline')
        self.assertEqual(m['Content-Disposition'], 'inline')

    def content_object_as_non_ascii_filename_receiver(self, obj, mimetype):
        if False:
            i = 10
            return i + 15
        m = self._make_message()
        m.set_content(obj, *mimetype, disposition='inline', filename='bár.txt')
        self.assertEqual(m['Content-Disposition'], 'inline; filename="bár.txt"')
        self.assertEqual(m.get_filename(), 'bár.txt')
        self.assertEqual(m['Content-Disposition'].params['filename'], 'bár.txt')

    def content_object_as_cid_receiver(self, obj, mimetype):
        if False:
            print('Hello World!')
        m = self._make_message()
        m.set_content(obj, *mimetype, cid='some_random_stuff')
        self.assertEqual(m['Content-ID'], 'some_random_stuff')

    def content_object_as_params_receiver(self, obj, mimetype):
        if False:
            for i in range(10):
                print('nop')
        m = self._make_message()
        params = {'foo': 'bár', 'abc': 'xyz'}
        m.set_content(obj, *mimetype, params=params)
        if isinstance(obj, str):
            params['charset'] = 'utf-8'
        self.assertEqual(m['Content-Type'].params, params)
if __name__ == '__main__':
    unittest.main()